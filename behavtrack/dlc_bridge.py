"""Minimal local COCO detector export / installed DLC training bridge.

No count label becomes a box or keypoint. Final references never enter training.
Prepared DLC pose projects can use the installed official train_network entrypoint.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import re
import logging
import copy
import csv
import math
import types
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
import cv2
import numpy as np
from .integrity import read_json,atomic_json,digest,utcnow,read_events
from .research_data import latest
from .recording_io import load_times,RecordingVideo
from .activity import heavy


def load_training_image(filepath,color_mode='RGB'):
    """OpenCV's Windows filename reader cannot reliably open Korean paths."""
    if color_mode not in ('RGB','BGR'):raise ValueError('Unsupported training image color mode')
    image=cv2.imdecode(np.fromfile(str(filepath),dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f'학습 이미지를 읽지 못했습니다: {filepath}')
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB) if color_mode=='RGB' else image


def detection_only_annotations(anns,num_joints,num_unique=0):
    """Keep real box-only targets; all pose keypoints remain UNLABELED."""
    n=len(anns);boxes=np.asarray([a['bbox'] for a in anns],dtype=float).reshape(n,4)
    if not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=0):raise ValueError('검출 상자 정답 크기 오류')
    merged={'area':boxes[:,2]*boxes[:,3],
            'category_id':np.array([a['category_id'] for a in anns],dtype=int),
            'iscrowd':np.array([a.get('iscrowd',0) for a in anns],dtype=int),
            'individual_id':np.array([a.get('individual_id',-1) for a in anns],dtype=int)}
    return np.zeros((n,num_joints,3)),np.zeros((num_unique,3)),boxes,merged


def export_detector(entries,destination,reviewer):
    destination=Path(destination)
    groups={}; data={r:{"images":[],"annotations":[],"categories":[{"id":1,"name":"mouse","keypoints":[],"skeleton":[]}]} for r in ("train","development")}
    pending=[]; sources=[]
    for entry in entries:
        folder=Path(entry["session"])
        role=entry["role"]
        if role not in data:
            raise ValueError("검출기 내보내기는 학습/개발만 포함합니다. 최종 평가 자료는 제외하세요.")
        spec=read_json(folder/"protocol.json");info,_=load_times(folder)
        cage=spec["cage"]
        if cage in groups and groups[cage]!=role:
            raise ValueError("동일 케이지가 검출기 학습과 개발 평가에 동시에 들어갑니다.")
        groups[cage]=role
        events=[e for e in latest(read_events(folder/"box_labels.jsonl")) if e["reviewer"]==reviewer]
        if not events:
            raise ValueError(f"{folder}: 직접 작성한 상자 정답이 없습니다.")
        for event in events:
            if event["source_sha256"]!=info["sha256"] or event["purpose"]!="development":
                raise ValueError("상자 정답 원본이 변경됐거나 최종 평가 자료입니다.")
        pending.append((folder,role,events))
        sources.append({"session":str(folder.resolve()),"cage":cage,"role":role,"source_sha256":info["sha256"],"labels_sha256":digest(folder/"box_labels.jsonl")})
    if not all(any(p[1]==role for p in pending) for role in data):
        raise ValueError("검출기 추가 학습에는 서로 다른 케이지의 학습/개발 상자 정답이 필요합니다.")
    if len({s["session"] for s in sources})!=len(sources):
        raise ValueError("중복 촬영 세션을 제거하세요.")
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"images").mkdir();(destination/"annotations").mkdir()
    for folder,role,events in pending:
        cap=RecordingVideo(folder)
        try:
            for e in events:
                cap.set(cv2.CAP_PROP_POS_FRAMES,e["session_frame"]);ok,frame=cap.read()
                if not ok:
                    raise ValueError("상자에 연결된 원본 프레임을 읽지 못했습니다.")
                d=data[role];image_id=len(d["images"])+1
                filename=f"{folder.name}_{e['session_frame']:09d}.png"
                ok,encoded=cv2.imencode(".png",frame)
                if not ok:
                    raise ValueError("검출기 학습 장면 저장 실패")
                (destination/"images"/filename).write_bytes(encoded.tobytes())
                d["images"].append({"id":image_id,"file_name":filename,"width":frame.shape[1],"height":frame.shape[0]})
                for x1,y1,x2,y2 in e["boxes"]:
                    d["annotations"].append({"id":len(d["annotations"])+1,"image_id":image_id,"category_id":1,
                                             "bbox":[x1,y1,x2-x1,y2-y1],"area":(x2-x1)*(y2-y1),"iscrowd":0,
                                             "keypoints":[],"num_keypoints":0})
        finally:
            cap.release()
    for role,d in data.items():
        atomic_json(destination/"annotations"/(role+".json"),d)
    manifest={"schema":1,"task":"detector_boxes","created_utc":utcnow(),"sources":sources,"split_unit":"cage",
              "final_excluded":True,"files":{str(p.relative_to(destination)):digest(p) for p in destination.rglob("*") if p.is_file()}}
    atomic_json(destination/"export.json",manifest)
    return manifest


@heavy
def launch(root,export,config,mode="detector",cancel=None,progress=None,epochs=5):
    root,export,config=Path(root),Path(export),Path(config)
    executable=root/".ai_venv/Scripts/python.exe"
    if not executable.is_file() or not config.is_file():
        raise ValueError("기존 .ai_venv와 DeepLabCut pytorch_config.yaml이 필요합니다. ai_setup.cmd 및 DLC 프로젝트 구성을 확인하세요.")
    manifest=read_json(export/"export.json")
    if manifest.get("task")!="detector_boxes" or mode!="detector":
        raise ValueError("이 내보내기는 검출기용입니다. 포즈 학습에는 DLC에서 사람이 작성한 관절 정답이 필요합니다.")
    for name,sha in manifest["files"].items():
        path=(export/name).resolve()
        if not path.is_relative_to(export.resolve()) or digest(path)!=sha:
            raise ValueError("고정 검출기 학습 자료가 변경됐습니다.")
    candidate=root/"results/dlc_candidates"/("dlc_"+uuid.uuid4().hex)
    shutil.copytree(export,candidate)
    shutil.copyfile(config,candidate/"pytorch_config.yaml")
    initial=root/"models/topviewmouse/detector.pt"
    if not initial.is_file():
        raise ValueError("기존 기준 검출기 detector.pt가 없습니다. 새 구조를 임의로 학습하지 않습니다.")
    shutil.copyfile(initial,candidate/"initial_detector.pt")
    status={"state":"training","validated":False,"mode":mode,"created_utc":utcnow(),"export_sha256":digest(export/"export.json"),"epochs":epochs}
    atomic_json(candidate/"status.json",status)
    cancel=cancel or threading.Event()
    env=os.environ.copy();env["PYTHONUTF8"]="1";env["DLClight"]="True"
    env['TORCH_HOME']=str(root/'models/torch')
    progress=progress or (lambda *a:None)
    progress(0,epochs,'공식 DLC 검출기 추가 학습 준비 중')
    with (candidate/"training.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([str(executable),"-u","-m","behavtrack.dlc_bridge","worker",str(candidate),"--epochs",str(epochs)],
                                 cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        while process.poll() is None:
            try:
                tail=(candidate/'training.log').read_text(encoding='utf-8',errors='replace')[-16000:]
                matches=re.findall(r'Epoch\s+(\d+)/(\d+)',tail)
                if matches:
                    epoch,total=map(int,matches[-1]);progress(epoch,total,f'검출기 추가 학습 · {epoch}/{total} epoch')
            except OSError:pass
            if cancel.wait(.25):
                process.terminate();process.wait(timeout=20)
                status.update(state="cancelled",error="사용자가 취소했습니다. 기존 모델은 유지됩니다.")
                break
        else:
            status.update(state="candidate" if process.returncode==0 else "failed",returncode=process.returncode)
    status["snapshots"]={p.name:digest(p) for p in candidate.glob("snapshot*.pt")}
    if status["state"]=="candidate" and not status["snapshots"]:
        status.update(state="failed",error="학습 스냅샷이 생성되지 않았습니다. training.log를 확인하세요.")
    atomic_json(candidate/"status.json",status)
    return dict(status,path=str(candidate))


def detector_worker(folder,epochs):
    """Installed DLC 3.0.2 API; only Task.DETECT sees the bbox-only reference."""
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    from deeplabcut.pose_estimation_pytorch import COCOLoader
    from deeplabcut.pose_estimation_pytorch.config import PoseConfig
    from deeplabcut.pose_estimation_pytorch.apis.training import train
    import torch
    torch.set_num_threads(4)
    from deeplabcut.pose_estimation_pytorch.data import dataset as dlc_dataset
    # Worker-local adapter; never modify the installed DLC package.
    dlc_dataset.load_image=load_training_image
    from deeplabcut.pose_estimation_pytorch.task import Task
    folder=Path(folder)
    cfg=PoseConfig.from_yaml(folder/"pytorch_config.yaml")
    if cfg.detector is None or cfg.detector.model.get("variant")!="fasterrcnn_mobilenet_v3_large_fpn":
        raise ValueError("기존 기준 검출기와 같은 fasterrcnn_mobilenet_v3_large_fpn 설정이 필요합니다.")
    parts=list(cfg.metadata.bodyparts)
    # COCO loader requires the schema's part list even for detection; zero visibility
    # explicitly denotes UNLABELED keypoints and is never sent to pose training.
    for role in ("train","development"):
        path=folder/"annotations"/(role+".json")
        data=read_json(path);data["categories"][0]["keypoints"]=parts
        for a in data["annotations"]:
            a["keypoints"]=[0,0,0]*len(parts)
        atomic_json(path,data)
    cfg.detector.train_settings.epochs=int(epochs)
    cfg.detector.train_settings.dataloader_workers=0
    cfg.detector.train_settings.batch_size=1
    cfg.detector.runner.snapshots.save_epochs=1
    cfg.detector.runner.snapshot_prefix="snapshot-detector"
    cfg.to_yaml(folder/"pytorch_config.yaml")
    class DetectionLoader(COCOLoader):
        # DLC's pose-oriented COCO loader removes animals without visible
        # keypoints and recomputes their boxes. Detection labels already contain
        # the real boxes; preserve them without inventing pose annotations.
        def load_data(self,mode='train'):
            data=copy.deepcopy(self.train_json if mode=='train' else self.test_json)
            data=self.validate_images(self.validate_categories(data));counts={}
            for a in data['annotations']:
                i=a['image_id'];idx=counts.get(i,0);counts[i]=idx+1
                a['individual']=f'individual{idx}';a['keypoints']=np.asarray(a['keypoints'],float)
                a['bbox']=np.asarray(a['bbox'],float)
            return data
        def create_dataset(self,transform=None,mode='train',task=Task.DETECT):
            if task!=Task.DETECT:raise ValueError('상자 정답은 검출 학습에만 사용할 수 있습니다.')
            ds=super().create_dataset(transform,mode,task)
            def extract(instance,anns,image_shape):
                return detection_only_annotations(anns,instance.parameters.num_joints,instance.parameters.num_unique_bpts)
            ds.extract_keypoints_and_bboxes=types.MethodType(extract,ds)
            return ds
    loader=DetectionLoader(project_root=folder,model_config=folder/"pytorch_config.yaml",train_json_filename="train.json",test_json_filename="development.json")
    from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
    audit_transform=build_transforms(loader.model_cfg['detector']['data']['inference'])
    audit={}
    for mode in ('train','test'):
        ds=loader.create_dataset(transform=audit_transform,mode=mode,task=Task.DETECT);sample=ds[0]
        count=int(np.count_nonzero(np.asarray(sample['annotations']['boxes'])[:,2]>0))
        expected=sum(a['image_id']==sample['image_id'] for a in (loader.train_json if mode=='train' else loader.test_json)['annotations'])
        if count!=expected or not count:raise ValueError('실제 검출 상자 정답이 학습 입력에서 누락됐습니다.')
        audit[mode]={'images':len(ds),'first_image_boxes':count,'expected_boxes':expected,
                     'pose_keypoints_labeled':int(np.count_nonzero(sample['annotations']['keypoints'][...,2]))}
    atomic_json(folder/'input_audit.json',audit)
    logging.info('Detection input audit: %s',audit)
    if epochs==0:return
    atomic_json(folder/"runtime.json",{p:importlib.metadata.version(p) for p in ("deeplabcut","torch","torchvision")})
    train(loader,loader.model_cfg["detector"],Task.DETECT,device="cpu",snapshot_path=folder/"initial_detector.pt")
    stats=folder/'learning_stats_detector.csv'
    if not stats.is_file():raise ValueError('학습 손실 기록이 없습니다.')
    with stats.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)<epochs or any(not math.isfinite(float(r['losses/train.total_loss'])) for r in rows):
        raise ValueError('학습 손실이 유한하지 않거나 epoch 기록이 부족합니다. 후보로 등록하지 않습니다.')


def import_detector(snapshot,destination):
    snapshot,destination=Path(snapshot),Path(destination)
    status=read_json(snapshot.parent/"status.json")
    if status.get("state")!="candidate" or status.get("snapshots",{}).get(snapshot.name)!=digest(snapshot):
        raise ValueError("완료된 DLC 후보 학습에서 나온 해시가 일치하는 snapshot*.pt를 선택하세요.")
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(snapshot,destination/"detector.pt")
    shutil.copyfile(snapshot.parent/"pytorch_config.yaml",destination/"pytorch_config.yaml")
    metadata={"state":"candidate","validated":False,"sha256":digest(destination/"detector.pt"),
              "source":str(snapshot.resolve()),"training":status,"export":read_json(snapshot.parent/"export.json")}
    atomic_json(destination/"detector.json",metadata)
    return dict(metadata,path=str(destination))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["worker"]);parser.add_argument("folder");parser.add_argument("--epochs",type=int,default=5)
    args=parser.parse_args();detector_worker(args.folder,args.epochs)
