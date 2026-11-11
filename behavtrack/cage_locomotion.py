"""Conservative, identity-free cage locomotion from adjacent centroid detections.

Only intervals with every expected animal separately detected and unambiguous
mutual nearest matches contribute. Never bridge a gap or infer persistent IDs.
"""
from __future__ import annotations
import base64
import csv
import html
import json
import math
import threading
import subprocess
import uuid
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from .activity import heavy
from .integrity import atomic_json, digest, json_digest, read_json, utcnow
from .recording_io import load_times, RecordingVideo
from .locomotion_config import VERSION, settings
from . import cage_activity as activity

FIELDS=['pair','start_s','end_s','duration_s','start_frame','end_frame','status','reason','distance','mean_speed','links_json']
REASONS=dict(valid='유효',not_analyzed='미분석',missing_frame='프레임 부족',count_mismatch='마릿수 불일치',
             invalid_boxes='검출 좌표 오류',overlap='동물 겹침',ambiguous_match='연결 불확실',
             implausible_step='큰 좌표 점프',box_shape_change='검출 상자 급변',time_gap='촬영 시각 공백',
             camera_shift='고정 배경 이동',illumination_change='조명 변화',large_scene_change='큰 장면 변화')

def geometry(config, size, roi):
    w,h=size;c=config['calibration']
    polygon=np.array(c['points'] if c else [[roi[0],roi[1]],[roi[2],roi[1]],[roi[2],roi[3]],[roi[0],roi[3]]],np.float32)*[w,h]
    transform=cv2.getPerspectiveTransform(polygon.astype(np.float32),np.array([[0,0],[c['width_cm'],0],
                     [c['width_cm'],c['height_cm']],[0,c['height_cm']]],np.float32)) if c else None
    return polygon.astype(np.float32),transform

def candidates(prediction, expected, config, polygon, transform):
    if prediction.get('box_format')!='xyxy':return None,'invalid_boxes'
    try:
        boxes=np.asarray(prediction['boxes'],float).reshape(-1,4);scores=np.asarray(prediction['scores'],float).reshape(-1)
    except (KeyError,ValueError,TypeError):return None,'invalid_boxes'
    if len(boxes)!=len(scores) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():return None,'invalid_boxes'
    centers=np.asarray(prediction.get('centers',(boxes[:,:2]+boxes[:,2:])/2),float)
    if centers.shape!=(len(boxes),2) or not np.isfinite(centers).all():return None,'invalid_boxes'
    centers=centers[scores>=config['min_score']];boxes=boxes[scores>=config['min_score']]
    if np.any(boxes[:,2:]<=boxes[:,:2]):return None,'invalid_boxes'
    keep=[cv2.pointPolygonTest(polygon,tuple(map(float,p)),False)>=0 for p in centers]
    boxes=boxes[np.array(keep,dtype=bool)];centers=centers[np.array(keep,dtype=bool)]
    if len(boxes)!=expected:return None,'count_mismatch'
    area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
    for i in range(expected):
        for j in range(i):
            intersection=np.maximum(0,np.minimum(boxes[i,2:],boxes[j,2:])-np.maximum(boxes[i,:2],boxes[j,:2])).prod()
            if intersection/(area[i]+area[j]-intersection)>config['max_iou']:return None,'overlap'
    xy=cv2.perspectiveTransform(centers.astype(np.float32).reshape(1,-1,2),transform)[0].astype(float) if transform is not None else centers
    if not np.isfinite(xy).all():return None,'invalid_boxes'
    return dict(boxes=boxes.tolist(),centers=centers.tolist(),xy=xy.tolist(),areas=area.tolist()),'valid'

class BrightForeground:
    """Explicit fixed-threshold white-animal segmentation; no learned confidence."""
    def __init__(self,config,polygon,size):
        self.config=config;self.polygon=polygon;self.mask=np.zeros((size[1],size[0]),np.uint8)
        cv2.fillPoly(self.mask,[polygon.astype(np.int32)],255)
        self.runtime={'name':'bright_foreground','reference_point':'binary_body_centroid','confidence':'not probabilistic'}
    def predict(self,frame,fid):
        c=self.config;gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);sat=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1]
        mask=((gray>=c['white_threshold'])&(sat<=c['max_saturation'])&(self.mask>0)).astype(np.uint8)*255
        k=int(c['opening'])
        if k>1:mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
        count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
        total=frame.shape[0]*frame.shape[1];boxes=[];points=[]
        for stat,p in zip(stats[1:],centers[1:]):
            x,y,w,h,area=map(int,stat)
            if c['min_area_fraction']*total<=area<=c['max_area_fraction']*total and area/(w*h)>=.3:
                boxes.append([x,y,x+w,y+h]);points.append(p.tolist())
        return dict(boxes=boxes,scores=[1.]*len(boxes),centers=points,box_format='xyxy',method='bright_foreground',confidence='not_probabilistic')
    def close(self):pass

def link(previous,current,dt,config,size):
    """All-cage interval or unknown. Detector order never determines matching."""
    a,b=np.asarray(previous['xy']),np.asarray(current['xy']);n=len(a)
    costs=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    forward=costs.argmin(axis=1);backward=costs.argmin(axis=0)
    if len(set(forward.tolist()))!=n or any(backward[j]!=i for i,j in enumerate(forward)):
        return None,'ambiguous_match'
    if n>1:
        for values in (costs,costs.T):
            ordered=np.sort(values,axis=1)
            if np.any(ordered[:,1]<=1e-9) or np.any(ordered[:,0]>config['ambiguity_ratio']*ordered[:,1]):
                return None,'ambiguous_match'
    steps=costs[np.arange(n),forward]
    limit=config['max_speed_cm_s']*dt if config['calibration'] else math.hypot(*size)*config['max_step_diagonal']
    if np.any(steps>limit):return None,'implausible_step'
    ratios=np.asarray(current['areas'])[forward]/np.asarray(previous['areas'])
    if np.any(ratios>2.5) or np.any(ratios<.4):return None,'box_shape_change'
    if config['calibration']:steps=np.where(steps<=config['deadband_cm'],0,steps)
    links=[dict(start=previous['centers'][i],end=current['centers'][int(j)],distance=float(steps[i])) for i,j in enumerate(forward)]
    return dict(distance=float(steps.sum()),mean_speed=float(steps.sum()/(n*dt)),links=links),'valid'

def summarize(rows, animals):
    valid=[r for r in rows if r['status']=='valid'];known=sum(r['duration_s'] for r in valid);total=sum(r['duration_s'] for r in rows)
    distance=sum(r['distance'] for r in valid) if valid else None
    return dict(pairs=len(rows),valid_pairs=len(valid),unknown_pairs=len(rows)-len(valid),observed_s=total,valid_s=known,
                unknown_s=total-known,coverage_pct=known/total*100 if total else None,
                distance_total=distance,distance_per_animal=distance/animals if distance is not None else None,
                mean_speed=distance/(known*animals) if known else None,
                unknown_reasons=dict(Counter(r['reason'] for r in rows if r['status']!='valid')))

def time_bins(rows,seconds,animals):
    bins={}
    for row in rows:
        start=row['start_s']
        while start<row['end_s']-1e-9:
            index=math.floor((start+1e-9)/seconds);stop=min(row['end_s'],(index+1)*seconds)
            part=dict(row,start_s=start,end_s=stop,duration_s=stop-start)
            if row['distance'] is not None:part['distance']=row['distance']*(stop-start)/row['duration_s']
            bins.setdefault(index,[]).append(part);start=stop
    return [dict(summarize(v,animals),start_s=v[0]['start_s'],end_s=v[-1]['end_s']) for _,v in sorted(bins.items())]

def overlay(frame,point,row,polygon):
    view=frame.copy();cv2.polylines(view,[polygon.astype(np.int32)],True,(80,210,130),2)
    if point:
        for box,p in zip(point['boxes'],point['centers']):
            x,y,xx,yy=map(round,box);cv2.rectangle(view,(x,y),(xx,yy),(90,210,255),2);cv2.circle(view,tuple(map(round,p)),4,(0,220,255),-1)
    if row and row.get('links'):
        for edge in row['links']:cv2.arrowedLine(view,tuple(map(round,edge['start'])),tuple(map(round,edge['end'])),(255,100,40),2,tipLength=.2)
    text=f"{row['end_s']:.2f}s | {row['reason']}" if row else 'first sample'
    cv2.rectangle(view,(0,0),(min(view.shape[1],700),32),(20,25,20),-1)
    cv2.putText(view,text,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    width=min(960,view.shape[1]);return cv2.resize(view,(width,round(view.shape[0]*width/view.shape[1])))

def browser_video(path):
    """Use the existing server's bundled FFmpeg for browser-compatible H.264."""
    root=Path(__file__).resolve().parents[1]
    binaries=list((root/'.ai_venv/Lib/site-packages/imageio_ffmpeg/binaries').glob('ffmpeg*.exe'))
    if not binaries:return False
    temp=path.with_name('tracking_browser.mp4')
    result=subprocess.run([str(binaries[0]),'-hide_banner','-loglevel','error','-y','-i',str(path),'-an',
                           '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-preset','fast','-crf','23',
                           '-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],capture_output=True,timeout=120,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode or not temp.is_file():
        temp.unlink(missing_ok=True);return False
    temp.replace(path);return True

@heavy
def analyze(folder,values,executable,model_path,cancel=None,progress=None,predictor_factory=None):
    from .research import WorkerPredictor
    factory=predictor_factory or WorkerPredictor
    folder=Path(folder);cancel=cancel or threading.Event();progress=progress or (lambda *a:None)
    info,times=load_times(folder);protocol=read_json(folder/'protocol.json');config=settings(values)
    scene=activity.settings(dict(values,interval_s=config['interval_s'],time_tolerance_s=config['time_tolerance_s']))
    meta=activity.metadata(values,protocol);size=protocol['criteria']['frame_size'];animals=int(meta['animals'])
    if not 1<=animals<=20:raise ValueError('마릿수 범위를 확인하세요.')
    start=float(values.get('start_s',0));end=float(values.get('end_s') if values.get('end_s') is not None else times[-1])
    if not math.isfinite(start+end) or start<0 or end>times[-1]+1e-6 or end-start<config['interval_s']:
        raise ValueError('영상 범위 안에서 관찰 시작·종료를 지정하세요.')
    plan=activity._plan(times,start,end,config);polygon,transform=geometry(config,size,scene['roi'])
    boundaries=[float(times[p['frame']]) if p['frame'] is not None else p['target'] for p in plan]
    rows=[dict(pair=i-1,start_s=boundaries[i-1],end_s=boundaries[i],duration_s=boundaries[i]-boundaries[i-1],
               start_frame=plan[i-1]['frame'],end_frame=plan[i]['frame'],status='unknown',reason='not_analyzed',
               distance=None,mean_speed=None,links=[]) for i in range(1,len(plan))]
    run=folder/'locomotion'/('loco_'+uuid.uuid4().hex[:12]);run.mkdir(parents=True)
    manifest=dict(schema=1,id=run.name,method=VERSION,state='running',created_utc=utcnow(),source_session=folder.name,
                  metadata=meta,settings=config,scene_settings=scene,unit='cm' if transform is not None else 'px',
                  window=dict(start_s=start,end_s=plan[-1]['target'],requested_end_s=end),source_sha256=info['sha256'],
                  recorded_start_utc=info.get('started_utc'),timestamp_source=info.get('timestamp_source'),
                  validated=False,scope='cage_complete_intervals_no_persistent_identity',error=None,examples=[])
    manifest['comparison_signature']=json_digest(dict(config=config,scene=scene,animals=animals,size=size,duration=round(plan[-1]['target']-start,6)))
    manifest['code_sha256']=digest(Path(__file__));manifest['model_sha256']=digest(Path(model_path)) if config['detector']=='dlc' and Path(model_path).is_file() else None
    atomic_json(run/'locomotion.json',manifest);cap=RecordingVideo(folder,verify=False);predictor=None;video=None
    decoded=-1;previous=None;previous_gray=None;previous_ref=None;previous_reason='missing_frame'
    try:
        progress(0,len(plan),'마우스 검출 모델 준비 중')
        if cancel.is_set():manifest['state']='cancelled'
        else:
            if config['detector']=='bright_foreground':predictor=BrightForeground(config,polygon,size)
            else:predictor=factory(executable,animals,cancel,detector_path=model_path) if factory is WorkerPredictor else factory(executable,animals,cancel)
            manifest['runtime']=predictor.runtime
            with (run/'detections.jsonl').open('x',encoding='utf-8') as log:
                for index,point in enumerate(plan):
                    if cancel.is_set():manifest['state']='cancelled';break
                    fid=point['frame'];current=None;gray=reference=None;reason='missing_frame';prediction=None
                    if fid is not None:
                        while decoded<fid:
                            ok,frame=cap.read();decoded+=1
                            if not ok:raise ValueError('추적할 원본 프레임을 읽지 못했습니다.')
                        prediction=predictor.predict(frame,fid);current,reason=candidates(prediction,animals,config,polygon,transform)
                        gray=activity.gray_roi(frame,scene['roi'],scene)
                        if scene['reference_roi']:reference=activity.gray_roi(frame,scene['reference_roi'],scene)
                    log.write(json.dumps(dict(frame=fid,target_s=point['target'],reason=reason,prediction=prediction),allow_nan=False)+'\n');log.flush()
                    row=rows[index-1] if index else None
                    if row is not None:
                        row['reason']=previous_reason if previous is None else reason
                        dt=row['duration_s']
                        if dt<=0 or abs(dt-config['interval_s'])>2*config['time_tolerance_s']+1e-6:row['reason']='time_gap'
                        elif previous is not None and current is not None:
                            measurement,_=activity.pair_measure(previous_gray,gray,scene,previous_ref,reference)
                            linked,link_reason=link(previous,current,dt,config,size)
                            row['reason']=measurement['reason'] if measurement['status']!='valid' else link_reason
                            if row['reason']=='valid':row.update(status='valid',**linked)
                    if fid is not None:
                        view=overlay(frame,current,row,polygon)
                        if video is None:
                            video=cv2.VideoWriter(str(run/'tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1/config['interval_s'],(view.shape[1],view.shape[0]))
                            if not video.isOpened():raise ValueError('추적 확인 영상을 저장하지 못했습니다.')
                        video.write(view)
                        if index%max(1,len(plan)//8)==0 and len(manifest['examples'])<9:
                            name=f'example_{index:06d}.jpg';cv2.imencode('.jpg',view)[1].tofile(str(run/name))
                            manifest['examples'].append(dict(file=name,time_s=boundaries[index],reason=row['reason'] if row else reason))
                    previous,previous_reason,previous_gray,previous_ref=current,reason,gray,reference
                    progress(index+1,len(plan),'마우스 중심점 추적 · 이동거리·속도 계산 중')
                else:manifest['state']='complete'
    except (Exception,SystemExit) as exc:manifest.update(state='cancelled' if cancel.is_set() else 'failed',error=str(exc))
    finally:
        cap.release()
        if predictor:predictor.close()
        if video:video.release()
        if video:
            try:manifest['browser_video_compatible']=browser_video(run/'tracking.mp4')
            except (OSError,subprocess.TimeoutExpired):manifest['browser_video_compatible']=False
        with (run/'pairs.csv').open('x',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            for row in rows:writer.writerow({**{k:v for k,v in row.items() if k!='links'},'links_json':json.dumps(row['links'])})
        manifest.update(finished_utc=utcnow(),summary=summarize(rows,animals),bins=time_bins(rows,config['bin_seconds'],animals))
        manifest['artifacts_sha256']={name:digest(run/name) for name in ('pairs.csv','detections.jsonl','tracking.mp4') if (run/name).is_file()}
        atomic_json(run/'locomotion.json',manifest)
    try:write_report(run,manifest)
    except Exception as exc:manifest['report_error']=str(exc);atomic_json(run/'locomotion.json',manifest)
    return dict(kind='cage_locomotion',state=manifest['state'],session=folder.name,run=run.name,path=str(run),report=manifest,error=manifest['error'])

def read_report(run,verify_source=True):
    run=Path(run);report=read_json(run/'locomotion.json')
    for name,sha in report['artifacts_sha256'].items():
        if name not in ('pairs.csv','detections.jsonl','tracking.mp4') or digest(run/name)!=sha:raise ValueError('추적 결과 파일이 변경됐습니다.')
    with (run/'pairs.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        for k in ('start_s','end_s','duration_s','distance','mean_speed'):row[k]=float(row[k]) if row[k] else None
        if row['duration_s']<=0 or not math.isclose(row['duration_s'],row['end_s']-row['start_s'],abs_tol=1e-8):raise ValueError('추적 표본 시각이 잘못됐습니다.')
        if row['status'] not in ('valid','unknown') or row['status']=='valid' and (row['distance'] is None or not math.isfinite(row['distance']) or row['distance']<0):raise ValueError('추적 표본 값이 잘못됐습니다.')
    n=report['metadata']['animals']
    if json_digest(summarize(rows,n))!=json_digest(report['summary']) or json_digest(time_bins(rows,report['settings']['bin_seconds'],n))!=json_digest(report['bins']):raise ValueError('추적 요약이 CSV와 다릅니다.')
    if verify_source and load_times(run.parent.parent)[0]['sha256']!=report['source_sha256']:raise ValueError('추적 원본이 분석 당시와 다릅니다.')
    return report

def comparison(reports,mode='water',metric='mean_speed',min_coverage_pct=80.):
    if metric not in ('mean_speed','distance_per_animal'):raise ValueError('평균 속도 또는 마리당 이동거리를 선택하세요.')
    threshold=float(min_coverage_pct)
    if not math.isfinite(threshold) or not 80<=threshold<=100:raise ValueError('비교에 필요한 추적 비율은 80~100%로 지정하세요.')
    eligible=[];excluded=[]
    for r in reports:
        reason='실측 길이 보정 없음' if r['unit']!='cm' else '유효 추적 비율 부족' if (r['summary']['coverage_pct'] or 0)<threshold else None
        if reason:excluded.append(dict(session=r['source_session'],reason=reason));continue
        eligible.append(r)
    result=activity.comparison(eligible,mode,metric=metric,unit='cm/s' if metric=='mean_speed' else 'cm',kind='cage_locomotion_comparison')
    result.update(excluded=excluded+result['excluded'],metric=metric,min_coverage_pct=threshold,
                  criteria='동일한 보정·촬영·관찰 길이·마릿수·추적 조건. 누락 구간을 0이나 전체 시간으로 보간하지 않음.')
    for s in result['strata']:
        s['interpretation']='케이지별 중심점 기반 이동 추정치의 기술적 비교입니다. 가림이 운동성과 관련되면 누락 편향이 생길 수 있습니다. 전체 시간 이동거리나 물의 인과 효과를 확정하지 않습니다.'
    return result

def write_report(run,r):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    bins=r['bins'];unit=r['unit'];s=r['summary'];fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
    axes[0].plot([b['end_s'] for b in bins],[b['mean_speed'] if b['mean_speed'] is not None else np.nan for b in bins],marker='o',color='#24775e')
    axes[0].set_ylabel(f'Mean speed ({unit}/s)');axes[1].bar([b['start_s'] for b in bins],[b['coverage_pct'] or 0 for b in bins],width=[b['end_s']-b['start_s'] for b in bins],align='edge',color='#adcdbc')
    axes[1].set_ylim(0,105);axes[1].set_ylabel('Tracked (%)');axes[1].set_xlabel('Recording time (s)');fig.tight_layout();fig.savefig(run/'chart.png',dpi=130);plt.close(fig)
    def f(v):return '계산 불가' if v is None else f'{v:.3f}'
    image=base64.b64encode((run/'chart.png').read_bytes()).decode();m=r['metadata']
    notes=[f"이동거리 합계는 {m['animals']}마리가 모두 추적된 {s['valid_s']:.2f}초 구간의 합입니다. 미확인 {s['unknown_s']:.2f}초를 0으로 넣거나 전체 시간으로 확대하지 않습니다.",
           '평균 속도 = 유효 구간의 전체 마우스 이동거리 합계 ÷ (마릿수 × 유효 추적 시간). 마리당 이동거리는 합계 ÷ 마릿수이며 개체별 측정값이 아닙니다.',
           '검출된 몸 중심점의 2차원 표본 경로 추정치입니다. 검출 흔들림은 거리를 늘리고, 표본 사이 굽은 경로·가림은 거리를 줄일 수 있습니다. 실제 정답으로 정확도를 검증하지 않았습니다.',
           '네 모서리 보정은 바닥 평면의 원근만 보정합니다. 렌즈 왜곡·몸 높이·일어서기·등반은 보정하지 않습니다. 고정된 상부 촬영이 필요합니다.',
           '추적 비율은 정확도가 아닙니다. 겹침·누락·불확실 연결은 판정 불가이며 가림 전후의 개체 ID를 연결하지 않습니다.',
           '추적 확인 영상은 선택 표본을 순서대로 재생합니다. 누락 프레임이 있으면 실제 재생 시간과 다를 수 있으므로 영상에 표시된 촬영 시각을 확인하세요.',
           f"거리 잡음 제외 기준: {r['settings']['deadband_cm']} cm/표본. 속도 상한 {r['settings']['max_speed_cm_s']} cm/s는 기술적 점프 제외 기준이며 정상 운동성 기준이 아닙니다."]
    if unit=='px':notes.insert(0,'실측 길이 미보정: 픽셀 결과만 제공합니다. cm 결과 및 그룹·물 종류별 비교에서 제외됩니다.')
    if (s['coverage_pct'] or 0)<80:notes.insert(0,'유효 추적 비율이 80% 미만이므로 그룹·물 종류별 비교에서 제외됩니다. 아래 값은 확인된 일부 구간의 탐색 결과입니다.')
    if r['settings']['calibration'] and r['settings']['calibration']['lengths_approximate']:notes.insert(0,'케이지 길이는 사용자가 제공한 대략값입니다. 거리·속도도 그 길이를 기준으로 한 추정치입니다.')
    notes.insert(0,'검출 방식: '+('밝은 흰 마우스와 어두운 고정 배경의 명암 분리. 털색·조명·반사에 따라 누락·오검출이 생길 수 있습니다.' if r['settings']['detector']=='bright_foreground' else 'DeepLabCut 검출 상자 중심. 실제 카메라 영상의 검출 상태를 확인해야 합니다.'))
    r['notes']=notes;atomic_json(run/'locomotion.json',r)
    reasons=' · '.join(f'{REASONS.get(k,k)} {v}구간' for k,v in s['unknown_reasons'].items())
    page=f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>케이지 이동거리·속도</title><style>body{{font:16px 'Malgun Gothic',sans-serif;max-width:1000px;margin:30px auto;line-height:1.8;background:#f5f7f3;padding:24px}}section{{background:white;padding:24px;border-radius:12px;margin:16px 0}}img{{max-width:100%}}strong{{color:#16765b}}</style><h1>케이지 이동거리·평균 속도 추정</h1><p>{html.escape(m['group'])} · {html.escape(m['cage'])} · {m['animals']}마리 · {activity.WATER[m['water_type']]}</p><section><h2>유효 구간 결과</h2><p>전체 마우스 이동거리 합계 <strong>{f(s['distance_total'])} {unit}</strong></p><p>마리당 환산 이동거리 <strong>{f(s['distance_per_animal'])} {unit}</strong></p><p>마리당 평균 속도 <strong>{f(s['mean_speed'])} {unit}/s</strong></p><p>유효 추적 {s['valid_s']:.2f}초 / {s['observed_s']:.2f}초 · 추적 비율 {f(s['coverage_pct'])}%</p><p>{html.escape(reasons)}</p></section><section><img src='data:image/png;base64,{image}' alt='속도와 유효 추적 비율'></section><section><h2>해석과 한계</h2><ul>{''.join('<li>'+html.escape(n)+'</li>' for n in notes)}</ul></section></html>"""
    (run/'report.html').write_text(page,encoding='utf-8')

def write_comparison(output,reports,mode='water',metric='mean_speed'):
    result=comparison(reports,mode,metric);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'comparison.json',result)
    with (output/'cages.csv').open('x',encoding='utf-8-sig',newline='') as f:
        fields=['setup','block','administration','cage','group','water_type','value','unit','recordings','valid_s','observed_s','coverage_pct']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for s in result['strata']:
            for g in s['groups']:
                for p in g['points']:w.writerow(dict(setup=s['setup_id'],block=s['comparison_block'],administration=s['administration'],unit=result['unit'],**p))
    table=''.join(f"<tr><td>{html.escape(s['setup_id'])} / {html.escape(s['comparison_block'])}</td><td>{html.escape(g['name'])}</td><td>{g['n_cages']}</td><td>{g['mean']:.3f}</td><td>{g['sd'] if g['sd'] is not None else '계산 불가'}</td></tr>" for s in result['strata'] for g in s['groups'])
    (output/'report.html').write_text(f"<!doctype html><meta charset='utf-8'><h1>케이지별 이동성 기술적 비교</h1><p>{html.escape(metric)} ({result['unit']}) · 독립 단위: 케이지 · 평균 ± 표본 SD · p값 미계산</p><table border='1' cellpadding='10'><tr><th>촬영·블록</th><th>그룹</th><th>케이지 n</th><th>평균</th><th>SD</th></tr>{table}</table><p>동일 케이지 반복 촬영은 한 점으로 합칩니다. 유효 추적 80% 미만·길이 미보정 결과는 제외합니다. 가림에 의한 누락 편향과 촬영 조건을 확인해야 하며 물의 인과 효과를 확정하지 않습니다.</p><pre>{html.escape(json.dumps(result['excluded'],ensure_ascii=False,indent=2))}</pre>",encoding='utf-8')
    return result
