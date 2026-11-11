"""Device-scoped TLS ingestion, resumable files and a durable analysis queue.

The administrative browser remains loopback-only. This listener exposes only
paired-device upload/status/report routes; no camera, shell or filesystem API.
"""
from __future__ import annotations
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import ssl
import socket
from collections import deque
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from .integrity import atomic_json, read_json, digest, json_digest, utcnow, contained_file
from .recording_io import load_times
from .remote_transport import CHUNK


def tls_identity(folder, host):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    folder.mkdir(parents=True, exist_ok=True)
    cert, key = folder/"server.crt", folder/"server.key"
    if not cert.exists() or not key.exists():
        private=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,"BehavTrack paired receiver")])
        now=datetime.now(timezone.utc)
        signed=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(private.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(days=1))
            .not_valid_after(now+timedelta(days=3650)).add_extension(x509.BasicConstraints(ca=False,path_length=None),True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(host))]),False)
            .sign(private,hashes.SHA256()))
        # Neither file is included in the capture computer package.
        key.write_bytes(private.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
        cert.write_bytes(signed.public_bytes(serialization.Encoding.PEM))
    der=ssl.PEM_cert_to_DER_cert(cert.read_text())
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(cert),str(key))
    return context, hashlib.sha256(der).hexdigest()


def file_name(name):
    return isinstance(name,str) and (name in ("protocol.json","recording.json","frame_times.csv") or bool(re.fullmatch(r"raw(?:_\d{6})?\.mp4(?:\.json)?",name)))


class RemoteHub:
    def __init__(self, app):
        self.app=app;self.root=app.root/"remote_server";self.root.mkdir(exist_ok=True)
        self.uploads=self.root/"uploads";self.uploads.mkdir(exist_ok=True)
        self.lock=threading.RLock();self.stop=threading.Event();self.server=None;self.worker=None;self.commit_locks={}
        self.devices=read_json(self.root/"devices.json") if (self.root/"devices.json").exists() else {}
        self.live={};self.host=None;self.url=None;self.fingerprint=None;self.error=None
        for p in self.uploads.glob("upload_*/receipt.json"):
            r=read_json(p)
            if r["state"]=="running":r.update(state="queued",message="서버 재시작 후 대기열 복구");atomic_json(p,r)

    def start(self, host, port=8877):
        with self.lock:
            if self.server:return self.snapshot()
            address=ipaddress.ip_address(host)
            if address.version!=4 or address.is_unspecified or address.is_multicast:
                raise ValueError("이 컴퓨터의 VPN 또는 내부망 IPv4 주소를 지정하세요.")
            context,self.fingerprint=tls_identity(self.root,host)
            server=make_receiver(self,host,port,context)
            self.server=server;self.host=host;self.url=f"https://{host}:{server.server_port}"
            atomic_json(self.root/"settings.json",{"host":host,"port":server.server_port})
            threading.Thread(target=server.serve_forever,name="remote-tls",daemon=True).start()
            self.worker=threading.Thread(target=self._queue,name="remote-analysis-queue",daemon=True);self.worker.start()
            return self.snapshot()

    def pair(self, name):
        if not self.server:raise ValueError("원격 수신 서버를 먼저 시작하세요.")
        name=str(name).strip()
        if not 1<=len(name)<=80:raise ValueError("촬영 컴퓨터 이름을 입력하세요.")
        ident="device_"+secrets.token_hex(6);key=secrets.token_urlsafe(40)
        with self.lock:
            self.devices[ident]={"name":name,"token_sha256":hashlib.sha256(key.encode()).hexdigest(),"created_utc":utcnow(),"revoked":False}
            atomic_json(self.root/"devices.json",self.devices)
        return {"schema":1,"device":ident,"name":name,"url":self.url,"certificate_sha256":self.fingerprint,"token":key}

    def revoke(self, ident):
        with self.lock:
            if ident not in self.devices:raise ValueError("촬영 장치를 선택하세요.")
            self.devices[ident]["revoked"]=True;atomic_json(self.root/"devices.json",self.devices)
        return {"revoked":True}

    def authorize(self, ident, authorization):
        with self.lock:
            d=self.devices.get(ident,{})
            supplied=hashlib.sha256(authorization.removeprefix("Bearer ").encode()).hexdigest()
            return bool(authorization.startswith("Bearer ") and d and not d.get("revoked") and secrets.compare_digest(supplied,d["token_sha256"]))

    def path(self, ident):
        if not isinstance(ident,str) or not re.fullmatch(r"upload_[a-f0-9]{24}",ident):raise ValueError("전송 ID가 올바르지 않습니다.")
        folder=(self.uploads/ident).resolve()
        if folder.parent!=self.uploads.resolve():raise ValueError("전송 경로 오류")
        return folder

    def receipt(self, ident, device=None):
        r=read_json(self.path(ident)/"receipt.json")
        if device is not None and r["device"]!=device:raise PermissionError("이 촬영 장치의 자료만 조회할 수 있습니다.")
        return r

    def save(self,r):atomic_json(self.path(r["id"])/"receipt.json",r)

    def public(self,r):
        result={k:v for k,v in r.items() if k not in ("files","analysis","manifest_hash")}
        if r.get("group") and "region_purpose" not in r:
            spec=read_json(self.app.results/r["server_session"]/"protocol.json")
            result.update(region_purpose=spec.get("region_purpose"),source_type=spec.get("source_type"),test_note=str(spec.get("test_note",""))[:1000])
        if r["state"]=="uploading":
            result["bytes_done"]=0
            for f in r["files"]:
                try:result["bytes_done"]+=(self.path(r["id"])/"source"/f["name"]).stat().st_size
                except FileNotFoundError:pass
        result["bytes_total"]=sum(f["size"] for f in r["files"])
        return result

    def snapshot(self,device=None):
        with self.lock:
            records=[]
            for p in self.uploads.glob("upload_*/receipt.json"):
                r=read_json(p)
                if device is None or r["device"]==device:records.append(self.public(r))
            devices=[]
            for ident,d in self.devices.items():
                if device is not None and device!=ident:continue
                live=self.live.get(ident,{})
                devices.append(dict(id=ident,name=d["name"],revoked=d["revoked"],**{k:v for k,v in live.items() if k!="preview" and not k.startswith('_')},
                    online=time.time()-live.get("seen_epoch",0)<12,has_preview=bool(live.get("preview")),**self.preview_stats(live)))
            return {"enabled":bool(self.server),"url":self.url,"host":self.host,"error":self.error,
                    "devices":devices,"records":sorted(records,key=lambda r:r["created_utc"],reverse=True)}

    def heartbeat(self,device,v):
        c=v.get("capture",{});safe={k:c.get(k) for k in ("recording","frames","elapsed","connected","error","capture_fps")}
        if len(json.dumps(safe))>4000:raise ValueError("촬영 상태 크기 오류")
        preview=None
        if v.get("preview"):
            preview=base64.b64decode(v["preview"],validate=True)
            if len(preview)>180000 or not preview.startswith(b"\xff\xd8"):raise ValueError("미리보기 이미지 오류")
        client=v.get("client_status",{})
        if not isinstance(client,dict) or len(json.dumps(client,allow_nan=False))>5000:raise ValueError("촬영 PC 화면 상태 크기 오류")
        with self.lock:
            live=self.live.setdefault(device,{})
            live.setdefault('preview',None)
            live.update(capture=safe,last_seen_utc=utcnow(),seen_epoch=time.time(),client_status=client)
            # Compatibility for old clients. Their slow heartbeat image must
            # never overwrite the separate live-preview stream.
            if preview and time.monotonic()-live.get('_fast_preview_seen',-1e9)>5:
                self._accept_preview(live,preview,None,'legacy')
        return {"received":True}

    def preview_stats(self,live):
        times=live.get('_preview_times',[])
        age=max(0,time.monotonic()-times[-1]) if times else None
        fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] and age<3 else 0.
        return dict(preview_age_s=age,preview_fps=fps,preview_mode=live.get('_preview_mode','waiting'))

    def _accept_preview(self,live,content,frame,mode):
        now=time.monotonic()
        if frame is not None and frame==live.get('_preview_frame'):return
        live.update(preview=content,preview_seq=live.get('preview_seq',0)+1,
                    _preview_frame=frame,_preview_mode=mode)
        live.setdefault('_preview_times',deque(maxlen=31)).append(now)
        if mode=='live':live['_fast_preview_seen']=now

    def preview(self,device,content,frame):
        if not 4<=len(content)<=180000 or not content.startswith(b'\xff\xd8') or not content.endswith(b'\xff\xd9'):
            raise ValueError('미리보기 JPEG 크기·형식을 확인하세요.')
        if not isinstance(frame,str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,100}',frame):raise ValueError('미리보기 프레임 ID 오류')
        with self.lock:
            live=self.live.setdefault(device,{})
            self._accept_preview(live,content,frame,'live')
            return dict(received=True,sequence=live['preview_seq'])

    def preview_packet(self,device,after=None):
        with self.lock:
            if device not in self.devices or self.devices[device]['revoked']:raise PermissionError('활성 촬영 장치를 선택하세요.')
            live=self.live.get(device,{})
            return dict(content=live.get('preview') if str(live.get('preview_seq'))!=str(after) else None,
                        sequence=live.get('preview_seq',0),**self.preview_stats(live))

    def begin(self,device,v):
        local=v.get("local_session","")
        if not isinstance(local,str) or not re.fullmatch(r"research_[A-Za-z0-9_]{1,100}",local):raise ValueError("촬영 ID 오류")
        files=v.get("files",[])
        if not isinstance(files,list) or not 4<=len(files)<=10000:raise ValueError("전송 파일 목록 오류")
        names=set();total=0
        for f in files:
            name=f.get("name")
            if not file_name(name) or name in names:raise ValueError("지원하지 않거나 중복된 원본 파일입니다.")
            size=f.get("size")
            if isinstance(size,bool) or not isinstance(size,int) or not 0<size<=100*1024**3:raise ValueError("원본 파일 크기 오류")
            if name.endswith(".json") and size>4*1024**2:raise ValueError("원본 설정 파일이 너무 큽니다.")
            if name=="frame_times.csv" and size>256*1024**2:raise ValueError("촬영 시각 파일이 256 MB를 넘습니다. 관찰 촬영을 나누세요.")
            if not re.fullmatch(r"[a-f0-9]{64}",str(f.get("sha256",""))):raise ValueError("파일 SHA-256 누락")
            names.add(name);total+=size
        if not {"recording.json","protocol.json","frame_times.csv","raw.mp4"}<=names:raise ValueError("원본 영상·촬영 시각·조건 파일이 모두 필요합니다.")
        analysis=v.get("analysis",{})
        from .cage_activity import settings
        config=settings(analysis)
        from .locomotion_config import requested
        locomotion=requested(analysis)
        if locomotion:analysis=dict(analysis,locomotion=locomotion)
        if not str(analysis.get("setup_id","")).strip() or not str(analysis.get("comparison_block","")).strip():raise ValueError("촬영 설정 ID와 비교 블록을 입력하세요.")
        analysis=dict(analysis,**config)
        ident="upload_"+hashlib.sha256((device+":"+local).encode()).hexdigest()[:24]
        signature=json_digest({"files":files,"analysis":analysis})
        with self.lock:
            folder=self.path(ident)
            if (folder/"receipt.json").exists():
                old=self.receipt(ident,device)
                if old["manifest_hash"]!=signature:raise ValueError("같은 촬영 ID의 파일이나 분석 조건이 변경됐습니다. 원본을 확인하세요.")
                return self.upload_status(device,ident)
            if total+512*1024**2>shutil.disk_usage(self.app.results).free:raise ValueError("서버 저장 공간이 부족합니다. 촬영 PC의 원본은 보존됩니다.")
            (folder/"source").mkdir(parents=True,exist_ok=False)
            r={"id":ident,"device":device,"local_session":local,"server_session":"research_remote_"+ident[7:],
               "created_utc":utcnow(),"state":"uploading","message":"촬영 원본 수신 중","files":files,"analysis":analysis,
               "manifest_hash":signature,"done":0,"total":0,"results":{}}
            self.save(r);return self.upload_status(device,ident)

    def upload_status(self,device,ident):
        with self.lock:
            r=self.receipt(ident,device);out=self.public(r)
            out["offsets"]={f["name"]:(self.path(ident)/"source"/f["name"]).stat().st_size if (self.path(ident)/"source"/f["name"]).exists() else 0 for f in r["files"]} if r["state"]=="uploading" else {}
            return out

    def chunk(self,device,ident,name,offset,content):
        with self.lock:
            r=self.receipt(ident,device)
            if r["state"]!="uploading":return {"accepted":True,"state":r["state"]}
            f=next((f for f in r["files"] if f["name"]==name),None)
            if not f:raise ValueError("전송 목록에 없는 파일입니다.")
            path=contained_file(self.path(ident)/"source",name);position=path.stat().st_size if path.exists() else 0
            if offset!=position:return {"offset":position,"retry":True}
            if not content or len(content)>CHUNK or position+len(content)>f["size"]:raise ValueError("전송 조각 크기 오류")
            if shutil.disk_usage(self.app.results).free<len(content)+256*1024**2:raise ValueError("서버 저장 공간이 부족합니다.")
            with path.open("ab") as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
            return {"offset":position+len(content)}

    def commit(self,device,ident):
        with self.lock:
            r=self.receipt(ident,device)
            if r["state"]!="uploading":return self.public(r)
            guard=self.commit_locks.setdefault(ident,threading.Lock())
        if not guard.acquire(blocking=False):return self.public(self.receipt(ident,device))
        try:
            r=self.receipt(ident,device)
            if r["state"]!="uploading":return self.public(r)
            r.update(message="서버에서 원본 SHA-256·촬영 시각·프레임 수 검증 중")
            with self.lock:self.save(r)
            source=self.path(ident)/"source";dest=self.app.results/r["server_session"]
            # A move completed just before a crash is a recoverable commit.
            checked=dest if dest.is_dir() else source
            for f in r["files"]:
                p=contained_file(checked,f["name"])
                if not p.is_file() or p.stat().st_size!=f["size"] or digest(p)!=f["sha256"]:raise ValueError("원본 파일 검증 실패: "+f["name"])
            info,times=load_times(checked)
            if info["state"] not in ("complete","partial"):raise ValueError("녹화를 종료한 원본만 분석합니다.")
            spec=read_json(checked/"protocol.json")
            from .research import protocol
            from .cage_activity import metadata
            rebuilt=protocol(dict(spec,animals=spec["criteria"]["animals"],interval=spec["criteria"]["interval_s"],food_rect=spec["criteria"]["food_rect"]),spec["criteria"]["frame_size"])
            if rebuilt["criteria_sha256"]!=spec["criteria_sha256"]:raise ValueError("촬영 조건 해시가 다릅니다.")
            metadata(r["analysis"],spec)
            # Frame/container count validation does not decode a multi-hour movie.
            import cv2
            segments=info.get("segments") or [{"file":"raw.mp4","frames":info["frames"]}]
            for s in segments:
                cap=cv2.VideoCapture(str(contained_file(checked,s["file"])))
                try:
                    if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=s["frames"]:raise ValueError("영상 조각 프레임 수가 촬영 기록과 다릅니다.")
                    if [round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]!=spec["criteria"]["frame_size"]:raise ValueError("영상 해상도가 촬영 조건과 다릅니다.")
                finally:cap.release()
            if not dest.exists():os.replace(source,dest)
            atomic_json(dest/"remote_origin.json",{"device":device,"local_session":r["local_session"],"upload":ident,"received_utc":utcnow(),"files":r["files"]})
            r.update(state="queued",message="원본 검증 완료 · 자동 분석 대기",group=spec["group"],cage=spec["cage"],water_type=spec.get("water_type","unspecified"),frames=info["frames"],
                     region_purpose=spec.get("region_purpose"),source_type=spec.get("source_type"),test_note=str(spec.get("test_note",""))[:1000])
            with self.lock:self.save(r)
            return self.public(r)
        finally:guard.release()

    def retry(self,ident):
        with self.lock:
            r=self.receipt(ident)
            if r["state"] not in ("failed","cancelled"):raise ValueError("실패하거나 취소된 분석만 다시 실행할 수 있습니다.")
            r.update(state="queued",message="분석 재시도 대기");self.save(r);return self.public(r)

    def _queue(self):
        while not self.stop.wait(.5):
            try:
                with self.lock:
                    waiting=[read_json(p) for p in self.uploads.glob("upload_*/receipt.json") if read_json(p)["state"]=="queued"]
                    waiting=[r for r in waiting if not self.devices.get(r["device"],{}).get("revoked")]
                if not waiting:continue
                r=min(waiting,key=lambda r:r["created_utc"])
                if self.devices.get(r["device"],{}).get("revoked"):continue
                with self.app.lock:
                    if self.app.capture.state()["recording"] or self.app.capture.saving or (self.app.job_thread and self.app.job_thread.is_alive()):continue
                    self.app.start_job("원격 촬영 자동 분석",lambda cancel,progress:self._analyze(r["id"],cancel,progress))
                    current=self.app.job_thread
                while current.is_alive() and not self.stop.wait(.2):pass
            except Exception as exc:self.error=str(exc)

    def _analyze(self,ident,cancel,progress):
        from . import cage_activity as activity
        from . import research
        with self.lock:r=self.receipt(ident);r.update(state="running",message="케이지 활동성 분석 준비",stage="activity");self.save(r)
        folder=self.app.results/r["server_session"]
        def update(done,total,message):
            with self.lock:
                r.update(done=done,total=total,message=message);self.save(r)
            progress(done,total,message)
        try:
            if "activity" not in r["results"]:
                answer=activity.analyze(folder,r["analysis"],cancel,update)
                if answer["state"]!="complete":raise ValueError(answer.get("error") or "활동성 분석 중단")
                r["results"]["activity"]={"run":answer["run"],"summary":answer["report"]["summary"]}
                with self.lock:self.save(r)
            if r["analysis"].get("locomotion") and "locomotion" not in r["results"]:
                from . import cage_locomotion
                r["stage"]="locomotion";update(0,0,"이동거리·평균 속도 추적 준비")
                v=dict(r["analysis"],**r["analysis"]["locomotion"])
                answer=cage_locomotion.analyze(folder,v,self.app.root/".ai_venv/Scripts/python.exe",self.app.root/"models/topviewmouse/detector.pt",cancel,update,self.app.predictor_factory)
                if answer["state"]!="complete":raise ValueError(answer.get("error") or "이동거리 분석 중단")
                r["results"]["locomotion"]={"run":answer["run"],"summary":answer["report"]["summary"],"unit":answer["report"]["unit"]}
                with self.lock:self.save(r)
            if r["analysis"].get("proximity") is True and "proximity" not in r["results"]:
                r["stage"]="proximity";update(0,0,"사료 주변 접근 분석 준비")
                run,answer=research.analyze(folder,self.app.root/".ai_venv/Scripts/python.exe",self.app.root/"models/topviewmouse/detector.pt",cancel,
                    lambda done,total,run,message:update(done,total,message),self.app.predictor_factory)
                if answer["state"]!="complete":raise ValueError(answer.get("error") or "접근 분석 중단")
                r["results"]["proximity"]={"run":run.name,"summary":answer["summary"]}
            r.update(state="complete",message="자동 분석 완료",finished_utc=utcnow())
        except Exception as exc:r.update(state="cancelled" if cancel.is_set() else "failed",message=str(exc),finished_utc=utcnow())
        with self.lock:self.save(r)
        return {"kind":"remote_analysis","state":r["state"],"upload":ident,"session":r["server_session"],"results":r["results"],"error":r.get("message")}

    def result_file(self,device,ident,stage,name):
        with self.lock:r=self.receipt(ident,device)
        if stage=="activity":
            if name not in ("report.html","chart.png","pairs.csv","activity.json") and not re.fullmatch(r"example_\d{6}\.jpg",name):raise ValueError("지원하지 않는 결과 파일입니다.")
            info=r["results"].get(stage)
            if not info:raise ValueError("아직 활동성 결과가 없습니다.")
            folder=self.app.results/r["server_session"]/"cage_activity"/info["run"]
        elif stage=="locomotion":
            if name not in ("report.html","chart.png","pairs.csv","locomotion.json","tracking.mp4") and not re.fullmatch(r"example_\d{6}\.jpg",name):raise ValueError("지원하지 않는 이동거리 결과 파일입니다.")
            info=r["results"].get(stage)
            if not info:raise ValueError("아직 이동거리 결과가 없습니다.")
            folder=self.app.locomotion_run(r["server_session"],info["run"])
        elif stage=="proximity":
            if name not in ("interpretation.html","interpretation_chart.png","interpretation_bins.csv","samples.csv"):raise ValueError("지원하지 않는 결과 파일입니다.")
            info=r["results"].get(stage)
            if not info:raise ValueError("아직 접근 분석 결과가 없습니다.")
            folder=self.app.results/r["server_session"]/"analyses"/info["run"]
        else:raise ValueError("분석 종류 오류")
        path=contained_file(folder,name)
        if not path.is_file():raise ValueError("아직 생성되지 않은 결과 파일입니다.")
        return path

    def close(self):
        self.stop.set()
        if self.server:self.server.shutdown();self.server.server_close();self.server=None


def make_receiver(hub,host,port,context):
    class Receiver(ThreadingHTTPServer):
        daemon_threads=True
        def get_request(self):
            sock,address=super().get_request();sock.settimeout(20);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);return sock,address
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def reply(self,status,value):
            body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(body)
        def dispatch(self,post=False):
            device=self.headers.get("X-Device","")
            if self.headers.get("Origin") or not hub.authorize(device,self.headers.get("Authorization","")):
                self.close_connection=True
                return self.reply(403,{"error":"촬영 장치 연결 키를 확인하세요."})
            try:
                url=urlsplit(self.path);q={k:v[0] for k,v in parse_qs(url.query).items()}
                size=int(self.headers.get("Content-Length",0))
                limit=180000 if url.path=="/v1/preview" else CHUNK if url.path=="/v1/chunk" else 1024*1024
                if post and not 0<size<=limit:raise ValueError("요청 크기 오류")
                content=self.rfile.read(size) if post else None
                if post and len(content)!=size:raise ValueError("전송이 끊겼습니다. 다시 연결하면 재개합니다.")
                value=json.loads(content) if post and url.path not in ("/v1/chunk","/v1/preview") else content
                if not post and url.path=="/v1/state":return self.reply(200,hub.snapshot(device))
                if not post and url.path=="/v1/upload":return self.reply(200,hub.upload_status(device,q["id"]))
                if not post and url.path=="/v1/file":
                    p=hub.result_file(device,q["id"],q["stage"],q["file"])
                    import mimetypes
                    self.send_response(200);self.send_header("Content-Type",mimetypes.guess_type(p.name)[0] or "application/octet-stream");self.send_header("Content-Length",str(p.stat().st_size));self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff");self.end_headers()
                    with p.open("rb") as f:shutil.copyfileobj(f,self.wfile,1024*1024)
                    return
                if post and url.path=="/v1/heartbeat":answer=hub.heartbeat(device,value)
                elif post and url.path=="/v1/preview":answer=hub.preview(device,value,q.get('frame'))
                elif post and url.path=="/v1/begin":answer=hub.begin(device,value)
                elif post and url.path=="/v1/chunk":answer=hub.chunk(device,q["id"],q["file"],int(q["offset"]),value)
                elif post and url.path=="/v1/commit":answer=hub.commit(device,value["id"])
                else:return self.reply(404,{"error":"지원하지 않는 원격 요청입니다."})
                self.reply(200,answer)
            except PermissionError as exc:self.close_connection=True;self.reply(403,{"error":str(exc)})
            except (Exception,SystemExit) as exc:self.close_connection=True;self.reply(400,{"error":str(exc)})
        def do_GET(self):self.dispatch()
        def do_POST(self):self.dispatch(True)
    server=Receiver((host,port),Handler)
    server.socket=context.wrap_socket(server.socket,server_side=True,do_handshake_on_connect=False)
    return server
