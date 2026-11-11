"""Capture-only local browser application with outbound, pinned HTTPS sync."""
from __future__ import annotations
import argparse
import base64
import csv
import json
from pathlib import Path
import secrets
import threading
import time
from collections import deque
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from .integrity import atomic_json, read_json, digest, utcnow
from .recording_io import load_times
from .remote_transport import Transport, validate_pair, CHUNK
from .research_web import Capture, open_browser

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path(__file__).parent/"web"
BUILD="capture-2026.11.10.3"


class CaptureAgent:
    def __init__(self,root=ROOT,capture=None):
        self.root=Path(root);self.results=self.root/"capture_results";self.results.mkdir(parents=True,exist_ok=True)
        self.capture=capture or Capture(self.results)
        self.lock=threading.RLock();self.stop=threading.Event()
        self.pair=read_json(self.root/"capture_connection.json") if (self.root/"capture_connection.json").exists() else None
        if self.pair:validate_pair(self.pair)
        self.remote={"devices":[],"records":[]};self.error=None;self.connected=False;self.last_sync=None
        self.transfer={"state":"idle","message":"촬영 종료 후 자동 전송합니다.","done":0,"total":0}
        self.workers=[]
        self.ui_status={};self.heartbeat_error=None;self.cache_error=None
        self.preview_status={'target_fps':10,'sent_frames':0,'fps':0.,'error':None}
        self.preview_epoch=secrets.token_hex(6)
        self._load_cache()

    def _scope(self,pair):
        return {k:pair[k] for k in ("device","url","certificate_sha256")} if pair else None

    def _load_cache(self):
        try:
            path=self.root/"capture_result_cache.json"
            if path.exists():
                cached=read_json(path)
                if cached.get("scope")==self._scope(self.pair):
                    if not isinstance(cached.get("records"),list):raise ValueError("저장된 결과 목록 형식 오류")
                    self.remote={"devices":[],"records":cached.get("records",[])};self.last_sync=cached.get("last_sync_utc")
        except (OSError,ValueError,KeyError) as exc:self.cache_error=str(exc)

    def _accept_remote(self,pair,remote):
        if not isinstance(remote,dict) or not isinstance(remote.get("records"),list):raise ValueError("서버 결과 응답 형식을 확인하세요.")
        with self.lock:
            if self._scope(pair)!=self._scope(self.pair):return
            changed=self.remote.get("records")!=remote["records"]
            self.remote=remote;self.error=None;self.connected=True;self.last_sync=utcnow()
            if changed or not (self.root/"capture_result_cache.json").exists():
                try:
                    atomic_json(self.root/"capture_result_cache.json",{"scope":self._scope(pair),"last_sync_utc":self.last_sync,"records":remote["records"]})
                    self.cache_error=None
                except OSError as exc:self.cache_error=str(exc)

    def sync_once(self):
        pair=self.pair
        if pair:self._accept_remote(pair,Transport(pair,timeout=5).request("/v1/state"))

    def _sync_loop(self):
        while not self.stop.is_set():
            try:self.sync_once()
            except Exception as exc:
                with self.lock:self.error=str(exc);self.connected=False
            self.stop.wait(1)

    def pipeline(self,capture,local,remote,transfer):
        if capture.get("recording"):
            return {"state":"recording","message":"원본 녹화 중", "local_session":capture["recording"],"frames":capture["frames"],"elapsed_s":capture["elapsed"]}
        if capture.get("saving"):return {"state":"saving","message":"촬영 원본 저장 중"}
        latest=local[0] if local else None
        records=remote.get("records",[])
        if latest:
            if latest.get("recovery_required"):return {"state":"failed","message":"비정상 종료 촬영 · 원본을 보존하고 복구 점검하세요.","local_session":latest["id"]}
            marker=latest["transfer"];record=next((r for r in records if r.get("local_session")==latest["id"] or (marker.get("id") and r["id"]==marker["id"])),None)
            if record and record.get("state")!="uploading":return dict(record)
            if transfer.get("session")==latest["id"] and transfer["state"]=="uploading":return dict(transfer,local_session=latest["id"],bytes_done=transfer.get("done",0),bytes_total=transfer.get("total",0))
            if record:return dict(record)
            return {"state":"waiting","message":"서버 수신 완료 · 분석 상태 확인 중" if marker.get("state")=="received" else "촬영 완료 · 자동 전송 준비 중", "local_session":latest["id"]}
        return dict(records[0]) if records else {"state":"idle","message":"촬영을 종료하면 전송·분석 진행률이 여기에 표시됩니다."}

    def set_ui_status(self,value):
        safe={"build":str(value.get("build",""))[:80],"rendered_records":max(0,min(10000,int(value.get("rendered_records",0)))),
              "latest_state":str(value.get("latest_state",""))[:40],"error":str(value.get("error") or "")[:500],"seen_utc":utcnow()}
        with self.lock:self.ui_status=safe
        return {"received":True}

    def start_workers(self):
        for target,name in ((self._heartbeat,"capture-heartbeat"),(self._preview_loop,"capture-live-preview"),(self._sync_loop,"capture-result-sync"),(self._upload_loop,"capture-uploader")):
            t=threading.Thread(target=target,name=name,daemon=True);t.start();self.workers.append(t)

    def connect(self,value):
        pair=validate_pair(value)
        # Reject an untrusted endpoint before persisting a changed pairing.
        status=Transport(pair).request("/v1/state")
        with self.lock:
            if self.pair and self.pair["device"]!=pair["device"] and (self.capture.state()["recording"] or self.transfer["state"]=="uploading"):
                raise ValueError("녹화·전송을 마친 뒤 촬영 장치 연결을 바꾸세요.")
            atomic_json(self.root/"capture_connection.json",pair)
            if self._scope(self.pair)!=self._scope(pair):self.remote={"devices":[],"records":[]}
            self.pair=pair;self._accept_remote(pair,status)
        return {"connected":True}

    def state(self):
        capture=self.capture.state()
        active=None
        if capture["recording"]:
            folder=self.results/capture["recording"]
            if (folder/"remote_analysis.json").exists():active={"protocol":read_json(folder/"protocol.json"),"analysis":read_json(folder/"remote_analysis.json")}
        local=[]
        for p in sorted(self.results.glob("research_*/recording.json"),reverse=True):
            rec=read_json(p);spec=read_json(p.parent/"protocol.json")
            transfer=read_json(p.parent/"transfer.json") if (p.parent/"transfer.json").exists() else {"state":"waiting"}
            local.append({"id":p.parent.name,"group":spec["group"],"cage":spec["cage"],"frames":rec["frames"],"state":rec["state"],"transfer":transfer,
                          "recovery_required":rec["state"]=="recording" and p.parent.name!=self.capture.state()["recording"]})
        with self.lock:
            return {"build":BUILD,"capture":capture,"preview":dict(self.preview_status),"active":active,"pipeline":self.pipeline(capture,local,self.remote,self.transfer),"ui_status":dict(self.ui_status),"connection":{"configured":bool(self.pair),"connected":self.connected,
                    "name":self.pair.get("name") if self.pair else None,"url":self.pair.get("url") if self.pair else None,
                    "error":self.error,"last_sync_utc":self.last_sync,"heartbeat_error":self.heartbeat_error,"cache_error":self.cache_error},"transfer":dict(self.transfer),"remote":self.remote,"local":local,"output":str(self.results)}

    def start_recording(self,value):
        if not self.pair:raise ValueError("서버에서 발급한 촬영 장치 연결 파일을 먼저 등록하세요.")
        from .cage_activity import settings
        analysis=value.get("analysis",{})
        config=settings(analysis)
        from .locomotion_config import requested
        locomotion=requested(analysis)
        if locomotion:analysis=dict(analysis,locomotion=locomotion)
        for key in ("setup_id","comparison_block"):
            if not 1<=len(str(analysis.get(key,"")).strip())<=100:raise ValueError("촬영 설정 ID와 비교 블록을 입력하세요.")
        # Capture and analysis settings are frozen together at recording start.
        folder=self.capture.start(value)
        try:atomic_json(self.results/folder/"remote_analysis.json",dict(analysis,**config,device=self.pair["device"]))
        except Exception:
            self.capture.stop_recording(error="자동 분석 조건 저장 실패 · 원본 보존")
            raise
        return {"session":folder}

    def _heartbeat(self):
        while not self.stop.is_set():
            pair=self.pair
            if pair:
                try:
                    transport=Transport(pair,timeout=5)
                    with self.lock:client_status={"build":BUILD,"ui":dict(self.ui_status),"cached_records":len(self.remote.get("records",[])),"sync_error":self.error,'preview':dict(self.preview_status)}
                    transport.request("/v1/heartbeat",{"capture":self.capture.state(),"client_status":client_status})
                    self.heartbeat_error=None
                except Exception as exc:
                    with self.lock:self.heartbeat_error=str(exc)
            self.stop.wait(2)

    def _preview_loop(self):
        """One in-flight latest frame. Never queue frames or block recording."""
        transport=None;scope=None;last_frame=None;sent=deque(maxlen=31);rtt=.05
        try:
            while not self.stop.is_set():
                started=time.monotonic();pair=self.pair
                try:
                    if pair:
                        if self._scope(pair)!=scope:
                            if transport:transport.close()
                            transport=Transport(pair,timeout=1.2,keep_alive=True);scope=self._scope(pair);last_frame=None
                        width=480 if rtt>.18 else 640;quality=50 if rtt>.3 else 65
                        packet=self.capture.preview(with_metadata=True,max_width=width,quality=quality)
                        if packet and packet[1]!=last_frame:
                            image,key=packet
                            if len(image)>80000:
                                width,quality=480,45
                                packet=self.capture.preview(with_metadata=True,max_width=width,quality=quality)
                                if not packet:continue
                                image,key=packet
                            if len(image)>180000:raise ValueError('원격 미리보기 이미지가 너무 큽니다.')
                            t=time.monotonic()
                            transport.request('/v1/preview?'+urllib.parse.urlencode({'frame':self.preview_epoch+':'+key}),image)
                            measured=time.monotonic()-t;rtt=.8*rtt+.2*measured;last_frame=key;sent.append(time.monotonic())
                            fps=(len(sent)-1)/(sent[-1]-sent[0]) if len(sent)>1 and sent[-1]>sent[0] else 0.
                            with self.lock:self.preview_status.update(sent_frames=self.preview_status['sent_frames']+1,fps=fps,error=None,
                                                    round_trip_ms=round(measured*1000,1),width=width,quality=quality)
                        elif not packet:
                            with self.lock:self.preview_status.update(fps=0.)
                except Exception as exc:
                    with self.lock:self.preview_status.update(error=str(exc),fps=0.)
                    if transport:transport.close()
                    if self.stop.wait(.5):break
                # Account for transport/encode time; a slow link drops obsolete
                # frames instead of accumulating playback latency.
                self.stop.wait(max(.005,.1-(time.monotonic()-started)))
        finally:
            if transport:transport.close()

    def upload(self,folder,pair=None):
        folder=Path(folder);pair=pair or self.pair
        if not pair:raise ValueError("서버 연결 파일이 없습니다.")
        transport=Transport(pair)
        self._accept_remote(pair,transport.request("/v1/state"))
        info,_=load_times(folder,verify=False)
        if info["state"] not in ("complete","partial"):raise ValueError("녹화를 먼저 종료하세요.")
        config=read_json(folder/"remote_analysis.json")
        if config.get("device") and config["device"]!=pair["device"]:raise ValueError("이 촬영의 장치 연결 키와 현재 연결이 다릅니다.")
        names=["recording.json"]+list(info["sha256"])
        if any(not (folder/name).is_file() for name in names):raise ValueError("촬영 원본 파일이 없습니다.")
        files=[{"name":name,"size":(folder/name).stat().st_size,"sha256":digest(folder/name) if name=="recording.json" else info["sha256"][name]} for name in sorted(set(names))]
        total=sum(f["size"] for f in files)
        status=transport.request("/v1/begin",{"local_session":folder.name,"files":files,"analysis":config})
        marker={"state":"uploading","id":status["id"],"server_session":status["server_session"],"updated_utc":utcnow()}
        atomic_json(folder/"transfer.json",marker)
        done=0
        for f in files:
            offset=status.get("offsets",{}).get(f["name"],f["size"] if status["state"]!="uploading" else 0)
            if not 0<=offset<=f["size"]:raise ValueError("서버의 이어 보내기 위치가 올바르지 않습니다.")
            done+=offset
            with (folder/f["name"]).open("rb") as stream:
                stream.seek(offset)
                while offset<f["size"]:
                    if self.stop.is_set():return
                    content=stream.read(CHUNK)
                    with self.lock:self.transfer={"state":"uploading","session":folder.name,"message":f["name"]+" 자동 전송 중","done":done,"total":total}
                    query=urllib.parse.urlencode({"id":status["id"],"file":f["name"],"offset":offset})
                    answer=transport.request("/v1/chunk?"+query,content)
                    new=answer.get("offset",offset+len(content))
                    if not 0<=new<=f["size"]:raise ValueError("전송 재개 위치 오류")
                    done+=new-offset;offset=new;stream.seek(offset)
        answer=transport.request("/v1/commit",{"id":status["id"]})
        if answer["state"]=="uploading":raise OSError("서버에서 원본을 검증 중입니다. 잠시 후 상태를 다시 확인합니다.")
        marker.update(state="received",server_state=answer["state"],updated_utc=utcnow())
        atomic_json(folder/"transfer.json",marker)
        with self.lock:self.transfer={"state":"received","session":folder.name,"message":"서버 원본 검증 완료 · 자동 분석 대기","done":total,"total":total}
        try:self.sync_once()
        except Exception:pass  # The independent sync loop retries; a committed upload stays committed.
        return answer

    def _upload_loop(self):
        while not self.stop.wait(1):
            if not self.pair:continue
            for path in sorted(self.results.glob("research_*/remote_analysis.json")):
                if self.stop.is_set():return
                try:
                    rec=read_json(path.parent/"recording.json")
                    if rec["state"] not in ("complete","partial") or not rec["frames"]:continue
                    marker=read_json(path.parent/"transfer.json") if (path.parent/"transfer.json").exists() else {}
                    if marker.get("state")=="received":continue
                    self.upload(path.parent)
                except Exception as exc:
                    with self.lock:self.transfer.update(state="waiting",message="원본 보존 · 연결 후 재시도: "+str(exc))
                    if self.stop.wait(5):return
                    break

    def close(self):
        self.stop.set();self.capture.close()
        for worker in self.workers:worker.join(timeout=1)


def make_agent_server(agent,port=0,token=None):
    token=token or secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value,mime="application/json; charset=utf-8"):
            body=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type",mime);self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff")
            if not mime.startswith("text/html") or self.path.split('?')[0] in ("/","/capture.html"):
                self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            else:self.send_header("Content-Security-Policy","default-src 'none'; img-src data:; style-src 'unsafe-inline'; sandbox")
            self.end_headers();self.wfile.write(body)
        def allowed(self,api):
            expected=f"127.0.0.1:{self.server.server_port}"
            return self.headers.get("Host")==expected and self.headers.get("Origin","http://"+expected)=="http://"+expected and (not api or secrets.compare_digest(self.headers.get("Authorization",""),"Bearer "+token))
        def do_GET(self):
            url=urllib.parse.urlsplit(self.path);q={k:v[0] for k,v in urllib.parse.parse_qs(url.query).items()}
            if not self.allowed(url.path.startswith("/api/")):return self.reply(403,{"error":"촬영 전용 실행 파일로 다시 열어 주세요."})
            try:
                if url.path=="/capture-app.js":return self.reply(200,(ASSETS/"calibration.js").read_bytes()+b'\n'+(ASSETS/"remote-view.js").read_bytes()+b'\n'+(ASSETS/"capture.js").read_bytes(),"application/javascript; charset=utf-8")
                if url.path in ("/","/capture.js","/style.css","/remote-view.js"):
                    name="capture.html" if url.path=="/" else url.path[1:]
                    import mimetypes
                    return self.reply(200,(ASSETS/name).read_bytes(),mimetypes.guess_type(name)[0]+"; charset=utf-8")
                if url.path=="/api/state":return self.reply(200,agent.state())
                if url.path=="/api/preview":
                    image=agent.capture.preview()
                    if image:return self.reply(200,image,"image/jpeg")
                    return self.reply(404,{"error":"카메라 연결 전"})
                if url.path=="/api/remote-file":
                    path="/v1/file?"+urllib.parse.urlencode({k:q[k] for k in ("id","stage","file")})
                    body,mime=Transport(agent.pair).request(path,raw=True)
                    return self.reply(200,body,mime)
                self.reply(404,{"error":"지원하지 않는 요청입니다."})
            except (Exception,SystemExit) as exc:self.reply(400,{"error":str(exc)})
        def do_POST(self):
            if not self.allowed(True):return self.reply(403,{"error":"촬영 전용 실행 파일로 다시 열어 주세요."})
            try:
                size=int(self.headers.get("Content-Length",0))
                if not 0<size<=1024*1024:raise ValueError("요청 크기 오류")
                v=json.loads(self.rfile.read(size))
                if self.path=="/api/connect":answer=agent.connect(v)
                elif self.path=="/api/ui-status":answer=agent.set_ui_status(v)
                elif self.path=="/api/camera":agent.capture.open(v.get("camera"));answer={"opened":True}
                elif self.path=="/api/camera-close":
                    if agent.capture.state()["recording"]:raise ValueError("녹화를 먼저 종료하세요.")
                    agent.capture.close_camera();answer={"closed":True}
                elif self.path=="/api/start":answer=agent.start_recording(v)
                elif self.path=="/api/stop":answer={"session":agent.capture.stop_recording()}
                elif self.path=="/api/shutdown":
                    self.reply(200,{"stopping":True});threading.Thread(target=self.server.shutdown,daemon=True).start();return
                else:return self.reply(404,{"error":"촬영 전용 기능이 아닙니다."})
                self.reply(200,answer)
            except (Exception,SystemExit) as exc:
                message=str(exc)
                if self.path=="/api/camera":
                    if isinstance(exc,SystemExit):message=f"카메라 {v.get('camera')}번을 열 수 없습니다. USB 연결, Windows 카메라 권한과 카메라 번호를 확인하세요."
                    with agent.capture.lock:agent.capture.error=message
                self.reply(400,{"error":message})
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler);server.daemon_threads=True
    return server,token


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=ROOT);parser.add_argument("--no-browser",action="store_true");args=parser.parse_args()
    runtime=args.root/".capture_runtime.json"
    if runtime.exists():
        try:
            old=read_json(runtime)
            req=urllib.request.Request(f"http://127.0.0.1:{old['port']}/api/state",headers={"Authorization":"Bearer "+old["token"]})
            with urllib.request.urlopen(req,timeout=2) as response:state=json.load(response)
            if state.get("output")==str(args.root/"capture_results"):
                if not args.no_browser:open_browser(old["url"])
                print("이미 실행 중인 촬영 화면을 열었습니다.");return
        except (OSError,ValueError,KeyError):pass
    agent=CaptureAgent(args.root);server,key=make_agent_server(agent);agent.start_workers()
    url=f"http://127.0.0.1:{server.server_port}/#token={key}"
    atomic_json(runtime,{"port":server.server_port,"token":key,"url":url})
    print("BehavTrack 촬영 전용: "+url,flush=True)
    if not args.no_browser:open_browser(url)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:
        agent.close();server.server_close()
        if runtime.exists() and read_json(runtime).get("token")==key:runtime.unlink()


if __name__=="__main__":main()
