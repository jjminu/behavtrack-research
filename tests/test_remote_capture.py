import json
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
import numpy as np
import pytest
from behavtrack import research as r
from behavtrack import research_web as web
from behavtrack import capture_agent as agent_module
from behavtrack.capture_agent import CaptureAgent
from behavtrack.remote_transport import Transport, RemoteError
from behavtrack.remote_service import RemoteHub
from behavtrack.integrity import atomic_json, read_json, digest


def test_agent_camera_open_failure_returns_json_and_keeps_server_alive(tmp_path):
    def unavailable(_):raise SystemExit("camera unavailable")
    capture=web.Capture(tmp_path/"capture_results",opener=unavailable)
    agent=CaptureAgent(tmp_path,capture=capture)
    server,token=agent_module.make_agent_server(agent)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f"http://127.0.0.1:{server.server_port}"
    headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"}
    try:
        with pytest.raises(urllib.error.HTTPError) as failed:
            urllib.request.urlopen(urllib.request.Request(base+"/api/camera",data=b'{"camera":0}',headers=headers),timeout=3)
        assert failed.value.code==400
        message=json.load(failed.value)["error"]
        assert "카메라 0번" in message and "Windows" in message
        with urllib.request.urlopen(urllib.request.Request(base+"/api/state",headers=headers),timeout=3) as response:state=json.load(response)
        assert state["capture"]["error"]==message and not state["capture"]["connected"]
    finally:server.shutdown();server.server_close();agent.close()


def recorded(agent):
    spec=r.protocol(dict(group="원격 전송 합성 검증",cage="가상_C1",animals=2,food_rect=[0,0,.5,1],interval=.2,
                         administration="bottle_only",water_type="control"),(200,100))
    spec.update(source_type="synthetic_software_test",test_only=True)
    rec=r.ResearchRecording(agent.results,spec,0,10)
    for i in range(21):
        frame=np.full((100,200,3),200,np.uint8);frame[20:40,20+i:40+i]=10
        rec.record(frame,rec.start_monotonic+i*.1)
    rec.finish()
    atomic_json(rec.folder/"remote_analysis.json",dict(roi=[0,0,1,1],reference_roi=None,setup_id="합성_원격_검증",comparison_block="합성_기능시험",end_s=2))
    return rec.folder


@pytest.fixture
def connected(tmp_path,monkeypatch):
    (tmp_path/"server").mkdir()
    app=web.Application(tmp_path/"server")
    monkeypatch.setattr(app.remote,"_queue",lambda:None)
    app.remote.start("127.0.0.1",0)
    pair=app.remote.pair("합성 촬영 PC")
    agent=CaptureAgent(tmp_path/"client");agent.connect(pair)
    try:yield app,agent,pair
    finally:agent.close();app.close()


def manifest(folder):
    info=read_json(folder/"recording.json")
    names=sorted(set(["recording.json"]+list(info["sha256"])))
    return dict(local_session=folder.name,files=[dict(name=n,size=(folder/n).stat().st_size,sha256=digest(folder/n)) for n in names],analysis=read_json(folder/"remote_analysis.json"))


def test_completed_result_syncs_without_camera_and_survives_restart(connected):
    app,agent,pair=connected
    folder=recorded(agent);queued=agent.upload(folder)
    assert agent.state()["pipeline"]["state"]=="queued"
    answer=app.remote._analyze(queued["id"],threading.Event(),lambda *a:None)
    assert answer["state"]=="complete"
    agent.sync_once()
    assert agent.state()["pipeline"]["state"]=="complete"
    assert agent.state()["remote"]["records"][0]["results"]["activity"]["summary"]["mean_changed_area_pct"]>0
    reopened=CaptureAgent(agent.root)
    try:
        assert not reopened.state()["connection"]["connected"]
        assert reopened.state()["pipeline"]["state"]=="complete"
        assert reopened.state()["remote"]["records"]==agent.state()["remote"]["records"]
    finally:reopened.close()
    other=app.remote.pair("다른 장치")
    atomic_json(agent.root/"capture_connection.json",other)
    different=CaptureAgent(agent.root)
    try:assert different.state()["remote"]["records"]==[]
    finally:different.close()


def test_automatic_distance_reaches_both_screens_and_scoped_files(connected):
    app,agent,pair=connected
    class Detector:
        runtime={'name':'synthetic_known_centers'}
        def __init__(self,*args):pass
        def predict(self,frame,fid):return {'box_format':'xyxy','boxes':[[20+fid,20,28+fid,28],[100+fid,60,108+fid,68]],'scores':[.99,.99]}
        def close(self):pass
    app.predictor_factory=Detector
    folder=recorded(agent);values=read_json(folder/'remote_analysis.json')
    values['locomotion']={'calibration':{'points':[[0,0],[1,0],[1,1],[0,1]],'width_cm':40,'height_cm':20,'planar_confirmed':True}}
    atomic_json(folder/'remote_analysis.json',values)
    queued=agent.upload(folder);answer=app.remote._analyze(queued['id'],threading.Event(),lambda *a:None)
    assert answer['state']=='complete'
    server_record=app.remote.snapshot()['records'][0]
    assert server_record['results']['locomotion']['unit']=='cm'
    assert server_record['results']['locomotion']['summary']['mean_speed']==pytest.approx(2.)
    agent.sync_once();client_record=agent.state()['remote']['records'][0]
    assert client_record['results']==server_record['results']
    transport=Transport(pair)
    path='/v1/file?'+urllib.parse.urlencode(dict(id=queued['id'],stage='locomotion',file='pairs.csv'))
    assert b'distance' in transport.request(path,raw=True)[0]
    wrong=app.remote.pair('other')
    with pytest.raises(RemoteError):Transport(wrong).request(path,raw=True)


def test_preview_failure_does_not_stop_result_sync_or_health_reporting(connected,monkeypatch):
    app,agent,pair=connected
    def broken_preview(*a,**k):raise ValueError("preview encoder unavailable")
    monkeypatch.setattr(agent.capture,"preview",broken_preview)
    agent.set_ui_status({"build":agent_module.BUILD,"rendered_records":1,"latest_state":"complete"})
    agent.start_workers();deadline=time.monotonic()+6
    while time.monotonic()<deadline:
        if agent.connected and pair['device'] in app.remote.live:break
        time.sleep(.05)
    assert agent.connected
    live=app.remote.live[pair['device']]
    assert live['preview'] is None and live['client_status']['ui']['rendered_records']==1
    assert live['client_status']['build']==agent_module.BUILD


def test_capture_page_serves_one_complete_script_and_visible_results(connected):
    _,agent,_=connected;server,token=agent_module.make_agent_server(agent)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base) as response:page=response.read().decode('utf-8')
        assert page.index('id="agent-results-panel"')<page.index('id="agent-preview"')
        assert '/capture-app.js?' in page and 'src="/remote-view.js"' not in page
        with urllib.request.urlopen(base+'/capture-app.js?v=test') as response:
            assert response.headers.get_content_type()=='application/javascript'
            script=response.read().decode('utf-8')
        assert script.index('function remoteRecord')<script.index('async function agentPoll')
    finally:server.shutdown();server.server_close()


def test_tls_pairing_is_pinned_scoped_and_revocable(connected):
    app,agent,pair=connected
    bad=dict(pair,certificate_sha256="0"*64)
    with pytest.raises(ValueError,match="인증서"):Transport(bad).request("/v1/state")
    with pytest.raises(RemoteError) as e:Transport(dict(pair,token="x"*50)).request("/v1/state")
    assert e.value.status==403
    transport=Transport(pair)
    with pytest.raises(RemoteError) as e:transport.request("/v1/camera",{})
    assert e.value.status==404
    folder=recorded(agent);receipt=transport.request("/v1/begin",manifest(folder))
    other=app.remote.pair("다른 촬영 PC")
    with pytest.raises(RemoteError) as e:Transport(other).request("/v1/upload?id="+receipt["id"])
    assert e.value.status==403
    assert not Transport(other).request("/v1/state")["records"]
    app.remote.revoke(pair["device"])
    with pytest.raises(RemoteError) as e:transport.request("/v1/state")
    assert e.value.status==403
    assert pair["token"] not in (app.remote.root/"devices.json").read_text(encoding="utf-8")


def test_lost_ack_resumes_without_corruption_or_duplicate_commit(connected,monkeypatch):
    app,agent,pair=connected;folder=recorded(agent);sent=[]
    real=Transport
    class LostAck(real):
        def request(self,path,value=None,raw=False):
            result=super().request(path,value,raw)
            if path.startswith("/v1/chunk") and not sent:
                sent.append(len(value));raise OSError("connection lost after server persisted chunk")
            return result
    monkeypatch.setattr(agent_module,"Transport",LostAck)
    with pytest.raises(OSError):agent.upload(folder)
    receipt=next(app.remote.uploads.glob("*/receipt.json"));before=read_json(receipt)
    assert before["state"]=="uploading" and app.remote.upload_status(pair["device"],before["id"])["offsets"]
    monkeypatch.setattr(agent_module,"Transport",real)
    answer=agent.upload(folder);assert answer["state"]=="queued"
    server_folder=app.results/answer["server_session"]
    for f in manifest(folder)["files"]:assert digest(server_folder/f["name"])==digest(folder/f["name"])
    assert agent.upload(folder)["id"]==answer["id"]
    assert len(list(app.results.glob("research_remote_*")))==1
    assert read_json(folder/"transfer.json")["state"]=="received"
    changed=manifest(folder);changed["analysis"]["comparison_block"]="changed"
    with pytest.raises(RemoteError,match="변경"):Transport(pair).request("/v1/begin",changed)


def test_unverified_upload_never_publishes_or_queues_and_blocks_paths(connected):
    app,agent,pair=connected;folder=recorded(agent);transport=Transport(pair)
    info=manifest(folder);answer=transport.request("/v1/begin",info)
    with pytest.raises(RemoteError):transport.request("/v1/chunk?"+urllib.parse.urlencode(dict(id=answer["id"],file="../outside",offset=0)),b"x")
    for f in info["files"]:
        content=(folder/f["name"]).read_bytes()
        if f["name"]=="raw.mp4":content=b"x"*len(content)
        transport.request("/v1/chunk?"+urllib.parse.urlencode(dict(id=answer["id"],file=f["name"],offset=0)),content)
    with pytest.raises(RemoteError,match="검증 실패"):transport.request("/v1/commit",dict(id=answer["id"]))
    assert app.remote.receipt(answer["id"])["state"]=="uploading"
    assert not list(app.results.glob("research_remote_*"))
    assert (folder/"raw.mp4").is_file()


def test_restart_recovers_queue_runs_real_activity_and_delivers_same_report(connected):
    app,agent,pair=connected;folder=recorded(agent);answer=agent.upload(folder);ident=answer["id"]
    # Simulate a process exit between dequeue and stage completion.
    receipt=app.remote.receipt(ident);receipt.update(state="running");app.remote.save(receipt)
    port=app.remote.server.server_port;app.remote.close()
    app.remote=RemoteHub(app)
    assert app.remote.receipt(ident)["state"]=="queued"
    app.remote.start("127.0.0.1",0);pair=dict(pair,url=app.remote.url)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        result=Transport(pair).request("/v1/state")["records"][0]
        if result["state"] in ("complete","failed"):break
        time.sleep(.1)
    assert result["state"]=="complete",result
    assert result["results"]["activity"]["summary"]["mean_changed_area_pct"]>0
    query=urllib.parse.urlencode(dict(id=ident,stage="activity",file="report.html"))
    content,mime=Transport(pair).request("/v1/file?"+query,raw=True)
    path=app.remote.result_file(pair["device"],ident,"activity","report.html")
    assert content==path.read_bytes() and "html" in mime
    assert Transport(pair).request("/v1/commit",dict(id=ident))["state"]=="complete"
    assert len(list((app.results/result["server_session"]/"cage_activity").glob("activity_*")))==1


def test_offline_server_does_not_interrupt_camera_recording(tmp_path,monkeypatch):
    class Camera:
        def read(self):time.sleep(.01);return True,np.full((100,200,3),200,np.uint8)
        def release(self):pass
    cap=web.Capture(tmp_path/"capture_results",opener=lambda _:Camera())
    agent=CaptureAgent(tmp_path,capture=cap)
    agent.pair=dict(device="device_"+"a"*12,name="offline test",url="https://127.0.0.1:1",token="x"*50,certificate_sha256="0"*64)
    monkeypatch.setattr(Transport,"request",lambda *a,**k:(_ for _ in ()).throw(OSError("offline")))
    try:
        cap.open(0)
        deadline=time.monotonic()+2
        while not cap.state()["connected"] and time.monotonic()<deadline:time.sleep(.01)
        state=cap.state()
        result=agent.start_recording(dict(group="G",cage="C",animals=2,interval=.2,food_rect=[0,0,.5,1],size=state["size"],generation=state["generation"],analysis=dict(roi=[0,0,1,1],setup_id="S",comparison_block="B")))
        agent.start_workers();time.sleep(.2);first=cap.state()["frames"];time.sleep(.2)
        assert cap.state()["frames"]>first>=5
        cap.stop_recording();assert read_json(agent.results/result["session"]/"recording.json")["state"]=="complete"
    finally:agent.close()
