import base64
import json
from pathlib import Path
import threading
import time
import urllib.request
import cv2
import numpy as np
import pytest
from behavtrack import research_web as web
from behavtrack.capture_agent import CaptureAgent
from behavtrack.remote_transport import Transport,RemoteError,PinnedConnection
from test_remote_capture import connected

def jpeg():
    return cv2.imencode('.jpg',np.zeros((80,120,3),np.uint8))[1].tobytes()

def test_persistent_preview_pins_connections_and_rechecks_revocation(connected):
    app,agent,pair=connected;t=Transport(pair,keep_alive=True)
    try:
        t.request('/v1/preview?frame=one',jpeg());sock=t.connection.sock
        t.request('/v1/preview?frame=two',jpeg());assert t.connection.sock is sock
        # A new socket, including automatic reconnection, rechecks the pin.
        t.connection.close();t.connection.fingerprint='0'*64
        with pytest.raises(ValueError,match='인증서'):t.request('/v1/preview?frame=three',jpeg())
        assert app.remote.live[pair['device']]['preview_seq']==2
        t.request('/v1/preview?frame=four',jpeg());app.remote.revoke(pair['device'])
        with pytest.raises(RemoteError) as failed:t.request('/v1/preview?frame=five',jpeg())
        assert failed.value.status==403
    finally:t.close()

def test_latest_preview_survives_heartbeat_and_legacy_frame_cannot_replace_it(connected):
    app,agent,pair=connected;hub=app.remote;device=pair['device'];image=jpeg()
    hub.preview(device,image,'epoch:10');hub.preview(device,image,'epoch:10')
    assert hub.live[device]['preview_seq']==1
    hub.heartbeat(device,dict(capture={'connected':True},client_status={'build':'new'}))
    assert hub.preview_packet(device)['content']==image
    hub.heartbeat(device,dict(capture={'connected':True},preview=base64.b64encode(b'\xff\xd8legacy').decode()))
    assert hub.preview_packet(device)['content']==image
    assert hub.preview_packet(device,after=1)['content'] is None
    snapshot=hub.snapshot()['devices'][0]
    assert snapshot['preview_mode']=='live' and snapshot['preview_age_s']>=0
    assert not any(k.startswith('_') for k in snapshot)
    with pytest.raises(ValueError):hub.preview(device,b'not jpeg','one')
    with pytest.raises(ValueError):hub.preview(device,image,'bad\nheader')

def test_preview_endpoint_returns_no_body_for_unchanged_sequence(connected):
    app,agent,pair=connected;image=jpeg();app.remote.preview(pair['device'],image,'one')
    server,token=web.make_server(app);threading.Thread(target=server.serve_forever,daemon=True).start()
    def get(after=''):
        return urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/remote-preview?device={pair["device"]}&after={after}',headers={'Authorization':'Bearer '+token}),timeout=3)
    try:
        with get() as response:
            assert response.read()==image;seq=response.headers['X-Preview-Sequence']
            assert response.headers['X-Preview-Mode']=='live'
        with get(seq) as response:assert response.status==204 and response.read()==b''
        app.remote.preview(pair['device'],image,'two')
        with get(seq) as response:assert response.status==200 and response.headers['X-Preview-Sequence']!=seq
    finally:server.shutdown();server.server_close()

def test_ten_fps_preview_keeps_camera_recording_independent(connected):
    app,old_agent,pair=connected
    class Camera:
        count=0
        def read(self):
            time.sleep(1/30);self.count+=1;frame=np.full((240,320,3),90,np.uint8)
            cv2.circle(frame,(20+self.count%260,100),10,(220,220,220),-1)
            return True,frame
        def release(self):pass
    root=old_agent.root/'preview_test';cap=web.Capture(root/'capture_results',opener=lambda _:Camera());agent=CaptureAgent(root,capture=cap);agent.connect(pair)
    try:
        cap.open(0);deadline=time.monotonic()+2
        while not cap.state()['connected'] and time.monotonic()<deadline:time.sleep(.01)
        s=cap.state();agent.start_recording(dict(group='합성 전송 검사',cage='virtual',animals=2,food_rect=[0,0,1,1],interval=.2,generation=s['generation'],size=s['size'],analysis=dict(roi=[0,0,1,1],setup_id='test',comparison_block='test')))
        worker=threading.Thread(target=agent._preview_loop);worker.start();agent.workers.append(worker)
        time.sleep(2.2);recording=cap.state();preview=app.remote.snapshot()['devices'][0]
        assert recording['frames']>=40 and recording['capture_fps']>20
        assert preview['preview_seq']>=15 and preview['preview_fps']>6
        assert preview['preview_age_s']<.6
        assert agent.preview_status['error'] is None
        cap.stop_recording()
    finally:agent.close()

def test_slow_preview_never_queues_stale_frames(tmp_path,monkeypatch):
    from behavtrack import capture_agent as module
    values=[]
    class Capture:
        def preview(self,**kw):return jpeg(),str(int(time.monotonic()*30))
        def close(self):pass
    class SlowTransport:
        def __init__(self,*a,**k):pass
        def request(self,path,content):values.append((time.monotonic(),int(path.split('frame=')[1].split('%3A')[-1])));time.sleep(.23)
        def close(self):pass
    monkeypatch.setattr(module,'Transport',SlowTransport)
    agent=CaptureAgent(tmp_path,capture=Capture());agent.pair=dict(device='device_'+'a'*12,url='https://127.0.0.1:1',token='x'*50,certificate_sha256='0'*64)
    worker=threading.Thread(target=agent._preview_loop);worker.start();time.sleep(.9);agent.stop.set();worker.join(timeout=2)
    assert 3<=len(values)<=5
    assert all(b[1]-a[1]>=6 for a,b in zip(values,values[1:]))

def test_preview_profile_cache_does_not_reuse_wrong_size(tmp_path):
    cap=web.Capture(tmp_path);cap.frame=np.zeros((720,1280,3),np.uint8);cap.generation=1;cap.frame_number=1
    small=cap.preview(max_width=480,quality=65);large=cap.preview(max_width=960,quality=80)
    assert cv2.imdecode(np.frombuffer(small,np.uint8),1).shape[1]==480
    assert cv2.imdecode(np.frombuffer(large,np.uint8),1).shape[1]==960
