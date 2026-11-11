from pathlib import Path
import hashlib,json
import pytest
from behavtrack import capture_update as update


def prepared(tmp_path):
    target=tmp_path/'capture';app=target/'app';bundle=tmp_path/'update'
    (app/'.capture_venv/Scripts').mkdir(parents=True)
    (app/'.capture_venv/Scripts/python.exe').write_bytes(b'not executed')
    (app/'capture_connection.json').write_text('{}')
    files={}
    for name in update.ALLOWED:
        old=app/'behavtrack'/name;old.parent.mkdir(parents=True,exist_ok=True);old.write_bytes(b'old:'+name.encode())
        new=bundle/'payload'/name;new.parent.mkdir(parents=True,exist_ok=True);new.write_bytes(b'new:'+name.encode())
        files[name]=hashlib.sha256(new.read_bytes()).hexdigest()
    (bundle/'update_manifest.json').write_text(json.dumps({'files':files}))
    return target,app,bundle


def test_update_refuses_active_recording_before_shutdown_or_writes(tmp_path,monkeypatch):
    target,app,bundle=prepared(tmp_path)
    (app/'.capture_runtime.json').write_text('{"port":12345,"token":"test"}')
    calls=[]
    def request(runtime,path,body=None):
        calls.append(path);return {'output':str(app/'capture_results'),'capture':{'recording':'research_active'}}
    monkeypatch.setattr(update,'request',request)
    with pytest.raises(ValueError,match='Recording is active'):update.apply(target,bundle)
    assert calls==['/api/state']
    assert all((app/'behavtrack'/n).read_bytes().startswith(b'old:') for n in update.ALLOWED)


def test_update_checks_all_payload_hashes_before_contacting_program(tmp_path,monkeypatch):
    target,app,bundle=prepared(tmp_path);(bundle/'payload/web/capture.js').write_bytes(b'tampered')
    monkeypatch.setattr(update,'request',lambda *a:pytest.fail('must not contact or stop application'))
    with pytest.raises(ValueError,match='checksum'):update.apply(target,bundle)
    assert all((app/'behavtrack'/n).read_bytes().startswith(b'old:') for n in update.ALLOWED)


def test_failed_program_file_copy_restores_backup(tmp_path,monkeypatch):
    target,app,bundle=prepared(tmp_path);copy=update.shutil.copy2
    def fail(source,dest,*a,**k):
        if Path(source)==bundle/'payload/web/capture.js':raise OSError('simulated write failure')
        return copy(source,dest,*a,**k)
    monkeypatch.setattr(update.shutil,'copy2',fail)
    with pytest.raises(OSError,match='write failure'):update.apply(target,bundle)
    assert all((app/'behavtrack'/n).read_bytes().startswith(b'old:') for n in update.ALLOWED)
