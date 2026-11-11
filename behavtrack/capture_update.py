"""User-launched, bounded capture-client update. Does not replace data or keys."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time,urllib.request,uuid
BUILD='capture-2026.11.10.3'
ALLOWED=('research_web.py','remote_transport.py','locomotion_config.py','web/calibration.js','capture_agent.py','web/capture.html','web/capture.js','web/remote-view.js','web/style.css')

def request(runtime,path,body=None):
    port=int(runtime['port'])
    if not 1<=port<=65535:raise ValueError('Invalid local capture port')
    data=json.dumps(body).encode() if body is not None else None
    req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=data,headers={'Authorization':'Bearer '+runtime['token'],'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=8) as response:return json.load(response)

def apply(target,bundle,no_browser=False):
    target=Path(target).resolve();bundle=Path(bundle).resolve();app=(target/'app').resolve()
    app.relative_to(target)
    if not (app/'capture_connection.json').is_file() or not (app/'.capture_venv/Scripts/python.exe').is_file() or not (app/'behavtrack/capture_agent.py').is_file():
        raise ValueError('Select the original capture program folder containing 실행하기.cmd and app.')
    manifest=json.loads((bundle/'update_manifest.json').read_text(encoding='utf-8'))
    for name in ALLOWED:
        source=(bundle/'payload'/name).resolve();source.relative_to(bundle/'payload')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['files'][name]:raise ValueError('Update checksum failed: '+name)
        (app/'behavtrack'/name).resolve().relative_to(app/'behavtrack')
    runtime_file=app/'.capture_runtime.json'
    if runtime_file.exists():
        runtime=json.loads(runtime_file.read_text(encoding='utf-8'));state=request(runtime,'/api/state')
        if Path(state['output']).resolve()!=(app/'capture_results').resolve():raise ValueError('Local capture program path does not match selection')
        if state['capture'].get('recording') or state['capture'].get('saving'):raise ValueError('Recording is active. Finish and save it before updating.')
        request(runtime,'/api/shutdown',{})
        for _ in range(150):
            if not runtime_file.exists():break
            time.sleep(.1)
        if runtime_file.exists():raise ValueError('Capture program has not exited. Close it normally and run this update again.')
    backup=(app/'update_backups'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])).resolve()
    backup.relative_to(app)
    for name in ALLOWED:
        dest=(app/'behavtrack'/name).resolve();dest.relative_to(app/'behavtrack')
        old=backup/name;old.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():shutil.copy2(dest,old)
    try:
        for name in ALLOWED:
            dest=(app/'behavtrack'/name).resolve();dest.relative_to(app/'behavtrack')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(bundle/'payload'/name,dest)
    except Exception:
        for name in ALLOWED:
            if (backup/name).exists():shutil.copy2(backup/name,app/'behavtrack'/name)
        raise
    logs=app/'update_logs';logs.mkdir(exist_ok=True)
    with (logs/'capture_start.log').open('ab') as log:
        command=[str(app/'.capture_venv/Scripts/python.exe'),'-X','utf8','-m','behavtrack.capture_agent']+(['--no-browser'] if no_browser else [])
        process=subprocess.Popen(command,cwd=app,
            stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    for _ in range(150):
        if runtime_file.exists():
            try:
                runtime=json.loads(runtime_file.read_text(encoding='utf-8'));state=request(runtime,'/api/state')
                if state.get('build')==BUILD:return {'updated':True,'build':BUILD,'backup':str(backup),'restored_results':len(state['remote']['records'])}
            except (OSError,ValueError,KeyError):pass
        if process.poll() is not None:break
        time.sleep(.1)
    raise ValueError('Files updated. Check app/update_logs/capture_start.log and run 실행하기.cmd.')

if __name__=='__main__':
    try:print(json.dumps(apply(sys.argv[1],Path(__file__).parent),ensure_ascii=False,indent=2))
    except Exception as exc:print('UPDATE STOPPED: '+str(exc));raise SystemExit(1)
