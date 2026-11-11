'use strict';
let remoteSignature='',remoteDeviceSignature='';
const remoteDeviceCards=new Map();
function remotePreviewSchedule(card,delay){clearTimeout(card.timer);if(!card.removed)card.timer=setTimeout(()=>remotePreviewLoop(card),delay);}
function remotePreviewStatus(card,age,fps,mode){
 const d=card.data,source=d.capture?.capture_fps;
 card.info.textContent=`카메라 ${source?Number(source).toFixed(1):'—'} fps · 미리보기 ${Number(fps||0).toFixed(1)} fps`+(age!==null&&Number.isFinite(age)?` · 최근 수신 ${age.toFixed(1)}초 전`:' · 영상 수신 대기');
 card.info.className=age!==null&&age>3?'notice':'helper';
 card.upgrade.hidden=mode!=='legacy';
 card.upgrade.textContent='촬영 PC가 기존 방식으로 2초마다 영상을 보내고 있습니다. 촬영 PC에 미리보기 개선 업데이트를 적용하세요.';
}
async function remotePreviewLoop(card){
 if(card.removed||card.busy)return;
 const d=card.data;
 if(document.hidden||$('remote').hidden||!d.online||d.revoked||!d.capture?.connected){remotePreviewSchedule(card,500);return;}
 card.busy=true;const started=performance.now(),controller=new AbortController();card.controller=controller;
 const timeout=setTimeout(()=>controller.abort(),2500);
 try{
  card.failed=false;const r=await request('/api/remote-preview?'+new URLSearchParams({device:d.id,after:card.sequence??''}),undefined,controller.signal);
  const ageHeader=r.headers.get('X-Preview-Age'),age=ageHeader?Number(ageHeader):null;
  remotePreviewStatus(card,age,Number(r.headers.get('X-Preview-FPS')||0),r.headers.get('X-Preview-Mode'));
  if(r.status===204)return;
  const bitmap=await createImageBitmap(await r.blob());
  try{
   if(card.removed||document.hidden||$('remote').hidden||controller.signal.aborted)return;
   if(card.canvas.width!==bitmap.width||card.canvas.height!==bitmap.height){card.canvas.width=bitmap.width;card.canvas.height=bitmap.height;}
   card.context.drawImage(bitmap,0,0);card.sequence=r.headers.get('X-Preview-Sequence');
   card.canvas.dataset.sequence=card.sequence;card.canvas.hidden=false;
  }finally{bitmap.close();}
 }catch(e){card.failed=true;if(!card.removed)card.info.textContent='미리보기 연결 재시도 중 · 마지막 수신 화면 유지';}
 finally{clearTimeout(timeout);card.controller=null;card.busy=false;remotePreviewSchedule(card,card.failed?500:Math.max(15,100-(performance.now()-started)));}
}
function remoteRefreshDevices(devices){
 const node=$('remote-devices'),keep=new Set();
 if(devices.length)for(const p of node.querySelectorAll(':scope > p'))p.remove();
 for(const d of devices){
  keep.add(d.id);let card=remoteDeviceCards.get(d.id);
  if(!card){
   const root=remoteElement('article',undefined,'remote-device'),title=remoteElement('h3'),connection=remoteElement('p'),capture=remoteElement('p'),canvas=remoteElement('canvas'),info=remoteElement('p',undefined,'helper'),upgrade=remoteElement('p',undefined,'helper'),revoke=remoteElement('button','이 장치 연결 키 폐기');
   canvas.width=640;canvas.height=480;canvas.hidden=true;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',d.name+'의 실시간 원격 미리보기');upgrade.hidden=true;revoke.className='quiet';
   root.append(title,connection,capture,canvas,info,upgrade,revoke);node.append(root);
   card={root,title,connection,capture,canvas,info,upgrade,revoke,context:canvas.getContext('2d'),data:d,busy:false,removed:false,sequence:null};remoteDeviceCards.set(d.id,card);
   revoke.onclick=async()=>{if(!confirm(card.data.name+'의 앞으로의 접속을 차단할까요? 기존 원본은 보존합니다.'))return;await request('/api/remote/revoke',{device:d.id});await poll();};
   remotePreviewSchedule(card,0);
  }
  card.data=d;card.title.textContent=d.name;card.connection.textContent=d.revoked?'연결 키 폐기됨':d.online?'● 온라인':'연결 대기 · 촬영 PC의 네트워크를 확인하세요.';
  card.capture.textContent=d.capture?.recording?`녹화 중 · ${d.capture.frames}프레임 · ${Number(d.capture.elapsed).toFixed(0)}초`:d.capture?.connected?'카메라 연결됨 · 촬영 대기':'카메라 연결 전';
  card.revoke.disabled=d.revoked;
  if(!card.busy)remotePreviewStatus(card,d.preview_age_s??null,d.preview_fps||0,d.preview_mode);
 }
 for(const[id,card]of remoteDeviceCards)if(!keep.has(id)){card.removed=true;clearTimeout(card.timer);card.controller?.abort();card.root.remove();remoteDeviceCards.delete(id);}
 if(!devices.length&&!node.children.length)node.append(remoteElement('p','촬영 장치 연결 파일을 발급하면 여기에 표시됩니다.','helper'));
}
$('remote-start').onclick=async()=>{try{await request('/api/remote/start',{host:$('remote-host').value,port:Number($('remote-port').value)});await poll();notice('암호화된 원격 촬영 수신을 시작했습니다.',true);}catch(e){notice(e.message);}};
$('remote-pair').onclick=async()=>{try{const pair=await(await request('/api/remote/pair',{name:$('remote-name').value})).json();const b=new Blob([JSON.stringify(pair,null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='capture_connection.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),60000);notice('촬영 장치 연결 파일을 발급했습니다. 촬영 전용 폴더에 넣어 사용하세요.',true);await poll();}catch(e){notice(e.message);}};
async function refreshRemote(){
 if(!state?.remote)return;
 const r=state.remote;$('remote-url').textContent=r.enabled?'수신 중 · '+r.url:'원격 수신 대기 · VPN 주소로 서버를 시작하세요.';
 $('remote-error').textContent=r.error||'';$('remote-error').hidden=!r.error;$('remote-start').disabled=r.enabled;$('remote-pair').disabled=!r.enabled;
 if(r.host&&!$('remote-host').value)$('remote-host').value=r.host;
 remoteRefreshDevices(r.devices);
 const signature=JSON.stringify(r.records);if(signature!==remoteSignature){remoteSignature=signature;const list=$('remote-records');remoteRelease(list);list.replaceChildren();for(const record of r.records){const node=remoteElement('article');list.append(node);await remoteRecord(node,record,async q=>(await request('/api/remote-file?'+new URLSearchParams(q))).blob(),async id=>{await request('/api/remote/retry',{id});await poll();});}if(!r.records.length)list.append(remoteElement('p','촬영을 종료하면 자동으로 원본이 전송되고 분석이 시작됩니다.','helper'));}
}
setInterval(()=>{if(!$('remote').hidden)refreshRemote().catch(e=>notice(e.message));},1000);
