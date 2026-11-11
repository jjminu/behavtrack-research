'use strict';
const $ = id => document.getElementById(id);
let token = new URLSearchParams(location.hash.slice(1)).get('token') || sessionStorage.getItem('behavtrack-token');
if (token) { sessionStorage.setItem('behavtrack-token', token); history.replaceState(null, '', location.pathname); }
let state = null, food = null, drag = null, bitmap = null, generation = null, busy = false;
let resultsSignature = '', selectedRuns = {}, review = null, reviewIndex = 0, lastJob = '', reviewImageUrl = null, reviewRender = 0, reviewReady = false;
const PREVIEW_FRAME_MS = 1000 / 30;
let previewTimes = [], lastPreviewFrame = null;
const canvas = $('preview'), ctx = canvas.getContext('2d');
function notice(text, success = false) { $('notice').textContent = text; $('notice').classList.toggle('success', success); $('notice').hidden = !text; }
async function request(path, body, signal) {
  const options = {headers:{Authorization:'Bearer '+token},signal};
  if (body !== undefined) {options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify(body);}
  const response = await fetch(path, options);
  if (!response.ok) {let data;try{data=await response.json();}catch{data={error:'요청을 처리하지 못했습니다.'};}throw new Error(data.error);}
  return response;
}
async function action(path, body, message) {
  if (busy) return false;
  busy = true; updateControls(); notice(message || '처리 중입니다…', true);
  try {await request(path, body); await poll(); notice(''); return true;}
  catch(error){notice(error.message);return false;}
  finally {busy=false;updateControls();}
}
function switchPage(name) {document.querySelectorAll('.page').forEach(e=>e.hidden=e.id!==name);document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('active',e.dataset.page===name));resetPreviewRate();}
document.querySelectorAll('.tab').forEach(e=>e.onclick=()=>switchPage(e.dataset.page));
function locked(){return busy || !!state?.capture.recording || !!state?.capture.saving;}
function updateControls(){
  const isRecording=!!state?.capture.recording, analyzing=state?.job.state==='running';
  ['camera','group','cage','animals','interval','open-camera','treatment','administration','water-type'].forEach(id=>$(id).disabled=locked());
  $('start').disabled=locked()||analyzing||!food||!state?.capture.connected;
  $('stop').disabled=busy||!isRecording;
  $('reset-food').disabled=locked()||!food;
  canvas.classList.toggle('locked',locked());
  $('cancel-analysis').hidden=!analyzing;
  if(review){$('save-review').disabled=busy||!reviewReady;$('previous-sample').disabled=busy||reviewIndex<=0;$('next-sample').disabled=busy||reviewIndex>=review.plan.length-1;}
}
function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(bitmap)ctx.drawImage(bitmap,0,0,canvas.width,canvas.height);
  const rect=drag?[Math.min(drag[0][0],drag[1][0]),Math.min(drag[0][1],drag[1][1]),Math.max(drag[0][0],drag[1][0]),Math.max(drag[0][1],drag[1][1])]:food;
  if(rect){const[x1,y1,x2,y2]=rect;ctx.fillStyle='rgba(44,220,142,.12)';ctx.fillRect(x1*canvas.width,y1*canvas.height,(x2-x1)*canvas.width,(y2-y1)*canvas.height);ctx.strokeStyle='#3fe39a';ctx.lineWidth=3;ctx.strokeRect(x1*canvas.width,y1*canvas.height,(x2-x1)*canvas.width,(y2-y1)*canvas.height);}
}
function point(event){const r=canvas.getBoundingClientRect();return[Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))];}
canvas.onpointerdown=event=>{if(locked()||!bitmap||!state?.capture.connected)return;canvas.setPointerCapture(event.pointerId);drag=[point(event),point(event)];};
canvas.onpointermove=event=>{if(drag){drag[1]=point(event);draw();}};
canvas.onpointerup=event=>{if(!drag)return;const a=drag[0],b=point(event);drag=null;const r=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];if(r[2]-r[0]>=.01&&r[3]-r[1]>=.01){food=r;$('mark-help').textContent='사료 주변을 표시했습니다. 녹화를 시작하면 이 범위가 고정됩니다.';}draw();updateControls();};
canvas.onpointercancel=()=>{drag=null;draw();};
$('reset-food').onclick=()=>{food=null;draw();updateControls();};
$('open-camera').onclick=async()=>{if(await action('/api/camera',{camera:Number($('camera').value)},'카메라를 연결하고 있습니다…')){food=null;draw();}};
$('start').onclick=async()=>{await action('/api/start',{group:$('group').value,cage:$('cage').value,treatment:$('treatment').value,administration:$('administration').value,water_type:$('water-type').value,animals:Number($('animals').value),interval:Number($('interval').value),food_rect:food,size:state.capture.size,generation:state.capture.generation},'원본 녹화를 준비합니다…');};
$('stop').onclick=async()=>{if(await action('/api/stop',{analyze:false},'영상을 저장하고 원본 무결성을 확인합니다…'))switchPage('results');};
$('cancel-analysis').onclick=()=>action('/api/cancel',{},'분석 취소 요청 중…');
$('shutdown').onclick=async()=>{if(!confirm('프로그램을 종료할까요? 녹화 중인 영상은 저장하고, 진행 중인 분석은 취소합니다.'))return;try{await request('/api/shutdown',{});stopped=true;notice('종료를 요청했습니다. 영상 저장이 끝나 명령 창이 닫힐 때까지 컴퓨터를 켜 두세요.',true);}catch(e){notice(e.message);}};
let stopped=false;
function duration(seconds){const n=Math.floor(seconds||0);return String(Math.floor(n/60)).padStart(2,'0')+':'+String(n%60).padStart(2,'0');}
function fmt(value,suffix=''){return value===null||value===undefined?'—':Number(value).toFixed(1)+suffix;}
function labelState(value){return({complete:'완료',partial:'부분 녹화',recording:'녹화 중',running:'분석 중',failed:'분석 실패',cancelled:'분석 취소'})[value]||value;}
function resetPreviewRate(){previewTimes=[];lastPreviewFrame=null;updatePreviewRate();}
function updatePreviewRate(){
  const now=performance.now();previewTimes=previewTimes.filter(t=>now-t<2000);
  const fps=previewTimes.length>1?(previewTimes.length-1)*1000/(previewTimes.at(-1)-previewTimes[0]):null;
  $('preview-performance').textContent=`미리보기 ${fmt(fps)} FPS · 카메라 수신 ${fmt(state?.capture.capture_fps)} FPS`;
}
async function poll(){
  if(stopped)return;
  try{
    state=await(await request('/api/state')).json();
    const c=state.capture;
    if(generation!==c.generation){generation=c.generation;food=null;drag=null;bitmap?.close();bitmap=null;resetPreviewRate();draw();}
    // A refreshed browser restores the immutable marked area of an active recording.
    const active=state.entries.find(e=>e.id===c.recording);
    if(active){food=active.protocol.criteria.food_rect;$('group').value=active.protocol.group;$('cage').value=active.protocol.cage;$('animals').value=active.protocol.criteria.animals;$('interval').value=active.protocol.criteria.interval_s;}
    $('camera-status').textContent=c.recording?'● 녹화 중':c.connected?'실시간 영상':'연결 전';
    $('record-status').textContent=c.saving?'원본 저장 중…':c.recording?`원본 녹화 중 · ${c.frames}프레임`:c.error||'촬영 준비';
    $('elapsed').textContent=duration(c.elapsed);$('placeholder').hidden=c.connected;
    $('output-path').textContent='저장 위치: '+state.output;
    const j=state.job;$('job').hidden=j.state==='idle';$('job-title').textContent=labelState(j.state);$('job-text').textContent=(j.message||'')+(j.total?` · ${j.done} / ${j.total} 표본`:'');$('job-progress').max=j.total||1;$('job-progress').value=j.done||0;
    if(j.state!==lastJob&&j.state==='complete')notice(j.kind?j.kind+' 작업을 마쳤습니다. 점검 결과를 확인하세요.':'분석을 마쳤습니다. 표본 검토에서 사람의 판정과 비교해 주세요.',true);
    lastJob=j.state;
    const signature=JSON.stringify(state.entries);
    if(signature!==resultsSignature){resultsSignature=signature;renderResults();}
    updateControls();
    updatePreviewRate();
  }catch(error){if(!stopped)notice(error.message||'프로그램 연결을 확인해 주세요. 실행하기.cmd로 다시 열 수 있습니다.');}
}
async function preview(){
  if(stopped)return;
  const started=performance.now(), active=state?.capture.connected&&!document.hidden&&!$('capture').hidden;
  if(active){
    const requestedGeneration=generation, controller=new AbortController();
    const timeout=setTimeout(()=>controller.abort(),2000);
    try{
      const response=await request('/api/preview',undefined,controller.signal);
      const frameId=response.headers.get('X-Preview-Frame');
      const blob=await response.blob();
      if(requestedGeneration===generation&&(!frameId||frameId.startsWith(generation+':'))&&(!frameId||frameId!==lastPreviewFrame)){
        const next=await createImageBitmap(blob);
        if(stopped||requestedGeneration!==generation||document.hidden||$('capture').hidden){next.close();}
        else{
          bitmap?.close();bitmap=next;lastPreviewFrame=frameId;
          if(canvas.width!==next.width||canvas.height!==next.height){canvas.width=next.width;canvas.height=next.height;}
          draw();previewTimes.push(performance.now());updatePreviewRate();
        }
      }
    }catch{}finally{clearTimeout(timeout);}
  }
  // One request at a time: a slow preview never accumulates delayed frames.
  setTimeout(preview,active?Math.max(0,PREVIEW_FRAME_MS-(performance.now()-started)):250);
}
document.addEventListener('visibilitychange',resetPreviewRate);
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function button(text,callback,cls){const b=el('button',text,cls);b.onclick=callback;return b;}
async function download(session,run,file){
  try{const q=new URLSearchParams({session,file});if(run)q.set('run',run);const response=await request('/api/download?'+q);const url=URL.createObjectURL(await response.blob());const a=el('a');a.href=url;a.download=session+'_'+(run?run+'_':'')+file;a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);}catch(e){notice(e.message);}
}
let interpretationImageUrl=null;
async function openInterpretation(session,run){
  try{
    const query=new URLSearchParams({session,run,file:'interpretation.json'});
    const report=await(await request('/api/download?'+query)).json();
    query.set('file','interpretation_chart.png');
    const chart=await(await request('/api/download?'+query)).blob();
    $('auto-report-panel')?.remove();
    if(interpretationImageUrl)URL.revokeObjectURL(interpretationImageUrl);
    interpretationImageUrl=URL.createObjectURL(chart);
    const panel=el('section',undefined,'panel');panel.id='auto-report-panel';
    panel.append(el('h2','자동 결과 해석'),el('p',`${report.group} · ${report.cage} · ${report.animals}마리 · ${report.interval_s}초 간격`,'helper'));
    const metrics=el('div',undefined,'metrics'),s=report.overall;
    for(const[name,value]of[['판정 가능',`${s.known_samples}/${s.samples}`],[`${report.region_name} 안 비율`,fmt(s.approach_pct,'%')],['판정 불가',`${s.unknown_samples}개`]]){const m=el('div',undefined,'metric');m.append(el('span',name),el('strong',value));metrics.append(m);}
    const image=el('img');image.src=interpretationImageUrl;image.alt='시간대별 영역 안 관측 비율과 표본 판정 타임라인';image.style.maxHeight='none';
    const facts=el('ol');for(const text of report.observations)facts.append(el('li',text));
    const limits=el('details'),list=el('ul');limits.append(el('summary','해석 범위'));for(const text of report.limits)list.append(el('li',text));limits.append(list);
    const actions=el('div',undefined,'result-actions');actions.append(button('해석 보고서 HTML',()=>download(session,run,'interpretation.html')),button('시간대별 수치 CSV',()=>download(session,run,'interpretation_bins.csv')),button('닫기',()=>{panel.remove();URL.revokeObjectURL(interpretationImageUrl);interpretationImageUrl=null;}));
    panel.append(metrics,image,el('h3','결과가 보여주는 것'),facts,limits,actions);$('result-list').before(panel);panel.scrollIntoView({behavior:'smooth',block:'start'});
  }catch(e){notice(e.message);}
}
function renderResults(){
  $('result-list').replaceChildren();$('empty-results').hidden=state.entries.length>0;
  for(const item of state.entries){
    const card=el('article',undefined,'panel result-card'),top=el('div',undefined,'result-top'),title=el('div');
    title.append(el('h3',`${item.protocol.group} · ${item.protocol.cage}`),el('p',item.protocol.source_type==='synthetic_software_test'?'합성 시험 자료 · 실제 동물 촬영 아님':item.recording.started_utc?new Date(item.recording.started_utc).toLocaleString('ko-KR'):'원본 촬영 시각 미기록'),el('p',`${item.protocol.criteria.animals}마리 · ${item.protocol.criteria.interval_s}초 간격 · ${item.recording.frames}프레임 · 기준 ${item.protocol.criteria_sha256.slice(0,8)}`));
    top.append(title,el('span',labelState(item.recording.state),'pill'));card.append(top);
    const run=item.analyses.find(r=>r.id===selectedRuns[item.id])||item.analyses[0];
    if(run){selectedRuns[item.id]=run.id;const select=el('select');select.setAttribute('aria-label','분석 실행 선택');for(const r of item.analyses){const option=el('option',`${new Date(r.created_utc).toLocaleString('ko-KR')} · ${labelState(r.state)}`);option.value=r.id;select.append(option);}select.value=run.id;select.onchange=()=>{selectedRuns[item.id]=select.value;renderResults();};const picker=el('div',undefined,'analysis-picker');picker.append(el('span','분석 이력'),select);card.append(picker);
      if(run.summary){const s=run.summary,m=el('div',undefined,'metrics');for(const[name,value]of[['접근 관측 비율',fmt(s.approach_pct,'%')],['평균 주변 마릿수',fmt(s.mean_count,'마리')],['판정 불가 / 전체',`${s.unknown_samples} / ${s.samples}`]]){const metric=el('div',undefined,'metric');metric.append(el('span',name),el('strong',value));m.append(metric);}card.append(m);card.append(el('p',`접근 ${s.approach_samples} · 비접근 ${s.away_samples} · 판정 가능 ${s.known_samples} 표본. 연속 체류 시간이나 섭취량이 아닙니다.`,'helper'));}
      if(run.error)card.append(el('p',run.error,'notice'));
    }
    if(item.recording.error)card.append(el('p',item.recording.error,'notice'));
    const actions=el('div',undefined,'result-actions');
    const analyze=button(run?'새 분석 실행':'녹화 후 분석',()=>action('/api/analyze',{session:item.id},'원본 무결성을 확인하고 분석을 시작합니다…'));
    analyze.disabled=item.recording.state==='recording'||!!state.capture.recording||state.job.state==='running';actions.append(analyze);
    if(run&&run.state!=='running')actions.append(button('표본 검토',()=>openReview(item.id,run.id)),button('분석 CSV',()=>download(item.id,run.id,'samples.csv')),button('분석 조건',()=>download(item.id,run.id,'analysis.json')));
    if(run&&run.state==='complete'){const report=button('자동 결과 해석',()=>openInterpretation(item.id,run.id));report.disabled=state.job.state==='running';actions.append(report);}
    actions.append(button('촬영 조건',()=>download(item.id,null,'protocol.json')),button('촬영 시각 CSV',()=>download(item.id,null,'frame_times.csv')),button('원본 저장 폴더',()=>action('/api/open-folder',{session:item.id},'저장 폴더를 엽니다…')));
    if(item.recording.state==='recording'&&state.capture.recording!==item.id||item.recording.state==='partial')actions.append(button('비정상 종료 복구 점검',()=>work('recover',{session:item.id})));
    if(run&&run.state!=='running')actions.append(button('시간대·사건·측정 결과 내보내기',()=>work('export-results',{session:item.id,run:run.id})));
    card.append(actions);$('result-list').append(card);
  }
}
async function openReview(session,run){
  try{const reviewer=$('reviewer').value.trim();const q=new URLSearchParams({session,run,reviewer});const response=await(await request('/api/review?'+q)).json();review={session,run,reviewer,...response};reviewIndex=Math.max(0,review.plan.findIndex(p=>!p.review));$('review-panel').hidden=false;switchPage('results');await showSample();$('review-panel').scrollIntoView({behavior:'smooth',block:'start'});}catch(e){notice(e.message);}
}
async function showSample(){
  if(!review)return;
  const rendering=++reviewRender;
  reviewReady=false;$('review-image').removeAttribute('src');
  const p=review.plan[reviewIndex];
  updateControls();
  if(!p){$('sample-caption').textContent='검토 가능한 촬영 표본이 없습니다.';$('review-image').removeAttribute('src');return;}
  $('sample-caption').textContent=`${reviewIndex+1} / ${review.plan.length} · 촬영 ${Number(p.captured_s).toFixed(3)}초`;
  $('human-label').value=p.review?.label||'';$('human-count').value=p.review?.count??'';$('review-note').value=p.review?.note||'';
  $('saved-review').textContent=p.review?`저장된 판정 · 수정하면 새 이력으로 남습니다 (${p.review.utc})`:'';
  try{const q=new URLSearchParams({session:review.session,run:review.run,sample:p.sample});const response=await request('/api/review-image?'+q);const blob=await response.blob();if(rendering!==reviewRender||!review)return;if(reviewImageUrl)URL.revokeObjectURL(reviewImageUrl);reviewImageUrl=URL.createObjectURL(blob);$('review-image').src=reviewImageUrl;await $('review-image').decode();if(rendering!==reviewRender||!review)return;reviewReady=true;updateControls();}catch(e){if(rendering!==reviewRender||!review)return;notice(e.message);}
  const v=review.validation;$('validation').replaceChildren(el('b','검토자별 비교'));
  if(!v.reviewers.length)$('validation').append(el('p','아직 저장된 판정이 없습니다.'));
  for(const r of v.reviewers)$('validation').append(el('p',`${r.reviewer}: ${r.reviewed}개 검토 · 양쪽 판정 가능 ${r.paired_known}개 · 일치율 ${fmt(r.agreement_pct,'%')} · AI 판정 불가 ${r.ai_unknown} · 사람 판정 불가 ${r.human_unknown}`));
  $('validation').append(el('p','일치율은 양쪽 모두 판정 가능한 표본 기준입니다. 소수 표본의 일치율은 정확도 검증 완료를 의미하지 않습니다.'));
}
$('load-reviewer').onclick=()=>{if(review)openReview(review.session,review.run);};
$('reviewer').oninput=()=>{if(review&&review.reviewer!==$('reviewer').value.trim()){review.reviewer=$('reviewer').value.trim();review.plan.forEach(p=>p.review=null);$('human-label').value='';$('human-count').value='';$('review-note').value='';$('saved-review').textContent='';}};
$('previous-sample').onclick=()=>{if(reviewIndex>0){reviewIndex--;showSample();}};
$('next-sample').onclick=()=>{if(reviewIndex<review.plan.length-1){reviewIndex++;showSample();}};
$('save-review').onclick=async()=>{
  if(!review||!reviewReady||!review.plan[reviewIndex])return;
  const reviewer=$('reviewer').value.trim();if(!reviewer){notice('검토자 이름 또는 코드를 입력해 주세요.');return;}
  const p=review.plan[reviewIndex], count=$('human-count').value.trim();
  const ok=await action('/api/review',{session:review.session,run:review.run,sample:p.sample,reviewer,label:$('human-label').value,count:count===''?null:Number(count),note:$('review-note').value},'사람의 판정을 저장합니다…');
  if(ok){const index=reviewIndex;const q=new URLSearchParams({session:review.session,run:review.run,reviewer});Object.assign(review,await(await request('/api/review?'+q)).json());reviewIndex=Math.min(index+1,review.plan.length-1);await showSample();notice('검토 기록을 저장했습니다. AI 원본 결과는 변경하지 않았습니다.',true);}
};
$('close-review').onclick=()=>{$('review-panel').hidden=true;review=null;reviewReady=false;reviewRender++;};
$('download-validation').onclick=()=>review&&download(review.session,review.run,'validation.csv');
$('download-validation-json').onclick=()=>review&&download(review.session,review.run,'validation.json');
$('refresh').onclick=()=>poll();
$('summary-csv').onclick=async()=>{try{const r=await request('/api/summary.csv');const url=URL.createObjectURL(await r.blob());const a=el('a');a.href=url;a.download='group_summary.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);}catch(e){notice(e.message);}};
if(!token)notice('실행하기.cmd를 눌러 프로그램을 열어 주세요.');
poll();preview();setInterval(poll,1500);
