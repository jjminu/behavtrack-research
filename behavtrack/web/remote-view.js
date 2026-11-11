'use strict';
const remoteLabels={recording:'녹화 중',saving:'원본 저장 중',uploading:'원본 전송 중',queued:'자동 분석 대기',running:'자동 분석 중',complete:'분석 완료',failed:'분석 실패',cancelled:'분석 취소',received:'서버 수신 완료',waiting:'연결 후 재시도',idle:'촬영 대기'};
function remoteElement(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function remoteValue(v,suffix=''){return v===null||v===undefined?'계산 불가':Number(v).toFixed(3)+suffix;}
function remoteRelease(node){for(const im of node.querySelectorAll('[data-object-url]'))URL.revokeObjectURL(im.dataset.objectUrl);}
async function remoteRecord(node,r,loadFile,onRetry){
 remoteRelease(node);node.replaceChildren();node.className='panel remote-record';
 node.append(remoteElement('h3',(r.group||r.local_session)+' · '+(r.cage||'')),remoteElement('p',(remoteLabels[r.state]||r.state)+' · '+r.message,'remote-status'));
 const p=remoteElement('progress');p.max=r.state==='complete'?1:r.state==='uploading'?r.bytes_total||1:r.total||1;p.value=r.state==='complete'?1:r.state==='uploading'?r.bytes_done||0:r.done||0;node.append(p);
 node.append(remoteElement('p',r.state==='uploading'?`${((r.bytes_done||0)/1024**2).toFixed(1)} / ${(r.bytes_total/1024**2).toFixed(1)} MB`:`${r.stage==='proximity'?'사료 주변 접근':r.stage==='locomotion'?'이동거리·속도':'케이지 전체 활동성'} · ${r.done||0} / ${r.total||0} 표본`,'helper'));
 if(r.test_note)node.append(remoteElement('p',r.test_note,'notice'));
 const a=r.results?.activity;
 if(a?.summary){const m=remoteElement('div',undefined,'metrics');for(const[label,value]of[['평균 변화 픽셀',remoteValue(a.summary.mean_changed_area_pct,'%')],['유효 관측',Number(a.summary.valid_s).toFixed(1)+'초'],['유효 비율',Number(a.summary.coverage_pct).toFixed(1)+'%']]){const box=remoteElement('div',undefined,'metric');box.append(remoteElement('span',label),remoteElement('strong',value));m.append(box);}node.append(m);}
 if(r.results?.proximity)node.append(remoteElement('p',(r.region_purpose==='test_region_not_food'?'테스트 영역 안 관측 비율: ':'사료 주변 접근 관측 비율: ')+remoteValue(r.results.proximity.summary.approach_pct,'%')));
 const loco=r.results?.locomotion;
 if(loco?.summary){const s=loco.summary,u=loco.unit,m=remoteElement('div',undefined,'metrics');for(const[label,value]of[['유효 구간 전체 이동거리',remoteValue(s.distance_total,' '+u)],['마리당 환산 이동거리',remoteValue(s.distance_per_animal,' '+u)],['마리당 평균 속도',remoteValue(s.mean_speed,' '+u+'/s')],['유효 추적 시간',`${s.valid_s.toFixed(2)} / ${s.observed_s.toFixed(2)}초 (${s.coverage_pct?.toFixed(1)??'—'}%)`]]){const box=remoteElement('div',undefined,'metric');box.append(remoteElement('span',label),remoteElement('strong',value));m.append(box);}node.append(remoteElement('h4','이동거리·평균 속도 추정'),m,remoteElement('p','모든 마우스가 확인된 구간의 중심점 이동 추정치입니다. 겹침·검출 누락은 판정 불가이며 추적 비율은 정확도가 아닙니다. 평균 속도 = 이동거리 합계 ÷ (마릿수 × 유효 추적 시간).','helper'));if(u==='px')node.append(remoteElement('p','실측 길이 미보정 · 픽셀 미리보기. cm 단위 그룹 비교에서 제외됩니다.','notice'));}
 const actions=remoteElement('div',undefined,'result-actions');node.append(actions);
 if(loco?.summary&&(loco.summary.coverage_pct||0)<80)node.append(remoteElement('p','유효 추적 80% 미만 · 그룹·물 비교 제외. 확인된 일부 구간의 탐색 결과입니다.','notice'));
 for(const[stage,file,label]of[['locomotion','report.html','이동성 보고서'],['locomotion','pairs.csv','거리·속도 CSV'],['locomotion','tracking.mp4','추적 확인 영상'],['activity','report.html','활동성 보고서'],['activity','pairs.csv','활동성 CSV'],['proximity','interpretation.html','접근 분석 보고서'],['proximity','samples.csv','접근 분석 CSV']]){
  if(!r.results?.[stage])continue;
  const b=remoteElement('button',label);b.onclick=async()=>{try{const blob=await loadFile({id:r.id,stage,file}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=file;a.click();setTimeout(()=>URL.revokeObjectURL(u),60000);}catch(e){node.append(remoteElement('p',e.message,'notice'));}};actions.append(b);
 }
 if(loco){const play=remoteElement('button','추적 영상 화면에서 재생');play.onclick=async()=>{play.disabled=true;try{const blob=await loadFile({id:r.id,stage:'locomotion',file:'tracking.mp4'}),video=remoteElement('video');video.controls=true;video.src=URL.createObjectURL(blob);video.dataset.objectUrl=video.src;video.className='remote-chart';play.after(video);play.remove();}catch(e){play.disabled=false;node.append(remoteElement('p',e.message,'notice'));}};actions.append(play);}
 if(a||loco){const image=remoteElement('img');image.alt=loco?'시간대별 평균 속도와 유효 추적 비율':'서버가 계산한 케이지 활동성 그래프';image.className='remote-chart';node.append(image);
  const loadChart=()=>{Promise.resolve().then(()=>loadFile({id:r.id,stage:loco?'locomotion':'activity',file:'chart.png'})).then(b=>{if(image.isConnected){if(image.dataset.objectUrl)URL.revokeObjectURL(image.dataset.objectUrl);image.src=URL.createObjectURL(b);image.dataset.objectUrl=image.src;image.hidden=false;}}).catch(()=>{if(!image.isConnected)return;image.hidden=true;const retry=remoteElement('button','그래프 다시 불러오기');retry.onclick=()=>{retry.remove();loadChart();};image.after(retry);});};loadChart();}
 if(onRetry&&['failed','cancelled'].includes(r.state)){const retry=remoteElement('button','서버 분석 다시 시도');retry.onclick=()=>onRetry(r.id);actions.append(retry);}
 node.append(remoteElement('p',r.server_session+' · '+r.id,'helper'),remoteElement('p','변화 픽셀 7%는 화면 면적의 변화이며, 전체 시간의 7%만 움직였다는 뜻이 아닙니다. 거리·속도는 별도 추적 결과로 확인하세요. 물의 인과 효과는 확정하지 않습니다.','helper'));
}
