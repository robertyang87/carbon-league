const $=s=>document.querySelector(s), preview=document.body.dataset.mode==='preview';
let identity=null,csrf='',authMode='login',toastTimer,lastRankRevision=-1,lastQueueKey='',liveCardJobs=[null,null];
const escapeHTML=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=n=>Number(n||0).toLocaleString('zh-CN',{maximumFractionDigits:1});
const signed=n=>(Number(n)>=0?'+':'')+number(n);
const pathPart=value=>encodeURIComponent(String(value));
const submissionDateFormatter=new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'});
function submissionTimeMarkup(value){if(typeof value!=='number'||!Number.isFinite(value))return '';const date=new Date(value*1000);if(!Number.isFinite(date.getTime()))return '';const parts=Object.fromEntries(submissionDateFormatter.formatToParts(date).map(part=>[part.type,part.value])),full=`${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`,label=`提交时间：${full}（北京时间）`;return `<time class="submission-time" datetime="${date.toISOString()}" title="${escapeHTML(label)}" aria-label="${escapeHTML(label)}">提交于 ${full}</time>`}
const isAdmin=()=>!loggingOut&&identity?.user?.is_admin===true;
const MAX_SOURCE_BYTES=10_000_000;
let adminCursor=null,adminRequest=0,sourceRequest=0,privateEpoch=0,identityRequest=0,loggingOut=false;
let replayConfigRequest=0,replayStatusRequest=0,replaySyncBusy=false,replayConfigLoaded=false,replaySyncStatus=null,replayStatusInFlight=false;
const states={queued:'等待评测',evaluating:'定位赛中',active:'已入榜',failed:'策略异常',running:'正在对局',done:'已完成',cancelled:'已取消',retired:'已下线',error:'评测异常'};
const REFRESH_INTERVAL=60000,STATUS_REFRESH_INTERVAL=3000,MAX_RETRY_INTERVAL=300000;
const readRequests=new Map();
let readGeneration=0,mainPollTimer=null;
function readState(path){const key=privateEpoch+':'+(identity?.user?.id??'guest')+':'+path;let state=readRequests.get(key);if(!state){state={promise:null,failures:0,nextAutomatic:0,generation:-1};readRequests.set(key,state)}return state}
function readInterval(path){return path==='/api/status'?STATUS_REFRESH_INTERVAL:REFRESH_INTERVAL}
function toast(message){$('#toast').textContent=message;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,4500)}
async function requestJson(path,options={}){const response=await fetch(path,{credentials:'same-origin',...options,headers:{...(options.body instanceof FormData?{}:{'Content-Type':'application/json'}),'X-CSRF-Token':csrf,...options.headers}});let data;try{data=await response.json()}catch{throw Error('服务暂时无法连接，请稍后重试。')}if(!response.ok)throw Error(data.error||'请求未完成');return data}
async function api(path,options={}){
 if((options.method||'GET').toUpperCase()!=='GET'){
  const data=await requestJson(path,options);readGeneration++;
  if(['/api/login','/api/register','/api/logout'].includes(path)){identityRequest++;clearPrivateViews()}
  return data;
 }
 const state=readState(path);
 if(state.promise){if(state.generation===readGeneration)return state.promise;return state.promise.catch(()=>{}).then(()=>api(path,options))}
 state.generation=readGeneration;
 state.promise=requestJson(path,options).then(data=>{state.failures=0;state.nextAutomatic=Date.now()+readInterval(path);return data},error=>{state.failures++;state.nextAutomatic=Date.now()+Math.min(MAX_RETRY_INTERVAL,REFRESH_INTERVAL*2**Math.min(3,state.failures-1));throw error}).finally(()=>{state.promise=null});
 return state.promise;
}
function automaticRead(path,load){const state=readState(path);if(document.hidden||state.promise||Date.now()<state.nextAutomatic)return Promise.resolve();state.nextAutomatic=Date.now()+readInterval(path);return load()}
async function pollMain(){
 if(preview||document.hidden)return;
 const tasks=[automaticRead('/api/status',()=>loadStatus(true)),automaticRead('/api/leaderboard',loadRank)];
 if(!$('#tab-mine').hidden)tasks.push(automaticRead('/api/mine',loadMine));
 if(!$('#tab-matches').hidden)tasks.push(automaticRead('/api/matches',loadMatches));
 if(isAdmin()&&!$('#tab-admin').hidden)tasks.push(automaticRead('/api/admin/replay-sync/status',loadReplaySyncStatus));
 await Promise.allSettled(tasks);
}
function scheduleMainPoll(){clearTimeout(mainPollTimer);mainPollTimer=null;if(!preview&&!document.hidden)mainPollTimer=setTimeout(()=>{mainPollTimer=null;pollMain().finally(scheduleMainPoll)},STATUS_REFRESH_INTERVAL)}
function showTab(name){if(name==='admin'&&!isAdmin())return;document.querySelectorAll('.tab-panel').forEach(e=>e.hidden=e.id!=='tab-'+name);document.querySelectorAll('.nav').forEach(e=>e.classList.toggle('active',e.dataset.tab===name));const panel=$('#tab-'+name);panel.setAttribute('tabindex','-1');panel.focus({preventScroll:true});panel.scrollIntoView({behavior:'instant',block:'start'});if(name==='matches')loadMatches();if(name==='mine')loadMine();if(name==='messages')loadMessages();if(name==='admin'){loadAdminStrategies();loadReplaySyncConfig()}}
function openAuth(mode){if(preview)return toast('当前为界面预览，注册登录正在接入。');authMode=mode;$('#auth-title').textContent=mode==='register'?'注册参赛':'欢迎回来';$('#auth-submit').textContent=mode==='register'?'创建账号':'登录';$('#auth-error').textContent='';$('#password').autocomplete=mode==='register'?'new-password':'current-password';$('#auth-dialog').showModal()}
function openSubmit(){if(preview)return toast('当前为界面预览，策略提交正在接入。');if(!identity?.user)return openAuth('login');$('#submit-quota').textContent=`今天还可以提交 ${identity.remaining} 次 · 北京时间零点重置`;$('#submit-error').textContent='';setSubmissionMode($('#submit-mode-paste').checked?'paste':'file');$('#submit-dialog').showModal()}
function clearSourcePreview(close=false){sourceRequest++;$('#source-content').textContent='';$('#source-title').textContent='策略源码';$('#source-download').removeAttribute('href');if(close)$('#source-dialog').close()}
function clearPrivateViews(){privateEpoch++;adminRequest++;adminCursor=null;clearSourcePreview(true);clearReplaySync();$('#admin-strategy-list').innerHTML='';$('#my-list').innerHTML='';$('#detail-body').innerHTML='';$('#detail-dialog').close();$('#admin-strategies-button').hidden=true;if($('#tab-admin').hidden===false)showTab('rank');$('#tab-admin').hidden=true;document.querySelectorAll('[data-source], [data-delete-message], [data-message-reply], [data-reply-form]').forEach(b=>b.remove())}
async function loadIdentity(){const requestId=++identityRequest,previousId=identity?.user?.id,previousAdmin=isAdmin();const session=await api('/api/session');if(requestId!==identityRequest)return;identity=session;csrf=identity.csrf;if(previousId!==identity.user?.id||(previousAdmin&&!isAdmin()))clearPrivateViews();$('#account-name').textContent=identity.user?.name||'';$('#login-button').hidden=!!identity.user;$('#register-button').hidden=!!identity.user;$('#logout-button').hidden=!identity.user;$('#rename-button').hidden=!identity.user;$('#admin-strategies-button').hidden=!isAdmin();$('#quota-label').textContent=identity.user?`今天剩余 ${identity.remaining} 次 · 00:00 重置`:'北京时间 00:00 重置';if($('#tab-messages').hidden===false)await loadMessages()}

function openRename(){if(!identity?.user)return openAuth('login');$('#rename-username').value=identity.user.name;$('#rename-error').textContent='';$('#rename-dialog').showModal()}
async function renameAccount(event){event.preventDefault();if(!identity?.user)return openAuth('login');const userId=identity.user.id,epoch=privateEpoch;$('#rename-submit').disabled=true;$('#rename-error').textContent='';try{const result=await api('/api/account/username',{method:'POST',body:JSON.stringify({username:$('#rename-username').value})});if(loggingOut||identity?.user?.id!==userId||privateEpoch!==epoch)return;if(result.user)identity={...identity,user:result.user};$('#account-name').textContent=identity.user.name;$('#rename-dialog').close();toast('用户名已更新，下次请使用新用户名登录。');const refreshes=await Promise.allSettled([loadIdentity(),loadMine(),loadRank(),loadMessages()]);if(refreshes.some(r=>r.status==='rejected'))toast('用户名已更新，下次请使用新用户名登录。部分信息暂未刷新，请刷新页面。');if(isAdmin()&&$('#tab-admin').hidden===false)await loadAdminStrategies()}catch(error){$('#rename-error').textContent=error.message}finally{$('#rename-submit').disabled=false}}
function updatePasteSize(){const bytes=new Blob([$('#strategy-code').value||'']).size;$('#paste-byte-count').textContent='UTF-8 '+bytes.toLocaleString('zh-CN')+' / 10,000,000 字节（10 MB）';$('#paste-byte-count').classList.toggle('negative',bytes>MAX_SOURCE_BYTES);$('#strategy-code').setCustomValidity(bytes>MAX_SOURCE_BYTES?'源码超过 10 MB，请缩小后提交。':'');return bytes}
function setSubmissionMode(mode){const paste=mode==='paste';$('#submit-mode-paste').checked=paste;$('#submit-mode-file').checked=!paste;$('#file-source-panel').hidden=paste;$('#paste-source-panel').hidden=!paste;$('#strategy-file').disabled=paste;$('#strategy-file').required=!paste;$('#strategy-code').disabled=!paste;$('#strategy-code').required=paste;updatePasteSize()}
function buildSubmissionData(form){const data=new FormData();for(const key of ['name','description'])data.append(key,form.elements.namedItem(key).value);if(form.elements.namedItem('is_public').checked)data.append('is_public','on');let file,filename;if($('#submit-mode-paste').checked){const text=$('#strategy-code').value;file=new Blob([text],{type:'text/x-python;charset=utf-8'});filename='submission.py'}else{file=$('#strategy-file').files?.[0];filename=file?.name}if(!file||file.size===0)throw Error('请选择非空 .py 文件，或粘贴完整策略代码。');if(file.size>MAX_SOURCE_BYTES)throw Error('源码超过 10 MB，请缩小后提交。');data.append('file',file,filename);return data}
function adminSourceButton(strategy){return isAdmin()?`<button class="row-link" data-source="${escapeHTML(strategy.id)}" data-source-name="${escapeHTML(strategy.name)}">查看源码</button>`:''}
function wireSourceButtons(){document.querySelectorAll('[data-source]').forEach(b=>b.onclick=()=>openSource(b.dataset.source,b.dataset.sourceName))}
function adminStrategyMarkup(rows){if(!isAdmin())return '';return rows.map(s=>`<article class="match-card"><div><h3>${escapeHTML(s.name)}</h3><small>${escapeHTML(s.author)} · ${escapeHTML(new Date(s.created*1000).toLocaleString('zh-CN'))}</small></div><div class="match-actions"><span class="pill">${escapeHTML(states[s.status]||s.status)}</span><span class="pill">${s.is_public===true?'源码公开':'源码不公开'}</span>${adminSourceButton(s)}<a href="/source/${pathPart(s.id)}">下载源码 ↓</a></div></article>`).join('')}
async function loadAdminStrategies(more=false){if(!isAdmin())return;const requestId=++adminRequest;try{const d=await api('/api/admin/strategies'+(more&&adminCursor?'?before='+pathPart(adminCursor):''));if(!isAdmin()||requestId!==adminRequest)return;const html=adminStrategyMarkup(d.strategies);if(more)$('#admin-strategy-list').insertAdjacentHTML('beforeend',html);else $('#admin-strategy-list').innerHTML=html||'<p class="empty">暂无提交的策略。</p>';adminCursor=d.next_before;$('#more-admin-strategies').hidden=!adminCursor;wireSourceButtons()}catch(error){if(requestId===adminRequest)toast(error.message)}}
function replaySyncControls(){
 const allowed=isAdmin()&&replayConfigLoaded&&!replaySyncBusy;
 const running=replaySyncStatus?.pending===true||replaySyncStatus?.running===true||['pending','queued','running'].includes(replaySyncStatus?.state);
 $('#replay-sync-fields').disabled=!allowed;
 $('#replay-sync-save').disabled=!allowed;
 $('#replay-sync-run').disabled=!allowed||running;
 $('#replay-sync-refresh').disabled=replaySyncBusy||!isAdmin();
}
function clearReplaySync(){
 replayConfigRequest++;replayStatusRequest++;replaySyncBusy=false;replayConfigLoaded=false;replaySyncStatus=null;replayStatusInFlight=false;
 $('#replay-sync-enabled').checked=false;$('#replay-sync-repo').value='';$('#replay-sync-days').value='7';$('#replay-sync-limit').value='400';$('#replay-sync-interval').value='10';$('#replay-sync-strategies').value='';$('#replay-sync-authors').value='';
 for(const id of ['replay-sync-state','replay-sync-attempt','replay-sync-success','replay-sync-count','replay-sync-credential','replay-sync-result','replay-sync-error'])$('#'+id).textContent='';
 replaySyncControls();
}
function syncDate(value){if(value===null||value===undefined||value==='')return '尚无记录';const date=new Date(typeof value==='number'?value*1000:value);return Number.isFinite(date.getTime())?date.toLocaleString('zh-CN'):'尚无记录'}
function renderReplaySyncStatus(status={}){
 if(!isAdmin())return;
 replaySyncStatus=status;
 const labels={idle:'等待同步',pending:'等待执行',queued:'等待执行',running:'同步中',success:'同步成功',synced:'同步成功',failed:'同步失败，下次重试',error:'同步失败，下次重试',unchanged:'没有新对局',no_changes:'没有新对局',disabled:'已停用',skipped:'本次跳过'};
 $('#replay-sync-state').textContent=labels[status.state]||String(status.state||'等待同步');
 $('#replay-sync-attempt').textContent=syncDate(status.last_attempt);
 $('#replay-sync-success').textContent=syncDate(status.last_success);
 $('#replay-sync-count').textContent=String(Number.isFinite(Number(status.exported_games))?Math.max(0,Number(status.exported_games)):0)+' 局';
 $('#replay-sync-credential').textContent=status.credential_configured===true?'服务器已配置凭据':'未配置凭据（需在服务器后台设置）';
 $('#replay-sync-result').textContent=String(status.message||'');
 replaySyncControls();
}
function renderReplaySyncConfig(config={}){
 $('#replay-sync-enabled').checked=config.enabled===true;$('#replay-sync-repo').value=String(config.repo_url||'');
 $('#replay-sync-days').value=String(config.days??7);$('#replay-sync-limit').value=String(config.max_games??400);$('#replay-sync-interval').value=String((config.interval_seconds??600)/60);
 $('#replay-sync-strategies').value=Array.isArray(config.strategy_ids)?config.strategy_ids.join('\n'):'';$('#replay-sync-authors').value=Array.isArray(config.authors)?config.authors.join('\n'):'';
 replayConfigLoaded=true;
}
async function loadReplaySyncConfig(){
 if(!isAdmin()||replaySyncBusy)return;
 const requestId=++replayConfigRequest,epoch=privateEpoch,userId=identity.user.id;
 $('#replay-sync-error').textContent='';
 try{const data=await api('/api/admin/replay-sync/config');if(!isAdmin()||epoch!==privateEpoch||userId!==identity.user.id||requestId!==replayConfigRequest)return;renderReplaySyncConfig(data.config);renderReplaySyncStatus(data.status)}
 catch(error){if(isAdmin()&&epoch===privateEpoch&&requestId===replayConfigRequest)$('#replay-sync-error').textContent=error.message}
}
async function loadReplaySyncStatus(){
 if(!isAdmin()||$('#tab-admin').hidden||replaySyncBusy||replayStatusInFlight)return;
 const requestId=++replayStatusRequest,epoch=privateEpoch,userId=identity.user.id;replayStatusInFlight=true;
 try{const data=await api('/api/admin/replay-sync/status');if(!isAdmin()||epoch!==privateEpoch||userId!==identity.user.id||requestId!==replayStatusRequest)return;renderReplaySyncStatus(data.status)}
 catch(error){if(isAdmin()&&epoch===privateEpoch&&requestId===replayStatusRequest)$('#replay-sync-error').textContent=error.message}
 finally{if(requestId===replayStatusRequest)replayStatusInFlight=false}
}
function replaySyncPayload(){
 const list=id=>[...new Set($('#'+id).value.split(/[\r\n,，]+/).map(s=>s.trim()).filter(Boolean))];
 return {enabled:$('#replay-sync-enabled').checked,repo_url:$('#replay-sync-repo').value.trim(),days:Number($('#replay-sync-days').value),max_games:Number($('#replay-sync-limit').value),strategy_ids:list('replay-sync-strategies'),authors:list('replay-sync-authors'),interval_seconds:Number($('#replay-sync-interval').value)*60};
}
async function updateReplaySync(run=false){
 if(!isAdmin()||!replayConfigLoaded||replaySyncBusy)return;
 const requestId=++replayConfigRequest,epoch=privateEpoch,userId=identity.user.id;
 replayStatusRequest++;replayStatusInFlight=false;replaySyncBusy=true;replaySyncControls();$('#replay-sync-error').textContent='';
 try{const data=await api('/api/admin/replay-sync/'+(run?'run':'config'),{method:'POST',body:JSON.stringify(run?{}:replaySyncPayload())});if(!isAdmin()||epoch!==privateEpoch||userId!==identity.user.id||requestId!==replayConfigRequest)return;if(data.config)renderReplaySyncConfig(data.config);renderReplaySyncStatus(data.status);toast(run?'同步请求已提交，结果会自动刷新。':'同步配置已保存。')}
 catch(error){if(isAdmin()&&epoch===privateEpoch&&requestId===replayConfigRequest)$('#replay-sync-error').textContent=error.message}
 finally{if(requestId===replayConfigRequest){replaySyncBusy=false;replaySyncControls()}}
}
async function openSource(id,name){if(!isAdmin())return;const requestId=++sourceRequest,userId=identity.user.id;$('#source-title').textContent='源码 · '+String(name||id);$('#source-content').textContent='正在读取…';$('#source-download').href='/source/'+pathPart(id);$('#source-dialog').showModal();try{const response=await fetch('/source/'+pathPart(id),{credentials:'same-origin',cache:'no-store'});if(!response.ok)throw Error('无法读取源码（HTTP '+response.status+'）。');const content=await response.text();if(!isAdmin()||identity.user.id!==userId||requestId!==sourceRequest)return;$('#source-content').textContent=content}catch(error){if(isAdmin()&&requestId===sourceRequest)$('#source-content').textContent=error.message}}
function rankingPhaseText(strategy){const labels={top:'前排定位完成',focus:'定位完成',low:'低优先级跑榜',cap:'已达定位上限',withdrawn:'已退出'};return strategy.ranking_phase==='placement'?'定位中 · 已 '+number(strategy.games)+' 局（10–40 局）':strategy.ranking_phase==='ladder'?(labels[strategy.ranking_verdict]||'日常跑榜'):'历史 Elo'}
const oneDecimal=n=>Number(n||0).toLocaleString('zh-CN',{minimumFractionDigits:1,maximumFractionDigits:1});
function ratingMarkup(strategy){const sd=Number(strategy.rating_sd),show=['placement','ladder'].includes(strategy.ranking_phase)&&strategy.rating_sd!=null&&Number.isFinite(sd)&&sd>=0;return `<span class="rating-main">${oneDecimal(strategy.rating)}</span><small class="rating-sd"${show?' title="模型估计的不确定度，不保证真实置信区间"':''}>${show?`± ${oneDecimal(sd)}`:''}</small>`}
const rankOpen=new Set(),RANK_PAGE=50;
let rankShowAll=false,rankBoard='light',rankBoardRows=[];
function rankChangeMarkup(r){const change=Number(r.rank_change);if(!change)return '';return `<small class="rank-change ${change>0?'up':'down'}" title="与今日首次排名相比">${change>0?'▲':'▼'}${Math.abs(change)}</small>`}
function rankRows(rows){$('#leaderboard').innerHTML=rows.length?rows.slice(0,rankShowAll?rows.length:RANK_PAGE).map((r,i)=>`<tr class="rank-row${i<3?' top-rank rank-'+(i+1):''}"><td><span class="rank-number ${i<3?'top':''}">${escapeHTML(String(r.rank||i+1).padStart(2,'0'))}</span>${rankChangeMarkup(r)}</td><td><div class="strategy-name"><button class="rank-toggle" data-rank-toggle aria-expanded="${rankOpen.has(r.id)}" aria-label="展开或收起详情">${rankOpen.has(r.id)?'▾':'▸'}</button>${escapeHTML(r.name)}${preview?'<span class="preview-tag">示例</span>':r.ranking_phase==='placement'?'<span class="preview-tag">定位中 · '+escapeHTML(r.games)+' 局</span>':''}</div><div class="strategy-meta"><span class="author">${escapeHTML(r.author)}</span></div></td><td class="elo">${ratingMarkup(r)}</td><td>${r.games?(100*r.wins/r.games).toFixed(1):'0'}%</td><td>${escapeHTML(r.games)}</td><td class="${r.margin>=0?'positive':'negative'}">${signed(r.games?r.margin/r.games:0)}</td><td>${r.is_public===true?`<a class="row-link public-source" href="/source/${pathPart(r.id)}" title="下载作者公开的策略源码" aria-label="下载公开源码">源码 ↓</a>`:''}${adminSourceButton(r)}<button class="row-link" data-strategy="${escapeHTML(r.id)}" title="查看对局">↗</button></td></tr><tr class="rank-detail" data-rank-id="${escapeHTML(r.id)}"${rankOpen.has(r.id)?'':' hidden'}><td></td><td colspan="6"><div class="strategy-description">${submissionTimeMarkup(r.created)}${r.description?`<div>${escapeHTML(r.description)}</div>`:''}</div></td></tr>`).join('')+(rows.length>RANK_PAGE&&!rankShowAll?`<tr class="rank-more"><td colspan="7"><button class="text-button" data-rank-more>查看全部 ${rows.length} 名</button></td></tr>`:''):'<tr><td colspan="7" class="empty">排行榜等待第一位挑战者。提交策略，首局完成即登榜。</td></tr>';document.querySelectorAll('[data-strategy]').forEach(b=>b.onclick=()=>{if(preview)return toast('示例排名仅用于展示布局。');showStrategyMatches(b.dataset.strategy)});document.querySelectorAll('[data-rank-more]').forEach(b=>b.onclick=()=>{rankShowAll=true;rankRows(rows)});document.querySelectorAll('[data-rank-toggle]').forEach(b=>b.onclick=()=>{const detail=b.closest('tr').nextElementSibling,open=detail.hidden;detail.hidden=!open;if(open)rankOpen.add(detail.dataset.rankId);else rankOpen.delete(detail.dataset.rankId);b.textContent=open?'▾':'▸';b.setAttribute('aria-expanded',String(open))});wireSourceButtons()}
function showRankBoard(){rankRows(rankBoardRows.filter(r=>r.board===rankBoard).sort((a,b)=>a.board_rank-b.board_rank).map(r=>({...r,rank:r.board_rank,rank_change:r.board_change})))}
function chooseRankBoard(board){rankBoard=board;rankShowAll=false;document.querySelectorAll('[data-rank-board]').forEach(b=>{const on=b.dataset.rankBoard===board;b.classList.toggle('selected',on);b.setAttribute('aria-pressed',String(on))});if(!preview)showRankBoard()}
async function loadRank(){if(preview)return;try{const d=await api('/api/leaderboard');rankBoardRows=[...d.strategies,...(d.extra||[])];showRankBoard()}catch(e){$('#leaderboard').innerHTML=`<tr><td colspan="7" class="empty">${escapeHTML(e.message)}</td></tr>`}}
function activeStrategyName(active,slot){
 const name=active?.['name_'+slot];
 return typeof name==='string'&&name?name:'策略'+slot.toUpperCase();
}
function matchKindText(active,progress){
 if(!active)return '';
 return active.game_count>1?`历史双局 · 第 ${progress?.game||1} / ${active.game_count} 局`:active.kind==='newrank-placement'?'入场定位 · 信息量匹配':active.kind==='newrank-ladder'?'日常跑榜 · 信息量匹配':active.kind==='placement-top3'?`历史 Top 3 首轮 · 第 ${active.opening_rank} 场`:'单局对战 · 随机位置';
}
function renderLiveScores(active,progress,suffix=''){
 const values=['a','b'].map(slot=>{const value=active?progress?.scores?.[slot]:null;return typeof value==='number'&&Number.isFinite(value)&&value>=0?value:null});
 const maximum=Math.max(0,...values.filter(value=>value!==null)),scale=maximum||1,total=values.reduce((sum,value)=>sum+(value||0),0);
 for(const [index,slot] of ['a','b'].entries()){
  const name=active?activeStrategyName(active,slot):'等待策略',value=values[index],known=value!==null;
  const label=$('#live-score-'+slot+suffix);label.textContent=known?number(value):'—';label.title=known?number(value):'当前资金尚未就绪';
  const strategy=$('#live-name-'+slot+suffix);strategy.textContent=name;strategy.title=name;
  $('#live-bar-'+slot+suffix).style.height=(known?100*(value/scale):0).toFixed(2)+'%';
  const meter=$('#live-meter-'+slot+suffix);meter.style.width=(total?100*((known?value:0)/total):50).toFixed(2)+'%';const carried=active?progress?.in_transit?.[slot]:null;$('#live-carbon-'+slot+suffix).textContent=typeof carried==='number'&&Number.isFinite(carried)?carried.toFixed(1):'—';meter.setAttribute('aria-valuemax',String(scale));meter.setAttribute('aria-valuetext',known?number(value)+' 当前资金':'当前资金尚未就绪');meter.setAttribute('aria-label',name+' 当前资金');
  if(known)meter.setAttribute('aria-valuenow',String(value));else meter.removeAttribute('aria-valuenow');
 }
}
function shortName(name,limit=20){
  const text=String(name),width=c=>c.charCodeAt(0)>255?2:1,chars=[...text];
  if(chars.reduce((sum,c)=>sum+width(c),0)<=limit)return text;
  const take=(list,room)=>{const out=[];for(const c of list){room-=width(c);if(room<0)break;out.push(c)}return out};
  const head=take(chars,Math.ceil((limit-1)/2)),tail=take([...chars].reverse(),Math.floor((limit-1)/2)).reverse();
  return head.join('')+'*'+tail.join('');
}
function actionTimingParts(stats){
 const calls=Number(stats?.calls)||0,total=Number(stats?.total_seconds)||0,last=Number(stats?.last_seconds);
 if(!calls)return ['上一步 —','平均 —'];
 return [`上一步 ${stats?.last_seconds!=null&&Number.isFinite(last)?last.toFixed(3)+' 秒':'—'}`,`平均 ${(total/calls).toFixed(3)} 秒`];
}
function actionTimingText(slot,stats){return [slot,...actionTimingParts(stats)].join(' · ')}
function renderTiming(slot,suffix,name,stats){
 const label=$('#timing-'+slot+suffix),[step,average]=name?actionTimingParts(stats):['',''];
 label.textContent=name?shortName(name):(slot==='a'?'策略一':'策略二');label.title=name||'';
 $('#timing-step-'+slot+suffix).textContent=step;
 $('#timing-average-'+slot+suffix).textContent=average;
}
function progressText(p,online=true,active=null){
 const stamp=Number(p.updated_at),start=Number(p.game_started_at||p.started_at);
 const total=Math.max(0,(p.game_finished_at||stamp)-start);
 const elapsed=Number.isFinite(total)?total.toFixed(1):'—';
 if(!online)return `评测服务暂离线 · 最后记录本局 ${elapsed} 秒`;
 const phases={starting:'准备对局',starting_agent:'启动策略',advancing:'推进环境',exporting:'生成回放',done:'已完成',failed:'评测中断'};
 let detail=phases[p.phase]||'评测中';
 if(p.phase==='acting'&&p.waiting_slot){const wait=Math.max(0,stamp-Number(p.waiting_since));detail=`${activeStrategyName(active,p.waiting_slot)} 决策中 · 本步已等待 ${wait.toFixed(2)} 秒（采样）`;}
 return `本局已用 ${elapsed} 秒 · ${detail}`;
}
const RESULT_HOLD=5000,resultSeen=new Map();
function heldResults(status){const rows=Array.isArray(status.recent_results)?status.recent_results:[],current=Date.now(),ids=new Set(rows.map(row=>row.id));for(const id of resultSeen.keys())if(!ids.has(id))resultSeen.delete(id);return rows.filter(row=>{if(!resultSeen.has(row.id))resultSeen.set(row.id,current);return current-resultSeen.get(row.id)<RESULT_HOLD})}
function statusGames(status){const incoming=Array.isArray(status.active_games)?status.active_games.slice(0,2):status.active?[{...status.active,progress:status.progress}]:[],held=heldResults(status),byId=new Map([...incoming,...held].map(game=>[game.id,game])),cards=liveCardJobs.map(id=>byId.get(id)||null),assigned=new Set(cards.filter(Boolean).map(game=>game.id));for(const game of [...incoming,...held]){if(assigned.has(game.id))continue;const empty=cards.findIndex(value=>value===null);if(empty<0)break;cards[empty]=game;assigned.add(game.id)}liveCardJobs=cards.map(game=>game?.id??null);return cards}
const resultReasons={timeout:'对手超时',error:'对手报错',invalid:'对手违规'};
function renderResult(active,suffix){
 const box=$('#arena-result'+suffix),result=active?.result;
 $('#arena-card'+suffix)?.classList?.toggle('finished',!!result);
 if(!box)return;
 box.hidden=!result;
 if(!result)return;
 for(const slot of ['a','b']){const score=active.progress?.scores?.[slot];if(typeof score==='number'&&Number.isFinite(score)){const label=$('#live-score-'+slot+suffix);label.textContent=number(score);label.title=number(score)}}
 if(box.dataset.job===active.id)return;
 box.dataset.job=active.id;
 const head=result.winner==='draw'?'<b>平局</b>':`<b class="live-${result.winner==='a'?'blue':'red'}">${escapeHTML(activeStrategyName(active,result.winner))} 获胜</b>`,reason=resultReasons[result.reason]||'';
 box.innerHTML=`<div>${head}${reason?` · ${reason}`:''}</div>${active.replay_available?'<div><button type="button" class="text-link result-replay">看回放 ↗</button></div>':''}`;
 const replay=box.querySelector('.result-replay');
 if(replay)replay.onclick=()=>openReplay(active.id,Number(active.game_count)||1);
}
function idleRankingText(status){if(!status.worker_online)return '评测服务暂离线';const rank=status.ranking;if(rank?.enabled&&rank.converged===true&&!rank.placement_count&&!status.queued)return '榜单已收敛，等待新提交';if(status.resource_wait?.ranked?.waiting===true)return '服务器资源紧张，等待启动新对局';if(rank?.enabled&&rank.placement_count>0)return '等待可用对手 · 入场定位中';if(rank?.enabled&&rank.unsettled_count>0)return '等待可用对手 · 排名仍在复核';return '空闲 · 等待定位或跑榜复核'}
function renderStatusSlot(active,progress,index,status){const suffix=index?'-2':'';$('#arena-card'+suffix)?.classList?.toggle('idle',!active);renderLiveScores(active,progress,suffix);$('#arena-label'+suffix).textContent=active?'MATCH IN PROGRESS':'等待下一场';$('#current-match'+suffix).textContent=active?`${activeStrategyName(active,'a')} vs ${activeStrategyName(active,'b')}`:'等待安排对局';$('#current-match-detail'+suffix).textContent=matchKindText(active,progress);$('#round-label'+suffix).textContent=active&&progress?`${progress.steps||0} / ${progress.total_steps||299}`:'— / 299';$('#game-progress'+suffix).style.width=active&&progress?`${Math.max(0,Math.min(100,100*(Number(progress.steps)||0)/(Number(progress.total_steps)||299)))}%`:'0%';for(const slot of ['a','b'])renderTiming(slot,suffix,active?activeStrategyName(active,slot):'',progress?.actions?.[slot]);renderResult(active,suffix);$('#queue-label'+suffix).textContent='第 '+(index+1)+' 路';$('#worker-status'+suffix).textContent=active&&progress?progressText(progress,status.worker_online,active):active?'正在准备对局':idleRankingText(status)}
async function loadStatus(automatic=false){if(preview)return;try{const s=await api('/api/status'),games=statusGames(s);const revision=String(s.completed_games??s.completed_pairs*2)+':'+(s.rating_version||'0');if(revision!==lastRankRevision){lastRankRevision=revision;automatic===true?automaticRead('/api/leaderboard',loadRank):loadRank()}const queueKey=[revision,...games.map(game=>game?.id||''),s.queued].join(':');if(queueKey!==lastQueueKey){lastQueueKey=queueKey;if(!$('#tab-mine').hidden)automatic===true?automaticRead('/api/mine',loadMine):loadMine();if(!$('#tab-matches').hidden)automatic===true?automaticRead('/api/matches',loadMatches):loadMatches()}$('#stat-strategies').textContent=number(s.total_strategies);$('#stat-games').textContent=number(s.completed_games??s.completed_pairs*2);$('#worker-overview').textContent=s.worker_online?(s.worker_ready?'评测服务在线':'评测服务检查中'):'评测服务暂离线 · 已提交任务会保留';$('#queue-overview').textContent='排队 '+number(s.queued);$('#running-overview').textContent=number(s.running_count??games.filter(Boolean).length)+' / '+number(s.slots||2)+' 路运行';for(let index=0;index<2;index++)renderStatusSlot(games[index]||null,games[index]?.progress||null,index,s)}catch{$('#worker-overview').textContent='连接中断，正在重试…';for(const suffix of ['', '-2'])$('#worker-status'+suffix).textContent='连接中断，正在重试…'}}
function matchDataLinks(r){if(!identity?.user||!r.replay_available||r.state!=='done')return '';const games=r.game_count>1?[1,2]:[1];return games.map(game=>`<a class="match-data-link" href="/data/${pathPart(r.id)}/${game}" title="下载压缩的 JSON 对局数据">${games.length>1?'第 '+game+' 局数据 ↓':'JSON 数据 ↓'}</a>`).join('')}
function matchRatingMarkup(row){if(row.result?.some(result=>result.rating_review?.reason==='platform-initialization-miscount-20261025'))return '<small class="rating-change">初始化计时误判 · 本局不计评分</small>';if(!row.rated)return '';const details=row.result?.find(result=>result.elo?.rated)?.elo||{},method=String(row.rating_method||details.method||''),modern=method.startsWith('newrank-')||String(row.kind||'').startsWith('newrank-');const a0=row.rating_a_before??details.a_before,a1=row.rating_a_after??details.a_after,b0=row.rating_b_before??details.b_before,b1=row.rating_b_after??details.b_after;return `<small class="rating-change">${modern?'评分':'历史 Elo'}：${number(a0)} → ${number(a1)}（${signed(a1-a0)}） / ${number(b0)} → ${number(b1)}（${signed(b1-b0)}）</small>`}
function matchMarkup(rows){return rows.length?rows.map(r=>{const results=r.result||[];const text=results.length?results.map((x,i)=>`${r.game_count>1?"第"+(i+1)+"局 ":""}${number(x.scores.a)} : ${number(x.scores.b)}`).join(' · '):'对手将在开始评测时匹配';const label=r.kind==='newrank-placement'?'入场定位 · ':r.kind==='newrank-ladder'?'日常跑榜 · ':r.kind==='placement-top3'?'历史 Top 3 首轮 · ':r.game_count>1?'历史双局 · ':'单局 · ';return `<article class="match-card"><div><h3>${escapeHTML(r.name_a)} <span class="muted">vs</span> ${escapeHTML(r.name_b||'等待匹配')}</h3><small>${escapeHTML(text)}</small>${matchRatingMarkup(r)}</div><div class="match-actions"><span class="pill">${label}${escapeHTML(states[r.state]||r.state)}</span>${r.replay_available?`<button data-replay="${escapeHTML(r.id)}" data-games="${escapeHTML(r.game_count||2)}">回放 ↗</button>`:''}${matchDataLinks(r)}${identity?.user?`<button data-diagnostic="${escapeHTML(r.id)}">我的日志</button>`:''}</div></article>`}).join(''):'<div class="empty">还没有对局记录。</div>'}
function wireMatches(){document.querySelectorAll('[data-replay]').forEach(b=>b.onclick=()=>openReplay(b.dataset.replay,Number(b.dataset.games)));document.querySelectorAll('[data-diagnostic]').forEach(b=>b.onclick=()=>openDiagnostic(b.dataset.diagnostic))}
async function loadMatches(){if(preview){$('#match-list').innerHTML='<div class="empty">正式评测接入后，这里将展示双方比分、评分变化和可视化回放。</div>';return}try{$('#match-list').innerHTML=matchMarkup((await api('/api/matches')).matches);wireMatches()}catch(e){toast(e.message)}}
async function showStrategyMatches(id){try{$('#detail-title').textContent='策略最近对局';$('#detail-body').innerHTML=matchMarkup((await api('/api/matches?strategy='+encodeURIComponent(id))).matches);$('#detail-dialog').showModal();wireMatches()}catch(e){toast(e.message)}}
function openReplay(id,games=2){$('#detail-title').textContent=games>1?'历史双局回放':'单局回放';$('#detail-body').innerHTML=`<div class="hero-actions detail-games">${games>1?'<button class="button small" data-game="1">第 1 局</button>':''}${games>1?'<button class="refresh-button" data-game="2">第 2 局</button>':''}</div><p class="muted">A/B 代表策略身份，起始位置以回放为准。</p><iframe sandbox="allow-scripts" title="游戏回放" src="/replay/${encodeURIComponent(id)}/1"></iframe>`;$('#detail-dialog').showModal();document.querySelectorAll('[data-game]').forEach(b=>b.onclick=()=>$('#detail-body iframe').src=`/replay/${encodeURIComponent(id)}/${b.dataset.game}`)}
async function openDiagnostic(id){if(loggingOut)return;if(!identity?.user)return openAuth('login');const userId=identity.user.id,epoch=privateEpoch;try{const d=await api('/api/jobs/'+encodeURIComponent(id)+'/diagnostics');if(loggingOut||identity?.user?.id!==userId||privateEpoch!==epoch)return;$('#detail-title').textContent='我的策略日志';$('#detail-body').innerHTML='<pre class="diagnostic"></pre>';$('#detail-body pre').textContent=JSON.stringify(d,null,2);$('#detail-dialog').showModal()}catch(e){toast(e.message)}}
function mineMarkup(strategies){return strategies.length?strategies.map(s=>`<article class="match-card"><div><h3>${escapeHTML(s.name)}</h3>${submissionTimeMarkup(s.created)}<small>${escapeHTML(s.description||'尚未填写说明')}</small>${s.error?`<p class="negative">${escapeHTML(s.error)}</p>`:''}</div><div><span class="elo">${ratingMarkup(s)}</span><small> · ${escapeHTML(s.games)} 计分局 · ${escapeHTML(rankingPhaseText(s))} · ${s.rank?(s.board==='heavy'?'大文件榜':'轻量榜')+'第 '+escapeHTML(s.rank)+' 名':'尚无排名'}</small></div><div class="match-actions"><span class="pill">${escapeHTML(states[s.status]||s.status)}</span><span class="pill">${s.is_public===true?'源码公开':'源码不公开'}</span><a href="/source/${pathPart(s.id)}">源码 ↓</a><button data-strategy="${escapeHTML(s.id)}">对局 ↗</button><button data-visibility="${escapeHTML(s.id)}" data-public="${s.is_public===true?'false':'true'}">${s.is_public===true?'取消公开':'公开源码'}</button><button class="strategy-delete" data-delete-strategy="${escapeHTML(s.id)}" data-delete-name="${escapeHTML(s.name)}">删除</button></div></article>`).join(''):'<div class="empty">还没有策略，上传你的第一个 submission.py 吧。</div>'}
async function loadMine(){if(loggingOut)return;if(preview){$('#my-list').innerHTML='<div class="empty">登录后可查看所有自己的策略、定位赛进度、榜单外排名和异常日志。</div>';return}if(!identity?.user){$('#my-list').innerHTML='<div class="empty">请先登录，查看你的策略。</div>';return}const userId=identity.user.id,epoch=privateEpoch;try{const {strategies}=await api('/api/mine');if(loggingOut||identity?.user?.id!==userId||privateEpoch!==epoch)return;$('#my-list').innerHTML=mineMarkup(strategies);document.querySelectorAll('#my-list [data-strategy]').forEach(b=>b.onclick=()=>showStrategyMatches(b.dataset.strategy));document.querySelectorAll('#my-list [data-visibility]').forEach(b=>b.onclick=()=>setSourceVisibility(b.dataset.visibility,b.dataset.public==='true',b));document.querySelectorAll('#my-list [data-delete-strategy]').forEach(b=>b.onclick=()=>deleteStrategy(b.dataset.deleteStrategy,b.dataset.deleteName,b))}catch(e){toast(e.message)}}
async function deleteStrategy(id,name,button){if(!identity?.user)return openAuth('login');if(!confirm('删除“'+name+'”？\n\n它会从榜单下线并停止比赛，不能撤销。已有对局和回放保留，今天的提交次数不返还，策略名不能再用。'))return;button.disabled=true;try{await api('/api/strategies/'+pathPart(id)+'/delete',{method:'POST',body:'{}'});await loadMine();loadRank();toast('策略已删除。')}catch(e){toast(e.message);button.disabled=false}}
async function setSourceVisibility(id,isPublic,button){if(!identity?.user)return openAuth('login');button.disabled=true;try{await api('/api/strategies/'+pathPart(id)+'/visibility',{method:'POST',body:JSON.stringify({is_public:isPublic})});await loadMine();loadRank();toast(isPublic?'源码已公开，其他参赛者可下载。':'源码已取消公开；已经下载的副本无法收回。')}catch(e){toast(e.message)}finally{button.disabled=false}}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$('#'+b.dataset.close).close());$('#login-button').onclick=()=>openAuth('login');$('#register-button').onclick=()=>openAuth('register');$('#submit-cta').onclick=openSubmit;$('#mine-submit').onclick=openSubmit;$('#join-button').onclick=openSubmit;document.querySelectorAll('[data-rank-board]').forEach(b=>b.onclick=()=>chooseRankBoard(b.dataset.rankBoard));$('#mine-button').onclick=()=>showTab('mine');$('#refresh-button').onclick=()=>{loadRank();loadStatus();toast(preview?'当前展示示例布局。':'已刷新')};$('#refresh-matches').onclick=loadMatches;
$('#logout-button').onclick=async()=>{loggingOut=true;identityRequest++;clearPrivateViews();$('#strategy-code').value='';$('#strategy-file').value='';setSubmissionMode('file');$('#submit-dialog').close();$('#rename-dialog').close();try{await api('/api/logout',{method:'POST',body:'{}'});identity=null;$('#account-name').textContent='';$('#rename-button').hidden=true;$('#logout-button').hidden=true;$('#login-button').hidden=false;$('#register-button').hidden=false;showTab('rank');await loadIdentity();await loadRank()}catch(e){toast(e.message)}finally{loggingOut=false}};
$('#auth-form').onsubmit=async e=>{e.preventDefault();$('#auth-submit').disabled=true;try{await api('/api/'+authMode,{method:'POST',body:JSON.stringify({username:$('#username').value,password:$('#password').value})});$('#password').value='';await loadIdentity();await loadRank();$('#auth-dialog').close();toast(authMode==='register'?'注册成功，欢迎参赛！':'登录成功')}catch(error){$('#auth-error').textContent=error.message}finally{$('#auth-submit').disabled=false}};
$('#submit-form').onsubmit=async e=>{e.preventDefault();$('#upload-button').disabled=true;try{await api('/api/submit',{method:'POST',body:buildSubmissionData(e.target)});e.target.reset();setSubmissionMode('file');$('#submit-dialog').close();await loadIdentity();showTab('mine');loadStatus();toast('提交成功，策略已进入评测队列。')}catch(error){$('#submit-error').textContent=error.message}finally{$('#upload-button').disabled=false}};
for(let i=0;i<225;i++){const cell=document.createElement('i');if((i*37+11)%23<6)cell.classList.add('rich');if([35,54,80,109,145,181].includes(i))cell.classList.add('tree');if([66,92,122,156].includes(i))cell.classList.add('worker');if([92,156].includes(i))cell.classList.add('enemy');$('#board-art').append(cell)}
let messageCursor=null;
function canDeleteMessage(message){return !loggingOut&&!!identity?.user&&message?.deleted!==true&&(isAdmin()||message?.can_delete===true)}
function messageDeleteMarkup(message,label='留言'){return canDeleteMessage(message)?`<button type="button" class="message-delete" data-delete-message="${escapeHTML(message.id)}" data-can-delete="${message.can_delete===true?'true':'false'}" aria-label="删除这条${label}">删除</button>`:''}
function messageRepliesMarkup(rows){return (Array.isArray(rows)?rows:[]).map(reply=>`<div class="message-reply-item"><div class="message-meta"><strong>${escapeHTML(reply.author)}</strong><div class="message-tools"><time>${escapeHTML(new Date(reply.created*1000).toLocaleString('zh-CN'))}</time>${messageDeleteMarkup(reply,'回复')}</div></div><p>${escapeHTML(reply.content)}</p></div>`).join('')}
function messageReplyFormMarkup(message){if(loggingOut||!identity?.user||message.deleted===true)return '';return `<form id="reply-form-${escapeHTML(message.id)}" class="message-reply-form" data-reply-form="${escapeHTML(message.id)}" hidden><label for="reply-content-${escapeHTML(message.id)}">回复这条留言</label><textarea id="reply-content-${escapeHTML(message.id)}" name="content" rows="2" maxlength="1000" required placeholder="写下你的回复（与留言共用发布额度）"></textarea><p class="form-error reply-error" role="status"></p><div class="reply-form-actions"><span>留言与回复合计：每分钟 1 条，每 24 小时 10 条。</span><button type="button" class="text-button" data-cancel-reply>取消</button><button type="submit" class="button small">发送回复 ↗</button></div></form>`}
function messageMarkup(rows){return rows.map(m=>{const replies=Array.isArray(m.replies)?m.replies:[],count=Number.isFinite(Number(m.reply_count))?Math.max(0,Number(m.reply_count)):replies.length;return `<article class="message-card${m.deleted===true?' message-deleted':''}"><div class="message-meta"><strong>${m.deleted===true?'已删除留言':escapeHTML(m.author)}</strong><div class="message-tools"><time>${escapeHTML(new Date(m.created*1000).toLocaleString('zh-CN'))}</time>${messageDeleteMarkup(m)}</div></div><p class="message-content">${m.deleted===true?'原留言已删除，已有回复保留。':escapeHTML(m.content)}</p><div class="message-thread-tools">${!loggingOut&&identity?.user&&m.deleted!==true?`<button type="button" class="message-reply-button" data-message-reply="${escapeHTML(m.id)}" aria-expanded="false" aria-controls="reply-form-${escapeHTML(m.id)}">回复</button>`:''}${count?`<span class="reply-count">${escapeHTML(count)} 条回复</span>`:''}</div><div class="message-replies"${!replies.length&&!m.replies_next_before?' hidden':''}>${m.replies_next_before?`<button type="button" class="load-older-replies" data-older-replies="${escapeHTML(m.id)}" data-before="${escapeHTML(m.replies_next_before)}">查看更早的回复 ↑</button>`:''}<div class="message-reply-list">${messageRepliesMarkup(replies)}</div></div>${messageReplyFormMarkup(m)}</article>`}).join('')}
function wireMessageActions(){document.querySelectorAll('[data-delete-message]').forEach(b=>b.onclick=()=>deleteMessage(b.dataset.deleteMessage,b,{can_delete:b.dataset.canDelete==='true'}));document.querySelectorAll('[data-message-reply]').forEach(b=>b.onclick=()=>openMessageReply(b));document.querySelectorAll('[data-reply-form]').forEach(form=>{form.onsubmit=sendMessageReply;form.querySelector('[data-cancel-reply]').onclick=()=>closeMessageReply(form)});document.querySelectorAll('[data-older-replies]').forEach(b=>b.onclick=()=>loadOlderReplies(b))}
async function loadMessages(more=false){if(loggingOut)return;const userId=identity?.user?.id,epoch=privateEpoch;try{const d=await api('/api/messages'+(more&&messageCursor?'?before='+pathPart(messageCursor):''));if(loggingOut||identity?.user?.id!==userId||privateEpoch!==epoch)return;const html=messageMarkup(d.messages);if(more)$('#message-list').insertAdjacentHTML('beforeend',html);else $('#message-list').innerHTML=html||'<p class="empty">还没有留言，欢迎提出第一个建议。</p>';messageCursor=d.next_before;$('#more-messages').hidden=!messageCursor;wireMessageActions()}catch(e){toast(e.message)}}
function openMessageReply(button){if(loggingOut)return;if(!identity?.user)return openAuth('login');const form=button.closest('.message-card').querySelector('[data-reply-form]');if(!form)return;form.hidden=false;form.querySelector('.reply-error').textContent='';button.setAttribute('aria-expanded','true');form.elements.namedItem('content').focus()}
function closeMessageReply(form){form.hidden=true;form.reset();form.querySelector('.reply-error').textContent='';const button=form.closest('.message-card').querySelector('[data-message-reply]');if(button){button.setAttribute('aria-expanded','false');button.focus()}}
async function sendMessageReply(event){event.preventDefault();if(loggingOut)return;if(!identity?.user)return openAuth('login');const form=event.currentTarget||event.target,button=form.querySelector('button[type="submit"]'),cancel=form.querySelector('[data-cancel-reply]'),error=form.querySelector('.reply-error'),userId=identity.user.id,epoch=privateEpoch;button.disabled=true;cancel.disabled=true;error.textContent='';try{await api('/api/messages/'+pathPart(form.dataset.replyForm)+'/replies',{method:'POST',body:JSON.stringify({content:form.elements.namedItem('content').value})});if(loggingOut||identity?.user?.id!==userId||privateEpoch!==epoch)return;form.reset();await loadMessages();toast('回复已发布')}catch(e){if(!loggingOut&&identity?.user?.id===userId&&privateEpoch===epoch)error.textContent=e.message}finally{button.disabled=false;cancel.disabled=false}}
async function loadOlderReplies(button){if(loggingOut)return;const userId=identity?.user?.id,epoch=privateEpoch;button.disabled=true;try{const data=await api('/api/messages/'+pathPart(button.dataset.olderReplies)+'/replies?before='+pathPart(button.dataset.before));if(loggingOut||identity?.user?.id!==userId||privateEpoch!==epoch)return;button.closest('.message-replies').querySelector('.message-reply-list').insertAdjacentHTML('afterbegin',messageRepliesMarkup(data.replies));button.dataset.before=data.next_before||'';button.hidden=!data.next_before;wireMessageActions()}catch(error){if(!loggingOut&&identity?.user?.id===userId&&privateEpoch===epoch)toast(error.message)}finally{button.disabled=false}}
async function deleteMessage(id,button,permissions={}){if(!canDeleteMessage(permissions))return;if(!window.confirm('确定删除这条留言吗？删除后将不再公开显示。'))return;button.disabled=true;try{await api('/api/messages/'+pathPart(id)+'/delete',{method:'POST',body:'{}'});await loadMessages();toast('留言已删除')}catch(e){toast(e.message)}finally{button.disabled=false}}
$('#refresh-messages').onclick=()=>loadMessages();
$('#more-messages').onclick=async()=>{const b=$('#more-messages');b.disabled=true;try{await loadMessages(true)}finally{b.disabled=false}};
$('#message-form').onsubmit=async e=>{e.preventDefault();if(!identity?.user)return openAuth('login');const b=$('#message-submit');b.disabled=true;$('#message-error').textContent='';try{await api('/api/messages',{method:'POST',body:JSON.stringify({content:$('#message-content').value})});e.target.reset();await loadMessages();toast('留言已发布')}catch(error){$('#message-error').textContent=error.message}finally{b.disabled=false}};

$('#rename-button').onclick=openRename;
$('#rename-form').onsubmit=renameAccount;
$('#admin-strategies-button').onclick=()=>showTab('admin');
$('#refresh-admin-strategies').onclick=()=>loadAdminStrategies();
$('#more-admin-strategies').onclick=async()=>{const b=$('#more-admin-strategies');b.disabled=true;try{await loadAdminStrategies(true)}finally{b.disabled=false}};
$('#replay-sync-form').onsubmit=e=>{e.preventDefault();updateReplaySync()};
$('#replay-sync-run').onclick=()=>updateReplaySync(true);
$('#replay-sync-refresh').onclick=()=>loadReplaySyncConfig();
$('#submit-mode-file').onchange=()=>setSubmissionMode('file');
$('#submit-mode-paste').onchange=()=>setSubmissionMode('paste');
$('#strategy-code').oninput=updatePasteSize;
$('#source-dialog').onclose=()=>clearSourcePreview();
$('#source-dialog').oncancel=()=>clearSourcePreview();
document.querySelectorAll('[data-close="source-dialog"]').forEach(b=>b.onclick=()=>clearSourcePreview(true));
setSubmissionMode('file');

if(preview){$('#preview-banner').hidden=false;$('#worker-status').textContent='界面预览 · 后端开发中';$('#arena-label').textContent='PREVIEW';rankRows([{id:'demo1',name:'Canopy · 示例策略',author:'示例作者 A',rating:1486,games:40,wins:31,margin:28640},{id:'demo2',name:'GreenCollector · 示例策略',author:'示例作者 B',rating:1392,games:36,wins:25,margin:16128},{id:'demo3',name:'CarbonFlow · 示例策略',author:'示例作者 C',rating:1308,games:28,wins:17,margin:5376}]);document.querySelectorAll('a[href="/example.py"]').forEach(a=>a.onclick=e=>{e.preventDefault();toast('正式版将提供可运行的示例策略。')})}else{loadIdentity().then(()=>Promise.allSettled([loadRank(),loadStatus()])).catch(e=>toast(e.message)).finally(scheduleMainPoll);document.addEventListener('visibilitychange',()=>{if(document.hidden){clearTimeout(mainPollTimer);mainPollTimer=null}else pollMain().finally(scheduleMainPoll)})}

// Shared community deep link from every game.
function sharedSection(){if(location.hash==='#messages')showTab('messages')}
window.addEventListener('hashchange',sharedSection);sharedSection();
