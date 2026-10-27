/* Offline PPG-only annotation interface; no external requests. */
const $ = id => document.getElementById(id);
let state = { reviewer_id: '', windows: {} }, current = 0;
let activeReviewer='';
const cacheKey=()=>`PPGReview:v2:${PACKET_ID}:${activeReviewer}`;
function entry() {
  const w = WINDOWS[current];
  if (!state.windows[w.id]) state.windows[w.id] = { window_id: w.id, quality: 'not_reviewed', scored_regions: [[.08,19.92]], pulses: [], notes: '', reviewed_no_reference_pulses: false, detector_overlay_ever_shown: false };
  return state.windows[w.id];
}
function save() {
  if(!activeReviewer)return;
  const e = entry(); e.quality = $('quality').value; e.notes = $('notes').value;
  e.reviewed_no_reference_pulses = $('empty_reference').checked;
  try { e.scored_regions = $('regions').value.trim() ? $('regions').value.split(';').map(r => r.split(':').map(Number)) : []; } catch (_) {}
  state.reviewer_id = activeReviewer;
  entry().region_text=$('regions').value;
  try { localStorage.setItem(cacheKey(), JSON.stringify(state)); } catch (_) { $('status').textContent = 'Local storage unavailable; export frequently.'; }
}
function load() {
  $('progress').textContent=`${WINDOWS.filter(w=>w.reference_subset && state.windows[w.id]?.quality!=='not_reviewed' && state.windows[w.id]).length} / ${WINDOWS.filter(w=>w.reference_subset).length} priority windows labelled (final export checks validity)`;
  const w = WINDOWS[current], e = entry(); $('window').value = String(current);
  $('quality').value = e.quality; $('notes').value = e.notes; $('empty_reference').checked = e.reviewed_no_reference_pulses;
  $('regions').value = e.region_text??e.scored_regions.map(r => r.join(':')).join(';'); $('overlay').checked = false;
  $('meta').textContent = `${w.id} · ${w.record} · ${w.duration}s · ${w.fs}Hz · ${w.reference_subset ? 'Priority reference subset' : 'Additional review window'}`;
  $('status').textContent = `${e.pulses.length} reference marks; quality: ${e.quality}`; draw();
}
function drawOne(id, values) {
  const canvas = $(id), ctx = canvas.getContext('2d'), w = WINDOWS[current], e = entry();
  const start = Math.max(0,Math.min(18,Number($('start').value)||0));
  const end = Math.min(20,start+Math.max(2,Math.min(20,Number($('span').value)||20)));
  const left=45,right=canvas.width-15,top=15,bottom=canvas.height-30;
  const a=Math.floor(start*w.fs),b=Math.min(values.length-1,Math.ceil(end*w.fs));
  const part=values.slice(a,b+1);let lo=Math.min(...part),hi=Math.max(...part);if(hi===lo){lo-=.01;hi+=.01;}
  const x=t=>left+(t-start)/(end-start)*(right-left),y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#334155';ctx.lineWidth=1;ctx.beginPath();
  for(let i=a;i<=b;i++){const px=x(i/w.fs),py=y(values[i]);if(i===a)ctx.moveTo(px,py);else ctx.lineTo(px,py);}ctx.stroke();
  ctx.fillStyle='#334155';ctx.font='12px sans-serif';ctx.fillText(hi.toFixed(3),2,top+10);ctx.fillText(lo.toFixed(3),2,bottom);
  for(let k=0;k<=4;k++){const t=start+(end-start)*k/4;ctx.fillText(t.toFixed(2),x(t)-10,canvas.height-8);}
  if(id==='filtered') {
    function marker(t,color,size){if(t<start||t>end)return;const i=Math.max(0,Math.min(values.length-1,Math.round(t*w.fs)));ctx.fillStyle=color;ctx.beginPath();ctx.arc(x(t),y(values[i]),size,0,Math.PI*2);ctx.fill();}
    if($('overlay').checked)w.detected.forEach(t=>marker(t,'#be123c',3));
    e.pulses.forEach(p=>marker(p.time_s,p.certainty==='clear'?'#047857':'#c2410c',5));
  }
  canvas.view={start,end,left,right};
}
function draw(){drawOne('raw',WINDOWS[current].raw);drawOne('filtered',WINDOWS[current].filtered);}
WINDOWS.forEach((w,i)=>{if(!w.reference_subset)return;const option=document.createElement('option');option.value=i;option.textContent=`${i+1}. ${w.id} · ${w.record}${w.reference_subset?' *':''}`;$('window').appendChild(option);});
$('reviewer').value=state.reviewer_id||'';
$('window').onchange=()=>{save();current=Number($('window').value);load();};
$('previous').onclick=()=>{save();current=priority[Math.max(0,priority.indexOf(current)-1)];load();};
$('next').onclick=()=>{save();current=priority[Math.min(priority.length-1,priority.indexOf(current)+1)];load();};
['start','span'].forEach(id=>$(id).oninput=draw);
['quality','notes','regions','empty_reference'].forEach(id=>$(id).oninput=()=>{save();});
$('overlay').onchange=()=>{if($('overlay').checked){entry().detector_overlay_ever_shown=true;save();}draw();};
function clickTime(event) {
  const c=$('filtered'),r=c.getBoundingClientRect(),v=c.view;const px=(event.clientX-r.left)*c.width/r.width,py=(event.clientY-r.top)*c.height/r.height;if(px<v.left||px>v.right||py<15||py>c.height-30)return null;
  return Math.round(Math.max(0,Math.min(20,v.start+(px-v.left)/(v.right-v.left)*(v.end-v.start)))*125)/125;
}
$('filtered').onclick=event=>{const t=clickTime(event),e=entry();if(t===null||!activeReviewer)return;if(!e.pulses.some(p=>p.time_s===t))e.pulses.push({time_s:t,certainty:$('certainty').value});e.pulses.sort((a,b)=>a.time_s-b.time_s);save();draw();$('status').textContent=`${e.pulses.length} reference marks`;};
$('filtered').oncontextmenu=event=>{event.preventDefault();const t=clickTime(event),e=entry();if(t===null||!activeReviewer)return;let k=-1,best=.16;e.pulses.forEach((p,i)=>{const gap=Math.abs(p.time_s-t);if(gap<best){best=gap;k=i;}});if(k>=0)e.pulses.splice(k,1);save();draw();};
$('undo').onclick=()=>{entry().pulses.pop();save();draw();};
function exportFile(finalized) {
  save();if(!activeReviewer){$('status').textContent='Start a reviewer session';return;}
  if(entry().scored_regions.some(r=>r.length!==2||!r.every(Number.isFinite))){$('status').textContent='Correct invalid region text before export';return;}const windows=Object.values(state.windows).filter(w=>w.quality!=='not_reviewed').map(({region_text,...w})=>w);
  if(finalized) {try {validateWindows(windows,true);}catch(e){$('status').textContent=e.message;return;}}
  const payload={packet_id:PACKET_ID,reviewer_id:state.reviewer_id,finalized,windows};
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`PPG_${PACKET_ID}_${state.reviewer_id||'draft'}_${finalized?'final':'draft'}.json`;a.click();URL.revokeObjectURL(a.href);
}
$('draft').onclick=()=>exportFile(false);$('final').onclick=()=>exportFile(true);
const priority=WINDOWS.map((w,i)=>w.reference_subset?i:-1).filter(i=>i>=0);current=priority[0];
$('activate').onclick=()=>{const id=$('reviewer').value.trim();if(!/^[A-Za-z0-9_-]{1,40}$/.test(id)){alert('Use 1–40 letters, digits, underscore or hyphen');return;}save();activeReviewer=id;try{state=JSON.parse(localStorage.getItem(cacheKey()))||{reviewer_id:id,windows:{}};}catch(_){state={reviewer_id:id,windows:{}};}load();};
$('import').onchange=async event=>{try{const p=JSON.parse(await event.target.files[0].text());if(!activeReviewer||p.packet_id!==PACKET_ID||p.reviewer_id!==activeReviewer||typeof p.finalized!=='boolean')throw Error('Start matching reviewer; packet/reviewer must match');validateWindows(p.windows,false,true);state={reviewer_id:activeReviewer,windows:Object.fromEntries(p.windows.map(w=>[w.window_id,w]))};load();save();$('status').textContent='Backup imported; re-export final after completing review';}catch(e){$('status').textContent='Import rejected: '+e.message;}};load();

function validateWindows(rows,complete,allowDraft=false){
 if(!activeReviewer)throw Error('Start a reviewer session first');
 if(!Array.isArray(rows))throw Error('windows must be an array');const seen=new Set();
 for(const w of rows){
  const source=WINDOWS.find(v=>v.id===w.window_id);if(!source||seen.has(w.window_id))throw Error('Unknown/duplicate window identifier');seen.add(w.window_id);
  const fields=['window_id','quality','scored_regions','pulses','notes','reviewed_no_reference_pulses','detector_overlay_ever_shown'];if(Object.keys(w).sort().join()!=fields.sort().join())throw Error('Missing/unknown window fields');
  if(!['adequately_interpretable','partially_interpretable','uninterpretable'].includes(w.quality))throw Error(w.window_id+': invalid quality');
  if(typeof w.notes!=='string'||typeof w.reviewed_no_reference_pulses!=='boolean'||typeof w.detector_overlay_ever_shown!=='boolean'||!Array.isArray(w.pulses)||!Array.isArray(w.scored_regions))throw Error('Invalid field types');
  let prev=-1;for(const r of w.scored_regions){if(!Array.isArray(r)||r.length!==2||!r.every(v=>typeof v==='number'&&Number.isFinite(v))||!(0<=r[0]&&r[0]<r[1]&&r[1]<=source.duration)||r[0]<=prev)throw Error(w.window_id+': invalid regions');prev=r[1];}
  prev=-1;for(const p of w.pulses){if(Object.keys(p).sort().join()!=='certainty,time_s'||typeof p.time_s!=='number'||!Number.isFinite(p.time_s)||p.time_s<0||p.time_s>source.duration||p.time_s<=prev||!['clear','ambiguous'].includes(p.certainty))throw Error(w.window_id+': invalid/duplicate/outside pulse');prev=p.time_s;}
  if((w.quality!=='adequately_interpretable'||w.pulses.some(p=>p.certainty==='ambiguous'))&&!w.notes.trim())throw Error(w.window_id+': notes required');
  const clear=w.pulses.filter(p=>p.certainty==='clear');const usable=clear.filter(p=>w.scored_regions.some(r=>r[0]<=p.time_s&&p.time_s<=r[1])&&!w.pulses.some(a=>a.certainty==='ambiguous'&&Math.abs(a.time_s-p.time_s)<=.080000000001));
  if(w.quality==='uninterpretable'){if(w.scored_regions.length||clear.length||w.reviewed_no_reference_pulses)throw Error(w.window_id+': uninterpretable requires no regions/clear pulses/empty confirmation');}
  else if(!w.scored_regions.length||(!usable.length&&!w.reviewed_no_reference_pulses)||(usable.length&&w.reviewed_no_reference_pulses))throw Error(w.window_id+': annotate scored regions or explicitly confirm no pulses');
 }
 if(complete){const missing=WINDOWS.filter(w=>w.reference_subset&&!seen.has(w.id));if(missing.length)throw Error('Incomplete priority review: '+missing.length+' windows missing');}
}
