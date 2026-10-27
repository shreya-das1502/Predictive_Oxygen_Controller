/* Offline PPG-only annotation interface; no external requests. */
const $ = id => document.getElementById(id);
let state = { reviewer_id: '', windows: {} }, current = 0;
try { state = JSON.parse(localStorage.getItem(PACKET_ID)) || state; } catch (_) {}
function entry() {
  const w = WINDOWS[current];
  if (!state.windows[w.id]) state.windows[w.id] = { window_id: w.id, quality: 'not_reviewed', scored_regions: [[.08,19.92]], pulses: [], notes: '', reviewed_no_reference_pulses: false, detector_overlay_ever_shown: false };
  return state.windows[w.id];
}
function save() {
  const e = entry(); e.quality = $('quality').value; e.notes = $('notes').value;
  e.reviewed_no_reference_pulses = $('empty_reference').checked;
  try { e.scored_regions = $('regions').value.trim() ? $('regions').value.split(';').map(r => r.split(':').map(Number)) : []; } catch (_) {}
  state.reviewer_id = $('reviewer').value.trim();
  try { localStorage.setItem(PACKET_ID, JSON.stringify(state)); } catch (_) { $('status').textContent = 'Local storage unavailable; export frequently.'; }
}
function load() {
  const w = WINDOWS[current], e = entry(); $('window').value = String(current);
  $('quality').value = e.quality; $('notes').value = e.notes; $('empty_reference').checked = e.reviewed_no_reference_pulses;
  $('regions').value = e.scored_regions.map(r => r.join(':')).join(';'); $('overlay').checked = false;
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
WINDOWS.forEach((w,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${i+1}. ${w.id} · ${w.record}${w.reference_subset?' *':''}`;$('window').appendChild(option);});
$('reviewer').value=state.reviewer_id||'';
$('window').onchange=()=>{save();current=Number($('window').value);load();};
$('previous').onclick=()=>{save();current=Math.max(0,current-1);load();};
$('next').onclick=()=>{save();current=Math.min(WINDOWS.length-1,current+1);load();};
['start','span'].forEach(id=>$(id).oninput=draw);
['quality','notes','regions','empty_reference','reviewer'].forEach(id=>$(id).onchange=()=>{save();load();});
$('overlay').onchange=()=>{if($('overlay').checked){entry().detector_overlay_ever_shown=true;save();}draw();};
function clickTime(event) {
  const c=$('filtered'),r=c.getBoundingClientRect(),v=c.view;const px=(event.clientX-r.left)*c.width/r.width;
  return Math.round(Math.max(0,Math.min(20,v.start+(px-v.left)/(v.right-v.left)*(v.end-v.start)))*125)/125;
}
$('filtered').onclick=event=>{const t=clickTime(event),e=entry();if(!e.pulses.some(p=>p.time_s===t))e.pulses.push({time_s:t,certainty:$('certainty').value});e.pulses.sort((a,b)=>a.time_s-b.time_s);save();draw();$('status').textContent=`${e.pulses.length} reference marks`;};
$('filtered').oncontextmenu=event=>{event.preventDefault();const t=clickTime(event),e=entry();let k=-1,best=.16;e.pulses.forEach((p,i)=>{const gap=Math.abs(p.time_s-t);if(gap<best){best=gap;k=i;}});if(k>=0)e.pulses.splice(k,1);save();draw();};
$('undo').onclick=()=>{entry().pulses.pop();save();draw();};
function exportFile(finalized) {
  save();const windows=Object.values(state.windows).filter(w=>w.quality!=='not_reviewed');
  if(finalized) {
    if(!state.reviewer_id||!windows.length){$('status').textContent='A reviewer code and actual reviewed windows are required.';return;}
    for(const w of windows) {
      if(w.quality==='uninterpretable'&&w.scored_regions.length){$('status').textContent=`${w.window_id}: remove scoring regions for uninterpretable waveform.`;return;}
      if(w.quality!=='uninterpretable'&&!w.scored_regions.length){$('status').textContent=`${w.window_id}: supply exhaustive scoring regions.`;return;}
      let previous=-1;
      for(const r of w.scored_regions){if(r.length!==2||!r.every(Number.isFinite)||!(0<=r[0]&&r[0]<r[1]&&r[1]<=20)||r[0]<previous){$('status').textContent=`${w.window_id}: invalid/overlapping scoring regions.`;return;}previous=r[1];}
      if(w.scored_regions.length&&!w.pulses.some(p=>p.certainty==='clear')&&!w.reviewed_no_reference_pulses){$('status').textContent=`${w.window_id}: blank reference is not ground truth; annotate or explicitly confirm no reference pulses.`;return;}
    }
  }
  const payload={packet_id:PACKET_ID,reviewer_id:state.reviewer_id,finalized,windows};
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`PPG_${PACKET_ID}_${state.reviewer_id||'draft'}_${finalized?'final':'draft'}.json`;a.click();URL.revokeObjectURL(a.href);
}
$('draft').onclick=()=>exportFile(false);$('final').onclick=()=>exportFile(true);load();
