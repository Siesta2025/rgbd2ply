"""Browser UI for reviewing and correcting one split segment (sam3 env).

Launch after auto masks exist:
  USE_PERFLIB=0 ~/miniconda3/envs/sam3/bin/python review_ui.py \
    /home/descfly/chiwan/rgbd2ply/runs/20260701_210506_f001011_f001117 --port 8899

The UI can switch between cam1/cam3. Add box/points on any frame; Run
Correction tracks those seeds forward+backward for the selected camera, then
writes:
  cam{N}_corrections.json
  cam{N}_labels_final.npz
"""
import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

# Use config system when available; fall back to env vars for standalone use.
try:
    from config import cfg
    REGISTRY = Path(cfg.paths.registry)
    WILOR_PY = Path(cfg.envs.wilor)
    RGBD2PLY = Path(__file__).resolve().parent
except ImportError:
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent
    REGISTRY = Path(os.environ.get("RGBD2PLY_REGISTRY", str(_HERE / "object_registry.json")))
    RGBD2PLY = Path(os.environ.get("RGBD2PLY_DIR", str(_HERE)))
    WILOR_PY = Path(os.environ.get("WILOR_PY", str(_ROOT / "venv" / "bin" / "python")))

try:
    from .camera_utils import sam3_masking_click as smc
except ImportError:
    from camera_utils import sam3_masking_click as smc
from labelspec import PAINT_ORDER, bgr, label_name  # noqa: E402


class State:
    segment_dir = None
    cam = 3
    frames_dir = None
    auto_npz = None
    final_npz = None
    corrections_json = None
    meta = None
    registry = None
    auto_labels = None
    final_labels = None
    width = 0
    height = 0
    predictor = None
    ckpt = smc.DEFAULT_CKPT
    loaded_cam = None
    lock = threading.RLock()


STATE = State()


def load_registry(path=REGISTRY):
    reg = json.load(open(path))
    return [x for x in reg['labels'] if int(x['id']) > 0]


def available_cams(segment_dir):
    cams = []
    for cam in (1, 3):
        frames = Path(segment_dir) / f'cam{cam}_frames'
        labels = Path(segment_dir) / f'cam{cam}_labels_auto.npz'
        if (frames / 'meta.json').exists() and labels.exists():
            cams.append(cam)
    return cams or [1, 3]


def load_camera(cam, force=False):
    cam = int(cam)
    if cam not in (1, 3):
        raise ValueError('cam must be 1 or 3')
    if STATE.segment_dir is None:
        raise ValueError('segment_dir is not initialized')
    if (not force and STATE.loaded_cam == cam and STATE.auto_labels is not None
            and STATE.final_labels is not None):
        return

    frames_dir = STATE.segment_dir / f'cam{cam}_frames'
    auto_npz = STATE.segment_dir / f'cam{cam}_labels_auto.npz'
    if not (frames_dir / 'meta.json').exists():
        raise FileNotFoundError(str(frames_dir / 'meta.json'))
    if not auto_npz.exists():
        raise FileNotFoundError(str(auto_npz))

    STATE.cam = cam
    STATE.loaded_cam = cam
    STATE.frames_dir = frames_dir
    STATE.auto_npz = auto_npz
    STATE.final_npz = STATE.segment_dir / f'cam{cam}_labels_final.npz'
    STATE.corrections_json = STATE.segment_dir / f'cam{cam}_corrections.json'
    STATE.meta = json.load(open(STATE.frames_dir / 'meta.json'))
    STATE.auto_labels = np.load(STATE.auto_npz)['labels']
    if STATE.final_npz.exists():
        STATE.final_labels = np.load(STATE.final_npz)['labels']
    else:
        STATE.final_labels = STATE.auto_labels.copy()
        np.savez_compressed(STATE.final_npz, labels=STATE.final_labels,
                            frame_indices=np.array(STATE.meta['frame_indices'], np.int64),
                            timestamps=np.array(STATE.meta['timestamps'], np.int64))
    STATE.width = int(STATE.meta['width'])
    STATE.height = int(STATE.meta['height'])


def read_frame(i):
    p = STATE.frames_dir / ('%05d.jpg' % int(i))
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(str(p))
    return im


def overlay_image(i, layer='final'):
    im = read_frame(i)
    labels = STATE.final_labels if layer == 'final' else STATE.auto_labels
    if labels is None:
        labels = STATE.auto_labels
    lab = labels[int(i)]
    for lv in PAINT_ORDER:
        m = lab == lv
        if m.any():
            im[m] = (0.45 * np.array(bgr(lv)) + 0.55 * im[m]).astype(np.uint8)
    return cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()


def _add_obj_at(predictor, state, obj_id, obj, W, H):
    kwargs = {}
    if obj.get('box') is not None:
        x1, y1, x2, y2 = obj['box']
        kwargs['box'] = np.array([[x1 / W, y1 / H, x2 / W, y2 / H]], dtype=np.float32)
    pts = obj.get('points') or []
    if pts:
        kwargs['points'] = torch.tensor([[p[0] / W, p[1] / H] for p in pts], dtype=torch.float32)
        kwargs['labels'] = torch.tensor([int(p[2]) for p in pts], dtype=torch.int32)
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=int(obj.get('frame_idx', 0)), obj_id=obj_id, **kwargs)


def _mask_np(m):
    m = m.cpu().numpy() if hasattr(m, 'cpu') else np.asarray(m)
    return np.squeeze(m).astype(bool)


def normalize_seed(seed):
    frame = int(seed.get('frame_idx', 0))
    start = int(seed.get('start_frame', 0))
    end = int(seed.get('end_frame', STATE.meta['n_frames'] - 1))
    start = max(0, min(start, STATE.meta['n_frames'] - 1))
    end = max(0, min(end, STATE.meta['n_frames'] - 1))
    if start > end:
        start, end = end, start
    obj = {
        'label': int(seed['label']),
        'frame_idx': frame,
        'start_frame': start,
        'end_frame': end,
        'mode': seed.get('mode', 'replace'),
    }
    if seed.get('box'):
        obj['box'] = [float(x) for x in seed['box']]
    if seed.get('points'):
        obj['points'] = [[float(p[0]), float(p[1]), int(p[2])] for p in seed['points']]
    if 'box' not in obj and 'points' not in obj:
        raise ValueError('seed needs box or points')
    if obj.get('mode') == 'replace' and 'box' not in obj:
        pts = obj.get('points') or []
        if pts and not any(int(p[2]) > 0 for p in pts):
            obj['mode'] = 'erase'
    return obj


def is_trackable_seed(seed):
    if seed.get('mode') == 'erase':
        return False
    if seed.get('box') is not None:
        return True
    return any(int(p[2]) > 0 for p in (seed.get('points') or []))


def track_seeds(seeds):
    seeds = [normalize_seed(s) for s in seeds]
    trackable = [s for s in seeds if is_trackable_seed(s)]
    if not trackable:
        return np.zeros_like(STATE.auto_labels), seeds
    T, H, W = STATE.auto_labels.shape
    if torch.cuda.is_available():
        torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
    if STATE.predictor is None:
        STATE.predictor = smc._build(STATE.ckpt)
    predictor = STATE.predictor
    state = predictor.init_state(video_path=str(STATE.frames_dir))
    predictor.clear_all_points_in_video(state)

    obj_meta = {}
    for oid, obj in enumerate(trackable, start=1):
        _add_obj_at(predictor, state, oid, obj, W, H)
        obj_meta[oid] = obj

    labels = np.zeros((T, H, W), np.int32)
    min_seed = min(int(s['frame_idx']) for s in trackable)
    max_seed = max(int(s['frame_idx']) for s in trackable)

    def paint_stream(reverse, start_frame, max_frames):
        for fidx, obj_ids, low, video_res_masks, scores in predictor.propagate_in_video(
                state, start_frame_idx=start_frame, max_frame_num_to_track=max_frames,
                reverse=reverse, propagate_preflight=True):
            for target in PAINT_ORDER:
                for mi, oid in enumerate(obj_ids):
                    oid = int(oid)
                    meta = obj_meta.get(oid)
                    if not meta or int(meta['label']) != int(target):
                        continue
                    if not (meta['start_frame'] <= int(fidx) <= meta['end_frame']):
                        continue
                    m = _mask_np(video_res_masks[mi] > 0.0)
                    labels[int(fidx)][m] = int(target)

    paint_stream(False, min_seed, T - min_seed)
    paint_stream(True, max_seed, max_seed + 1)
    return labels, seeds


def erase_connected_component(final, frame_idx, label, x, y, radius=16):
    frame_idx = int(frame_idx)
    label = int(label)
    x, y = int(round(x)), int(round(y))
    if not (0 <= frame_idx < final.shape[0]):
        return
    H, W = final.shape[1:]
    if not (0 <= x < W and 0 <= y < H):
        return
    mask = (final[frame_idx] == label).astype(np.uint8)
    if not mask.any():
        return
    if mask[y, x]:
        _, cc = cv2.connectedComponents(mask, connectivity=8)
        cid = int(cc[y, x])
        if cid > 0:
            final[frame_idx][cc == cid] = 0
        return
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= int(radius) ** 2
    final[frame_idx][disk & (final[frame_idx] == label)] = 0


def apply_erase_seed(final, seed):
    if seed.get('mode') != 'erase':
        return
    lab = int(seed['label'])
    start = int(seed['start_frame'])
    end = int(seed['end_frame'])
    box = seed.get('box')
    if box:
        H, W = final.shape[1:]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if x2 > x1 and y2 > y1:
            for f in range(start, end + 1):
                roi = final[f, y1:y2, x1:x2]
                roi[roi == lab] = 0
    for f in range(start, end + 1):
        for p in seed.get('points') or []:
            erase_connected_component(final, f, lab, p[0], p[1])


def compose_final(correction_labels, seeds):
    final = STATE.auto_labels.copy()
    for seed in seeds:
        if seed.get('mode', 'replace') != 'replace':
            continue
        lab = int(seed['label'])
        for f in range(int(seed['start_frame']), int(seed['end_frame']) + 1):
            final[f][final[f] == lab] = 0
    m = correction_labels > 0
    final[m] = correction_labels[m]
    for seed in seeds:
        apply_erase_seed(final, seed)
    return final


def save_final(final, corrections):
    np.savez_compressed(STATE.final_npz, labels=final,
                        frame_indices=np.array(STATE.meta['frame_indices'], np.int64),
                        timestamps=np.array(STATE.meta['timestamps'], np.int64))
    STATE.corrections_json.write_text(json.dumps(corrections, ensure_ascii=False, indent=2) + '\n')


def counts(labels):
    out = {}
    for lab in np.unique(labels):
        lab = int(lab)
        if lab:
            out[label_name(lab)] = int((labels == lab).sum())
    return out


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>rgbd2ply review</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font-family: system-ui, sans-serif; background: #171717; color: #ececec; }
  #top { display: flex; align-items: center; gap: 8px; padding: 10px 12px; border-bottom: 1px solid #303030; position: sticky; top:0; background:#171717; z-index:3; }
  button, select, input { background: #242424; color: #eee; border: 1px solid #444; border-radius: 5px; padding: 5px 8px; }
  button.active { outline: 2px solid #f0c14b; }
  #main { display: grid; grid-template-columns: minmax(640px, 1fr) 310px; gap: 12px; padding: 12px; }
  #stage { position: relative; display: inline-block; max-width: 100%; min-width: 640px; min-height: 360px; }
  #loading { padding: 24px; color: #bbb; font-family: ui-monospace, monospace; }
  #img, #cv { position: absolute; left: 0; top: 0; }
  #img { position: relative; display: block; max-width: 100%; height: auto; }
  #cv { cursor: crosshair; }
  .panel { background:#202020; border:1px solid #333; border-radius: 7px; padding: 10px; }
  .row { display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-bottom:8px; }
  #labels button { min-width: 84px; }
  #seeds { max-height: 280px; overflow:auto; font-size: 12px; }
  .seed { border-bottom:1px solid #333; padding:5px 0; cursor:pointer; }
  .seed.active { color:#f0c14b; }
  #status { white-space: pre-wrap; font-family: ui-monospace, monospace; font-size:12px; color:#ccc; }
  input[type=range] { width: 240px; }
</style></head>
<body>
<div id="top">
  <strong id="title">review</strong>
  <select id="cam"></select>
  <button id="prev">&lt;</button>
  <input id="frame" type="range" min="0" value="0">
  <button id="next">&gt;</button>
  <span id="ftext"></span>
  <select id="layer"><option value="final">final</option><option value="auto">auto</option></select>
  <button id="run">Run Correction</button>
  <button id="save">Save Seeds</button>
  <button id="fuse">Fuse Pointclouds</button>
</div>
<div id="main">
  <div id="stage"><div id="loading">loading...</div><img id="img"><canvas id="cv"></canvas></div>
  <div class="panel">
    <div class="row" id="labels"></div>
    <div class="row">
      <button class="tool active" data-tool="box">box</button>
      <button class="tool" data-tool="boxneg">box-</button>
      <button class="tool" data-tool="pos">point+</button>
      <button class="tool" data-tool="neg">point-</button>
    </div>
    <div class="row">
      <label>start <input id="start" type="number" min="0" style="width:70px"></label>
      <label>end <input id="end" type="number" min="0" style="width:70px"></label>
    </div>
    <div class="row">
      <button id="del">Delete Seed</button>
      <button id="clear">Clear Local</button>
    </div>
    <div id="seeds"></div>
    <pre id="status"></pre>
  </div>
</div>
<script>
let st=null, frame=0, scale=1, seeds=[], active=-1, label=1, tool='box', drag=null;
const img=document.getElementById('img'), cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const frameEl=document.getElementById('frame'), ftext=document.getElementById('ftext');
const statusEl=document.getElementById('status'), seedsEl=document.getElementById('seeds');
function q(id){return document.getElementById(id)}
function color(id){ const c=['#999','#dc2828','#28c828','#3cb4e6','#c83cc8','#f08228','#1446be','#c828a0']; return c[id]||'#ddd'; }
async function init(cam=null){
  q('loading').style.display='block';
  const url=cam?('/state?cam='+cam):'/state';
  try {
    st=await (await fetch(url)).json();
  } catch(e) {
    statusEl.textContent='ERROR loading state: '+e;
    return;
  }
  if(st.error){ statusEl.textContent='ERROR '+st.error; return; }
  seeds=st.corrections.seeds||[]; active=-1; label=1; frame=0; frameEl.value=0;
  q('title').textContent=st.segment;
  const camSel=q('cam'); camSel.innerHTML='';
  (st.cams||[1,3]).forEach(c=>{ const o=document.createElement('option'); o.value=c; o.textContent='cam'+c; camSel.appendChild(o); });
  camSel.value=String(st.cam);
  frameEl.max=st.n_frames-1; q('start').value=0; q('end').value=st.n_frames-1;
  const labels=q('labels'); labels.innerHTML='';
  st.labels.forEach(l=>{ const b=document.createElement('button'); b.textContent=l.id+' '+l.key; b.style.color=color(l.id); b.onclick=()=>{label=l.id; [...labels.children].forEach(x=>x.classList.remove('active')); b.classList.add('active')}; labels.appendChild(b); if(l.id===1)b.classList.add('active'); });
  loadFrame(); renderSeeds();
}
function loadFrame(){ frame=parseInt(frameEl.value); ftext.textContent='cam'+st.cam+'  '+frame+'/'+(st.n_frames-1)+' src '+st.frame_indices[frame]; q('loading').style.display='block'; img.src='/frame.jpg?cam='+st.cam+'&i='+frame+'&layer='+q('layer').value+'&t='+Date.now(); }
img.onload=()=>{ q('loading').style.display='none'; scale=img.clientWidth/st.width; cv.width=img.clientWidth; cv.height=img.clientHeight; draw(); };
img.onerror=()=>{ q('loading').textContent='image load failed'; };
function xy(e){ const r=cv.getBoundingClientRect(); return [(e.clientX-r.left)/scale,(e.clientY-r.top)/scale]; }
function hasPositiveSeed(s){ return !!s.box || (s.points||[]).some(p=>p[2]===1); }
function rangeSeed(){ return {start_frame:parseInt(q('start').value),end_frame:parseInt(q('end').value)}; }
function findSeed(pred){ for(let i=seeds.length-1;i>=0;i--){ if(pred(seeds[i])) return i; } return -1; }
function addPointSeed(p){
  if(tool==='neg'){
    let target = -1;
    if(active>=0 && seeds[active].label===label && seeds[active].frame_idx===frame) target=active;
    if(target<0) target=findSeed(s=>s.label===label && s.frame_idx===frame && s.mode!=='erase' && hasPositiveSeed(s));
    if(target>=0 && seeds[target].mode!=='erase' && hasPositiveSeed(seeds[target])){
      seeds[target].points=(seeds[target].points||[]); seeds[target].points.push(p); active=target; return;
    }
    target = (active>=0 && seeds[active].label===label && seeds[active].frame_idx===frame && seeds[active].mode==='erase') ? active : findSeed(s=>s.label===label && s.frame_idx===frame && s.mode==='erase');
    if(target>=0){ seeds[target].points=(seeds[target].points||[]); seeds[target].points.push(p); active=target; return; }
    seeds.push({label,frame_idx:frame,...rangeSeed(),mode:'erase',points:[p]}); active=seeds.length-1; return;
  }
  if(active>=0 && seeds[active].label===label && seeds[active].frame_idx===frame && seeds[active].mode!=='erase'){
    seeds[active].points=(seeds[active].points||[]); seeds[active].points.push(p); return;
  }
  seeds.push({label,frame_idx:frame,...rangeSeed(),mode:'replace',points:[p]}); active=seeds.length-1;
}
function draw(){ ctx.clearRect(0,0,cv.width,cv.height); seeds.forEach((s,i)=>{ if(s.frame_idx!==frame) return; ctx.strokeStyle=s.mode==='erase'?'#ff4eb8':color(s.label); ctx.lineWidth=i===active?3:2; if(s.box){ const b=s.box; ctx.strokeRect(b[0]*scale,b[1]*scale,(b[2]-b[0])*scale,(b[3]-b[1])*scale); if(s.mode==='erase'){ ctx.beginPath(); ctx.moveTo(b[0]*scale,b[1]*scale); ctx.lineTo(b[2]*scale,b[3]*scale); ctx.moveTo(b[2]*scale,b[1]*scale); ctx.lineTo(b[0]*scale,b[3]*scale); ctx.stroke(); } } (s.points||[]).forEach(p=>{ ctx.beginPath(); ctx.arc(p[0]*scale,p[1]*scale,5,0,7); ctx.fillStyle=p[2]?'#56e36b':'#ff4eb8'; ctx.fill(); if(!p[2]){ ctx.beginPath(); ctx.moveTo((p[0]-7)*scale,p[1]*scale); ctx.lineTo((p[0]+7)*scale,p[1]*scale); ctx.moveTo(p[0]*scale,(p[1]-7)*scale); ctx.lineTo(p[0]*scale,(p[1]+7)*scale); ctx.strokeStyle='#ff4eb8'; ctx.lineWidth=2; ctx.stroke(); } }); }); if(drag){ ctx.strokeStyle=drag.erase?'#ff4eb8':color(label); ctx.lineWidth=2; ctx.strokeRect(drag.x0*scale,drag.y0*scale,(drag.x1-drag.x0)*scale,(drag.y1-drag.y0)*scale); if(drag.erase){ ctx.beginPath(); ctx.moveTo(drag.x0*scale,drag.y0*scale); ctx.lineTo(drag.x1*scale,drag.y1*scale); ctx.moveTo(drag.x1*scale,drag.y0*scale); ctx.lineTo(drag.x0*scale,drag.y1*scale); ctx.stroke(); } } }
cv.onmousedown=e=>{ const [x,y]=xy(e); if(tool==='box'||tool==='boxneg'){ drag={x0:x,y0:y,x1:x,y1:y,erase:tool==='boxneg'}; return; } const p=[Math.round(x),Math.round(y),tool==='neg'?0:1]; addPointSeed(p); renderSeeds(); draw(); };
cv.onmousemove=e=>{ if(!drag)return; const [x,y]=xy(e); drag.x1=x; drag.y1=y; draw(); };
cv.onmouseup=e=>{ if(!drag)return; const b=[Math.round(Math.min(drag.x0,drag.x1)),Math.round(Math.min(drag.y0,drag.y1)),Math.round(Math.max(drag.x0,drag.x1)),Math.round(Math.max(drag.y0,drag.y1))]; if(Math.abs(b[2]-b[0])>4&&Math.abs(b[3]-b[1])>4){ seeds.push({label,frame_idx:frame,...rangeSeed(),mode:drag.erase?'erase':'replace',box:b,points:[]}); active=seeds.length-1; } drag=null; renderSeeds(); draw(); };
function renderSeeds(){ seedsEl.innerHTML=''; seeds.forEach((s,i)=>{ const d=document.createElement('div'); d.className='seed'+(i===active?' active':''); d.textContent='#'+i+' '+(s.mode==='erase'?'erase ':'')+'L'+s.label+' f'+s.frame_idx+' '+(s.box?'box':'pts')+' range '+s.start_frame+'-'+s.end_frame; d.onclick=()=>{active=i; frameEl.value=s.frame_idx; loadFrame(); renderSeeds();}; seedsEl.appendChild(d); }); statusEl.textContent='seeds: '+seeds.length; }
document.querySelectorAll('.tool').forEach(b=>b.onclick=()=>{ tool=b.dataset.tool; document.querySelectorAll('.tool').forEach(x=>x.classList.remove('active')); b.classList.add('active'); });
frameEl.oninput=loadFrame; q('layer').onchange=loadFrame; q('cam').onchange=()=>init(q('cam').value); q('prev').onclick=()=>{frameEl.value=Math.max(0,parseInt(frameEl.value)-1); loadFrame();}; q('next').onclick=()=>{frameEl.value=Math.min(st.n_frames-1,parseInt(frameEl.value)+1); loadFrame();};
q('del').onclick=()=>{ if(active>=0){ seeds.splice(active,1); active=-1; renderSeeds(); draw(); }};
q('clear').onclick=()=>{ seeds=[]; active=-1; renderSeeds(); draw(); };
q('save').onclick=async()=>{ await fetch('/save_corrections',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:st.cam,seeds})}); statusEl.textContent='saved cam'+st.cam+' corrections json'; };
q('run').onclick=async()=>{ statusEl.textContent='running SAM3 correction...'; const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:st.cam,seeds})}); const j=await r.json(); if(j.error){statusEl.textContent='ERROR '+j.error;return;} statusEl.textContent='saved final\n'+JSON.stringify(j.counts,null,2); q('layer').value='final'; loadFrame(); };
q('fuse').onclick=async()=>{ statusEl.textContent='running pointcloud fusion...'; const r=await fetch('/fuse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})}); const j=await r.json(); if(j.error){statusEl.textContent='ERROR '+j.error+'\n'+(j.output||'');return;} statusEl.textContent='fused pointclouds\nreal_color: '+j.real_color+'\nmask_only: '+j.mask_only+'\n\n'+(j.output||''); };
init();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def send_body(self, code, body, ctype='text/html; charset=utf-8'):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            u = urlparse(self.path)
            if u.path in ('/', '/index.html'):
                return self.send_body(200, PAGE)
            if u.path == '/state':
                qs = parse_qs(u.query)
                cam = int(qs.get('cam', [str(STATE.cam)])[0])
                with STATE.lock:
                    load_camera(cam)
                    corr = {'seeds': []}
                    if STATE.corrections_json.exists():
                        corr = json.load(open(STATE.corrections_json))
                    body = {
                        'segment': STATE.segment_dir.name,
                        'cam': STATE.cam,
                        'cams': available_cams(STATE.segment_dir),
                        'n_frames': STATE.meta['n_frames'],
                        'width': STATE.width,
                        'height': STATE.height,
                        'frame_indices': STATE.meta['frame_indices'],
                        'labels': STATE.registry,
                        'corrections': corr,
                    }
                return self.send_body(200, json.dumps(body), 'application/json')
            if u.path == '/frame.jpg':
                qs = parse_qs(u.query)
                cam = int(qs.get('cam', [str(STATE.cam)])[0])
                i = int(qs.get('i', ['0'])[0])
                layer = qs.get('layer', ['final'])[0]
                with STATE.lock:
                    load_camera(cam)
                    body = overlay_image(i, layer)
                return self.send_body(200, body, 'image/jpeg')
            return self.send_body(404, 'not found')
        except Exception as e:
            import traceback; traceback.print_exc()
            return self.send_body(500, str(e))

    def do_POST(self):
        try:
            n = int(self.headers.get('Content-Length', '0'))
            req = json.loads(self.rfile.read(n) or b'{}')
            if self.path == '/save_corrections':
                with STATE.lock:
                    load_camera(int(req.get('cam', STATE.cam)))
                    STATE.corrections_json.write_text(json.dumps({'seeds': req.get('seeds', [])}, ensure_ascii=False, indent=2) + '\n')
                return self.send_body(200, json.dumps({'ok': True, 'cam': STATE.cam}), 'application/json')
            if self.path == '/run':
                with STATE.lock:
                    load_camera(int(req.get('cam', STATE.cam)))
                    correction, normalized = track_seeds(req.get('seeds', []))
                    final = compose_final(correction, normalized)
                    save_final(final, {'seeds': req.get('seeds', [])})
                    STATE.final_labels = final
                    cam = STATE.cam
                return self.send_body(200, json.dumps({'ok': True, 'cam': cam, 'counts': counts(final)}), 'application/json')
            if self.path == '/fuse':
                # Read recording_dir from segment.json (saved by prepare step)
                seg_json = STATE.segment_dir / 'segment.json'
                if seg_json.exists():
                    seg = json.load(open(seg_json))
                    recording_dir = seg.get('recording_dir', str(STATE.segment_dir))
                else:
                    recording_dir = str(STATE.segment_dir)

                # Call fusion.py directly for both colour modes
                real_dir = STATE.segment_dir / 'pointclouds' / 'masked_rgb'
                mask_dir = STATE.segment_dir / 'pointclouds' / 'mask_only'
                out_lines = []
                ok = True
                for out_dir, color_mode, labeled_only in [
                    (real_dir, 'masked_rgb', False),
                    (mask_dir, 'label', True),
                ]:
                    cmd = [str(WILOR_PY), str(RGBD2PLY / 'fusion.py'),
                           str(STATE.segment_dir), recording_dir,
                           '--out', str(out_dir),
                           '--color-mode', color_mode]
                    if labeled_only:
                        cmd.append('--labeled-only')
                    proc = subprocess.run(cmd, cwd=str(RGBD2PLY), text=True,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    out_lines.append(proc.stdout or '')
                    if proc.returncode != 0:
                        ok = False
                out = '\n'.join(out_lines)
                body = {
                    'ok': ok,
                    'real_color': str(real_dir),
                    'mask_only': str(mask_dir),
                    'output': out[-8000:],
                }
                if not ok:
                    body['error'] = 'fuse failed'
                return self.send_body(200, json.dumps(body), 'application/json')
            return self.send_body(404, 'not found')
        except Exception as e:
            import traceback; traceback.print_exc()
            return self.send_body(200, json.dumps({'error': str(e)}), 'application/json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('segment_dir')
    ap.add_argument('--cam', type=int, choices=[1, 3], default=1)
    ap.add_argument('--port', type=int, default=8899)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--ckpt', default=smc.DEFAULT_CKPT)
    args = ap.parse_args()

    STATE.segment_dir = Path(args.segment_dir)
    STATE.ckpt = args.ckpt
    STATE.registry = load_registry()
    load_camera(args.cam)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print('serving %s at http://%s:%d/' % (STATE.segment_dir, args.host, args.port))
    print('cams ->', ', '.join('cam%d' % c for c in available_cams(STATE.segment_dir)))
    print('auto ->', STATE.auto_npz)
    print('final ->', STATE.final_npz)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == '__main__':
    main()
