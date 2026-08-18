import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const canvas = $("fluid");
const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
const video = $("camera");
const statusEl = $("status");
const intro = $("intro");
const panel = $("panel");
const toast = $("toast");

const ui = {
  fps: $("fpsStat"),
  particles: $("particleStat"),
  gesture: $("gestureStat")
};

let W = innerWidth;
let H = innerHeight;
let DPR = Math.min(devicePixelRatio || 1, 2);
let lastFrameAt = performance.now();
let fpsEMA = 60;
let adaptive = true;
let frameCounter = 0;

const cfg = {
  force: 1.25,
  radius: 58,
  visc: .965,
  vort: .8,
  glow: 1.2,
  trail: .07,
  particleTarget: 1200,
  particleMin: 650,
  particleMax: 2100
};

const presets = {
  silk:  { force:1.25, radius:58, visc:.965, vort:.80, glow:1.20, trail:.070 },
  neon:  { force:1.55, radius:52, visc:.972, vort:1.05, glow:1.85, trail:.052 },
  storm: { force:2.15, radius:72, visc:.951, vort:1.70, glow:1.40, trail:.088 },
  calm:  { force:.72,  radius:82, visc:.982, vort:.25, glow:.85, trail:.045 }
};

function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
function lerp(a,b,t){ return a+(b-a)*t; }
function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

function resize(){
  W = innerWidth; H = innerHeight; DPR = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.floor(W * DPR);
  canvas.height = Math.floor(H * DPR);
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  ctx.setTransform(DPR,0,0,DPR,0,0);
  const densityTarget = Math.round(clamp((W*H)/650, 850, cfg.particleMax));
  cfg.particleTarget = densityTarget;
  reconcileParticles();
}
addEventListener("resize", resize, { passive:true });

const particles = [];
function makeParticle(){
  const x = Math.random()*W, y = Math.random()*H;
  return {
    x, y, px:x, py:y,
    vx:(Math.random()-.5)*.18,
    vy:(Math.random()-.5)*.18,
    hue:Math.random()*360,
    size:.45+Math.random()*1.8
  };
}
function seed(n = cfg.particleTarget){
  particles.length = 0;
  for(let i=0;i<n;i++) particles.push(makeParticle());
  ui.particles.textContent = `${particles.length} particles`;
}
function reconcileParticles(){
  const target = Math.round(clamp(cfg.particleTarget, cfg.particleMin, cfg.particleMax));
  while(particles.length < target) particles.push(makeParticle());
  if(particles.length > target) particles.length = target;
  ui.particles.textContent = `${particles.length} particles`;
}

const sources = [];
function addForce(x,y,vx,vy,r=cfg.radius,amount=cfg.force,type="flow"){
  if(!Number.isFinite(x+y+vx+vy+r+amount)) return;
  sources.push({x,y,vx,vy,r,amount,life:1,type});
  if(sources.length > 34) sources.splice(0, sources.length-34);
}

const pointer = {x:W/2,y:H/2,px:W/2,py:H/2,down:false};
canvas.addEventListener("pointerdown", e => {
  pointer.down=true; pointer.x=e.clientX; pointer.y=e.clientY; pointer.px=pointer.x; pointer.py=pointer.y;
  canvas.setPointerCapture?.(e.pointerId);
});
canvas.addEventListener("pointerup", () => pointer.down=false);
canvas.addEventListener("pointercancel", () => pointer.down=false);
canvas.addEventListener("pointerleave", () => pointer.down=false);
canvas.addEventListener("pointermove", e => { pointer.x=e.clientX; pointer.y=e.clientY; });

function hsv(h,s,v){
  h=((h%360)+360)%360/60;
  const i=Math.floor(h), f=h-i, p=v*(1-s), q=v*(1-s*f), t=v*(1-s*(1-f));
  return [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i];
}

let auto = false;
let cameraMode = false;
let previewVisible = true;
let handLandmarker = null;
let cameraStream = null;
let trackingLoopId = 0;
let lastDetectAt = 0;
const DETECT_INTERVAL = 1000/24;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const handState = new Map();
function getHandState(index){
  if(!handState.has(index)) handState.set(index, {
    x:W/2,y:H/2,px:W/2,py:H/2,lastT:performance.now(),
    gesture:"none", openLatch:false, fistLatch:false
  });
  return handState.get(index);
}

function fingerExtended(lm, tip, pip, mcp){
  const wrist = lm[0];
  return dist(lm[tip], wrist) > dist(lm[pip], wrist) * 1.08 && dist(lm[tip], lm[mcp]) > dist(lm[pip], lm[mcp]) * .86;
}

function classifyGesture(lm){
  const index = fingerExtended(lm,8,6,5);
  const middle = fingerExtended(lm,12,10,9);
  const ring = fingerExtended(lm,16,14,13);
  const pinky = fingerExtended(lm,20,18,17);
  const extendedCount = [index,middle,ring,pinky].filter(Boolean).length;

  const palmScale = Math.max(.025, dist(lm[5], lm[17]));
  const pinchNorm = dist(lm[4], lm[8]) / palmScale;
  const pinch = clamp(1 - (pinchNorm - .22)/.65, 0, 1);

  if(pinch > .62) return {name:"pinch", pinch, extendedCount};
  if(extendedCount >= 4) return {name:"open", pinch, extendedCount};
  if(extendedCount === 0) return {name:"fist", pinch, extendedCount};
  if(index && extendedCount <= 2) return {name:"point", pinch, extendedCount};
  return {name:"move", pinch, extendedCount};
}

async function initHandTracking(){
  setStatus("MEMUAT HAND TRACKING", "loading");
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm");
  try{
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions:{ modelAssetPath:MODEL_URL, delegate:"GPU" },
      runningMode:"VIDEO",
      numHands:2,
      minHandDetectionConfidence:.55,
      minHandPresenceConfidence:.55,
      minTrackingConfidence:.55
    });
  }catch(gpuError){
    console.warn("GPU delegate gagal, fallback ke CPU", gpuError);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions:{ modelAssetPath:MODEL_URL, delegate:"CPU" },
      runningMode:"VIDEO",
      numHands:2,
      minHandDetectionConfidence:.55,
      minHandPresenceConfidence:.55,
      minTrackingConfidence:.55
    });
  }
}

async function startCamera(){
  if(cameraMode) return;
  try{
    if(!navigator.mediaDevices?.getUserMedia) throw new Error("Browser tidak mendukung getUserMedia");
    if(!handLandmarker) await initHandTracking();

    cameraStream = await navigator.mediaDevices.getUserMedia({
      video:{ facingMode:"user", width:{ideal:960}, height:{ideal:720}, frameRate:{ideal:30,max:30} },
      audio:false
    });
    video.srcObject = cameraStream;
    await video.play();
    video.style.display = previewVisible ? "block" : "none";
    cameraMode = true;
    intro.classList.add("hidden");
    setStatus("CAMERA HAND", "camera");
    $("cameraBtn").textContent = "■ Stop";
    startTrackingLoop();
    showToast("Hand tracking aktif");
  }catch(err){
    console.error(err);
    setStatus("KAMERA GAGAL", "error");
    intro.classList.remove("hidden");
    showToast(cameraErrorMessage(err));
  }
}

function stopCamera(){
  cameraMode = false;
  cancelAnimationFrame(trackingLoopId);
  if(cameraStream) cameraStream.getTracks().forEach(t=>t.stop());
  cameraStream = null;
  video.srcObject = null;
  video.style.display = "none";
  handState.clear();
  ui.gesture.textContent = "No hand";
  $("cameraBtn").textContent = "📷 Kamera";
  setStatus("TOUCH MODE", "idle");
}

function cameraErrorMessage(err){
  if(err?.name === "NotAllowedError") return "Izin kamera ditolak. Aktifkan izin kamera untuk situs ini.";
  if(err?.name === "NotFoundError") return "Kamera tidak ditemukan pada perangkat ini.";
  if(!isSecureContext) return "Kamera butuh HTTPS atau localhost.";
  return `Kamera gagal: ${err?.message || err?.name || "unknown error"}`;
}

function startTrackingLoop(){
  const tick = () => {
    if(!cameraMode || !handLandmarker) return;
    const now = performance.now();
    if(video.readyState >= 2 && now-lastDetectAt >= DETECT_INTERVAL){
      lastDetectAt = now;
      try{
        const result = handLandmarker.detectForVideo(video, now);
        processHands(result, now);
      }catch(err){
        console.warn("Hand detection frame skipped", err);
      }
    }
    trackingLoopId = requestAnimationFrame(tick);
  };
  trackingLoopId = requestAnimationFrame(tick);
}

function processHands(result, now){
  const hands = result?.landmarks || [];
  if(!hands.length){
    ui.gesture.textContent = "No hand";
    for(const state of handState.values()) state.gesture = "none";
    return;
  }

  const labels = [];
  hands.forEach((lm, i) => {
    const state = getHandState(i);
    const g = classifyGesture(lm);
    const tip = lm[8];
    const palm = lm[9];
    const rawX = (1-tip.x)*W;
    const rawY = tip.y*H;
    const smooth = g.name === "fist" ? .28 : .36;
    const x = lerp(state.x, rawX, smooth);
    const y = lerp(state.y, rawY, smooth);
    const dt = Math.max(.016,(now-state.lastT)/1000);
    const vx = (x-state.x)/dt;
    const vy = (y-state.y)/dt;
    const speed = Math.hypot(vx,vy);
    state.px=state.x; state.py=state.y; state.x=x; state.y=y; state.lastT=now;
    labels.push(g.name.toUpperCase());

    if(g.name === "pinch"){
      const radius = lerp(18, cfg.radius*.9, 1-g.pinch);
      addForce(x,y,vx*.0008,vy*.0008,radius,cfg.force*1.15,"pinch");
    }else if(g.name === "open"){
      const px=(1-palm.x)*W, py=palm.y*H;
      if(!state.openLatch || speed>560){
        addForce(px,py,0,0,cfg.radius*1.7,cfg.force*2.2,"burst");
        state.openLatch=true;
      }else{
        addForce(px,py,vx*.00035,vy*.00035,cfg.radius*1.25,cfg.force*.55,"flow");
      }
    }else if(g.name === "fist"){
      const px=(1-palm.x)*W, py=palm.y*H;
      addForce(px,py,0,0,cfg.radius*1.45,cfg.force*1.25,"vortex");
      state.fistLatch=true;
    }else{
      addForce(x,y,vx*.00105,vy*.00105,cfg.radius,cfg.force*(1+Math.min(1,speed/1000)),"flow");
    }

    if(g.name !== "open") state.openLatch=false;
    if(g.name !== "fist") state.fistLatch=false;
    state.gesture=g.name;
  });

  for(const key of [...handState.keys()]) if(key >= hands.length) handState.delete(key);
  ui.gesture.textContent = labels.join(" + ");
}

function applySourceToParticle(p,s){
  const dx=p.x-s.x, dy=p.y-s.y;
  const d2=dx*dx+dy*dy;
  if(d2>=s.r*s.r) return;
  const d=Math.sqrt(d2)+.001;
  const fall=1-d/s.r;

  if(s.type === "burst"){
    const push = s.amount*fall*.42;
    p.vx += (dx/d)*push;
    p.vy += (dy/d)*push;
    return;
  }
  if(s.type === "vortex"){
    const swirl = cfg.vort*s.amount*fall*.31;
    const pull = s.amount*fall*.035;
    p.vx += (-dy/d)*swirl - (dx/d)*pull;
    p.vy += ( dx/d)*swirl - (dy/d)*pull;
    return;
  }

  const swirl=cfg.vort*fall;
  p.vx += s.vx*s.amount*fall + (-dy/d)*swirl*.22;
  p.vy += s.vy*s.amount*fall + ( dx/d)*swirl*.22;
}

function render(){
  const frameNow = performance.now();
  const dtMs = Math.max(1, frameNow-lastFrameAt);
  lastFrameAt = frameNow;
  const instantFps = 1000/dtMs;
  fpsEMA = lerp(fpsEMA, instantFps, .06);

  ctx.fillStyle=`rgba(2,2,5,${cfg.trail})`;
  ctx.fillRect(0,0,W,H);
  const now=frameNow/1000;

  if(auto){
    const cx=W*.5+Math.cos(now*.73)*W*.28;
    const cy=H*.5+Math.sin(now*1.07)*H*.29;
    addForce(cx,cy,Math.cos(now*1.55)*.48,Math.sin(now*1.31)*.48,cfg.radius*1.15,cfg.force*.8,"flow");
  }
  if(pointer.down){
    const vx=pointer.x-pointer.px, vy=pointer.y-pointer.py;
    addForce(pointer.x,pointer.y,vx*.085,vy*.085,cfg.radius,cfg.force,"flow");
  }
  pointer.px=pointer.x; pointer.py=pointer.y;

  for(const s of sources){
    for(const p of particles) applySourceToParticle(p,s);
    s.life -= s.type === "burst" ? .16 : .085;
  }
  for(let i=sources.length-1;i>=0;i--) if(sources[i].life<=0) sources.splice(i,1);

  ctx.globalCompositeOperation="lighter";
  for(const p of particles){
    p.px=p.x; p.py=p.y;
    p.vx*=cfg.visc; p.vy*=cfg.visc;
    p.vx += Math.sin((p.y+p.x)*.0057+now)*.0022;
    p.vy += Math.cos((p.x-p.y)*.0048-now)*.0022;
    p.x+=p.vx; p.y+=p.vy;

    if(p.x<-12)p.x=W+12; else if(p.x>W+12)p.x=-12;
    if(p.y<-12)p.y=H+12; else if(p.y>H+12)p.y=-12;

    const speed=Math.min(9,Math.hypot(p.vx,p.vy));
    p.hue += .28 + speed*.78;
    const alpha=Math.min(.82,.13+speed*.09);
    const [r,g,b]=hsv(p.hue,.92,1);
    ctx.beginPath();
    ctx.strokeStyle=`rgba(${r*255|0},${g*255|0},${b*255|0},${alpha})`;
    ctx.lineWidth=p.size*(1+speed*.42)*cfg.glow;
    ctx.moveTo(p.px,p.py); ctx.lineTo(p.x,p.y); ctx.stroke();
  }
  ctx.globalCompositeOperation="source-over";

  if(sources.length){
    const last=sources[sources.length-1];
    ctx.beginPath();
    ctx.strokeStyle=`rgba(210,230,255,${.06*last.life})`;
    ctx.lineWidth=1;
    ctx.arc(last.x,last.y,last.r*(1.1+(1-last.life)*.5),0,Math.PI*2);
    ctx.stroke();
  }

  frameCounter++;
  if(frameCounter%24===0){
    ui.fps.textContent=`${fpsEMA.toFixed(0)} FPS`;
    if(adaptive) adaptParticleBudget();
  }
  requestAnimationFrame(render);
}

function adaptParticleBudget(){
  const current=cfg.particleTarget;
  if(fpsEMA<43 && current>cfg.particleMin){
    cfg.particleTarget=Math.max(cfg.particleMin,Math.floor(current*.90));
    reconcileParticles();
  }else if(fpsEMA>57 && current<cfg.particleMax){
    cfg.particleTarget=Math.min(cfg.particleMax,Math.ceil(current*1.04));
    reconcileParticles();
  }
}

function setStatus(text,state="idle"){
  statusEl.textContent=text;
  statusEl.dataset.state=state;
}
let toastTimer;
function showToast(text){
  toast.textContent=text;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer=setTimeout(()=>toast.classList.remove("show"),2200);
}

const controls=["force","radius","visc","vort","glow","trail"];
for(const id of controls){
  const el=$(id), out=$(id+"V");
  el.addEventListener("input",()=>{
    cfg[id]=+el.value;
    out.textContent=id==="visc"||id==="trail"?cfg[id].toFixed(3):cfg[id].toFixed(id==="radius"?0:2);
    document.querySelectorAll(".preset").forEach(b=>b.classList.remove("active"));
  });
}

function applyPreset(name){
  const preset=presets[name]; if(!preset) return;
  Object.assign(cfg,preset);
  for(const id of controls){
    $(id).value=cfg[id];
    $(id+"V").textContent=id==="visc"||id==="trail"?cfg[id].toFixed(3):cfg[id].toFixed(id==="radius"?0:2);
  }
  document.querySelectorAll(".preset").forEach(b=>b.classList.toggle("active",b.dataset.preset===name));
  showToast(`${name[0].toUpperCase()+name.slice(1)} preset`);
}

document.querySelectorAll(".preset").forEach(btn=>btn.addEventListener("click",()=>applyPreset(btn.dataset.preset)));
$("settingsBtn").addEventListener("click",()=>{
  const open=panel.classList.toggle("open");
  $("settingsBtn").setAttribute("aria-expanded",String(open));
});
$("closePanelBtn").addEventListener("click",()=>{panel.classList.remove("open");$("settingsBtn").setAttribute("aria-expanded","false")});
$("permissionBtn").addEventListener("click",startCamera);
$("touchOnlyBtn").addEventListener("click",()=>intro.classList.add("hidden"));
$("cameraBtn").addEventListener("click",()=>cameraMode?stopCamera():startCamera());
$("touchBtn").addEventListener("click",()=>{stopCamera(); auto=false; $("autoBtn").classList.remove("active"); $("touchBtn").classList.add("active")});
$("autoBtn").addEventListener("click",()=>{auto=!auto; $("autoBtn").classList.toggle("active",auto); if(auto) $("touchBtn").classList.remove("active")});
$("clearBtn").addEventListener("click",()=>{seed();sources.length=0;showToast("Fluid di-reset")});
$("burstBtn").addEventListener("click",()=>{addForce(W*.5,H*.5,0,0,Math.min(W,H)*.28,cfg.force*3,"burst");showToast("Center burst")});
$("previewBtn").addEventListener("click",()=>{previewVisible=!previewVisible;if(cameraMode)video.style.display=previewVisible?"block":"none";showToast(previewVisible?"Preview kamera tampil":"Preview kamera disembunyikan")});
$("adaptiveBtn").addEventListener("click",e=>{adaptive=!adaptive;e.currentTarget.classList.toggle("on",adaptive);e.currentTarget.setAttribute("aria-pressed",String(adaptive));showToast(adaptive?"Adaptive FPS aktif":"Adaptive FPS nonaktif")});

addEventListener("visibilitychange",()=>{ if(document.hidden) pointer.down=false; });
addEventListener("beforeunload",stopCamera);

resize();
seed();
render();
