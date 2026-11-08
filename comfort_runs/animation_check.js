
(() => {
  const root = document.getElementById('comfort-replay');
  const data = JSON.parse(root.querySelector('#cr-data').textContent);
  const caseInput = root.querySelector('#cr-case'), speedInput = root.querySelector('#cr-speed');
  const timeline = root.querySelector('#cr-time'), play = root.querySelector('#cr-play');
  const status = root.querySelector('#cr-status');
  const kind = 'comfort-v2-replay-20261108';
  const source = `comfort-${Math.random().toString(36).slice(2)}`;
  let caseKey = 'gentle', speed = 1, position = 0, playing = false, lastTime = null, raf = null;
  let selected, extent;
  const panels = ['v1','final'].map(name => {
    const section = root.querySelector(`[data-model="${name}"]`), svg = section.querySelector('svg');
    svg.innerHTML = '<line class="cr-track"/><line class="cr-center"/><rect class="cr-cart" rx="4"/><circle class="cr-wheel" r="5"/><circle class="cr-wheel" r="5"/><line class="cr-pole"/><circle class="cr-head" r="8"/><g class="cr-wind-vector"><line class="cr-arrow"/><polygon class="cr-arrowhead"/></g><text class="cr-label text-small" text-anchor="middle">0 m</text>';
    return {name,section,svg};
  });
  const fmt = (n, digits = 2) => n.toFixed(digits).replace('.',',').replace('-','−');
  const set = (el, attrs) => { for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v); };
  function choose() {
    selected = data.scenarios.find(s => s.key === caseKey) || data.scenarios[0];
    caseKey = selected.key;
    extent = Math.max(.8,...Object.values(selected.models).flatMap(r => r.frames.map(f => Math.abs(f[0]) + Math.abs(Math.sin(f[2])) + .3)));
    caseInput.value = caseKey; speedInput.value = String(speed);
    root.querySelector('#cr-profile').textContent = `maks. ±${fmt(selected.amplitude)} N · ciklus ${fmt(selected.period,0)} s · seed ${selected.seed}`;
    for (const p of panels) {
      const r = selected.models[p.name];
      p.section.querySelector('[data-field="summary"]').textContent = `Ceo snimak: RMS ubrzanja ${fmt(r.head_acc_rms,3)} m/s² · RMS trzaja ${fmt(r.head_jerk_rms,2)} m/s³ · ${r.survived ? 'uspešnih 10 s' : `pad na ${fmt(r.steps*data.dt)} s`}`;
    }
  }
  function draw() {
    const step = Math.min(500,Math.floor(position));
    timeline.value = String(step);
    root.querySelector('#cr-clock').textContent = `${fmt(step*data.dt)} / 10,00 s`;
    const wind = selected.models.final.frames[Math.min(step,selected.models.final.steps)][4];
    root.querySelector('#cr-wind').textContent = `Vetar sada: ${fmt(wind,3)} N`;
    for (const p of panels) {
      const run = selected.models[p.name], f = run.frames[Math.min(step,run.steps)];
      const width = Math.max(240,p.svg.getBoundingClientRect().width), height = 220;
      const scale = Math.min(120,(width-70)/(2*extent));
      const x = width/2 + f[0]*scale, y = 176;
      const tipX = x + Math.sin(f[2])*scale, tipY = y - Math.cos(f[2])*scale;
      const midX = (x+tipX)/2, midY = (y+tipY)/2;
      set(p.svg,{viewBox:`0 0 ${width} ${height}`});
      set(p.svg.querySelector('.cr-track'),{x1:12,y1:198,x2:width-12,y2:198});
      set(p.svg.querySelector('.cr-center'),{x1:width/2,y1:30,x2:width/2,y2:198});
      set(p.svg.querySelector('.cr-cart'),{x:x-23,y:y,width:46,height:15});
      p.svg.querySelectorAll('.cr-wheel').forEach((w,i)=>set(w,{cx:x+(i?15:-15),cy:193}));
      set(p.svg.querySelector('.cr-pole'),{x1:x,y1:y,x2:tipX,y2:tipY});
      set(p.svg.querySelector('.cr-head'),{cx:tipX,cy:tipY});
      const vector = p.svg.querySelector('.cr-wind-vector');
      const length = 320*f[4], endX = midX+length, sign = Math.sign(length);
      vector.style.display = Math.abs(length)<.6 ? 'none':'';
      set(p.svg.querySelector('.cr-arrow'),{x1:midX,y1:midY,x2:endX,y2:midY});
      set(p.svg.querySelector('.cr-arrowhead'),{points:`${endX},${midY} ${endX-sign*7},${midY-4} ${endX-sign*7},${midY+4}`});
      set(p.svg.querySelector('.cr-label'),{x:width/2,y:216});
      const field = (name,text) => { p.section.querySelector(`[data-field="${name}"]`).textContent=text; };
      field('acc',`Ubrzanje glave: ${f[7]===null?'—':fmt(f[7],3)} m/s²`);
      field('jerk',`Trzaj glave: ${f[8]===null?'—':fmt(f[8],2)} m/s³`);
      field('force',`Sila motora: ${f[5]===null?'—':fmt(f[5],2)} N`);
      field('angle',`Nagib: ${fmt(f[2]*180/Math.PI,1)}° · x = ${fmt(f[0])} m`);
    }
    play.textContent = playing ? 'Pauziraj' : (position>=500 ? 'Ponovi' : 'Pokreni');
  }
  function save() {
    window.openai?.setWidgetState?.({modelContent:{kind,scenario:caseKey,seed:selected.seed,step:Math.floor(position),rewardVersion:2},privateContent:{source,speed}})?.catch(()=>{});
  }
  function stop() { playing=false; lastTime=null; if(raf!==null) cancelAnimationFrame(raf); raf=null; }
  function tick(now) {
    if(!playing) return;
    if(lastTime!==null) position=Math.min(500,position+(now-lastTime)/1000*speed/data.dt);
    lastTime=now;
    if(position>=500) { stop(); status.textContent=Object.values(selected.models).every(r=>r.survived) ? 'Završeno · oba modela su ostala uspravna svih 10 s.' : 'Završeno · snimak svakog modela se zaustavlja pri padu.'; draw(); save(); return; }
    draw(); raf=requestAnimationFrame(tick);
  }
  function restore(saved) {
    const m=saved?.modelContent, p=saved?.privateContent;
    if(m?.kind!==kind || p?.source===source) return;
    stop();
    if(data.scenarios.some(s=>s.key===m.scenario)) caseKey=m.scenario;
    if(Number.isFinite(m.step)) position=Math.max(0,Math.min(500,m.step));
    if([.1,.5,1].includes(p?.speed)) speed=p.speed;
    choose(); draw();
  }
  play.addEventListener('click',()=>{
    if(playing) { stop(); status.textContent='Pauzirano.'; }
    else { if(position>=500) position=0; playing=true; lastTime=null; status.textContent='Reprodukcija.'; raf=requestAnimationFrame(tick); }
    draw(); save();
  });
  caseInput.addEventListener('change',()=>{ stop();caseKey=caseInput.value;position=0;choose();draw();status.textContent='Izabran novi snimak.';save(); });
  speedInput.addEventListener('change',()=>{ speed=Number(speedInput.value);lastTime=null;save(); });
  timeline.addEventListener('input',()=>{ stop();position=Number(timeline.value);draw(); });
  timeline.addEventListener('change',()=>{ status.textContent=`Pauzirano na ${fmt(position*data.dt)} s.`;save(); });
  window.addEventListener('openai:set_globals',event=>{ if(event.detail?.globals?.widgetState) restore(event.detail.globals.widgetState); });
  new ResizeObserver(()=>draw()).observe(root);
  choose();restore(window.openai?.widgetState);draw();
})();
