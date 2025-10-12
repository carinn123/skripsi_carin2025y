// ===== NAV (toggle section) =====
// ===== NAV (toggle section) =====
// document.querySelectorAll('.nav-link').forEach(link=>{
//   link.addEventListener('click', e=>{
//     e.preventDefault();
//     const targetId = link.getAttribute('data-section');
//     if (!targetId) return;

//     document.querySelectorAll('.nav-link').forEach(l=> l.classList.remove('active'));
//     document.querySelectorAll('.content-section').forEach(s=> s.classList.remove('active'));

//     link.classList.add('active');
//     const section = document.getElementById(targetId);
//     if (section) {
//       section.classList.add('active');
//       section.scrollIntoView({ behavior:'smooth', block:'start' });
//     }

//     // khusus: kalau klik upload, panggil loadUpload()
//     if (targetId === 'upload' && typeof window.loadUpload === 'function') {
//       try { window.loadUpload(); } catch(e){ console.error(e); }
//     }
//   });
// });

document.getElementById('scroll-to-beranda')
  ?.addEventListener('click', ()=>{
    // collapse hero supaya atas halaman tinggal navbar
    document.body.classList.add('hero-collapsed');

    // pindah ke tab Beranda & scroll halus
    document.querySelector('.nav-link[data-section="beranda"]')?.click();
    document.getElementById('beranda')?.scrollIntoView({ behavior:'smooth', block:'start' });

    // opsional: set hash supaya reload tetap tanpa hero
    history.replaceState(null, '', '#beranda');
  });


// ===== Utils =====
const monthsID=["","Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November","Desember"];
const rupiah=n=>new Intl.NumberFormat('id-ID').format(n);
const EVAL_DEFAULT_DAYS = 120; // jangka waktu otomatis untuk grafik evaluasi
const fmtNum = (x, dec=0) =>
  (x==null || Number.isNaN(x)) ? '-' :
  new Intl.NumberFormat('id-ID', { maximumFractionDigits: dec, minimumFractionDigits: dec }).format(x);

function normProv(s){return (s||"").toString().toLowerCase().replace(/provinsi|propinsi|prov\./g,"").replace(/\s+/g,"").replace(/[^\w]/g,"");}

// --- replace existing fetchJsonSafe with this robust helper ---
async function fetchJsonSafe(url, opts) {
  const r = await fetch(url, opts);
  const status = r.status;
  let text = null;
  try {
    text = await r.text();
  } catch (e) {
    text = null;
  }

  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (e) {
      data = null;
    }
  }

  return {
    ok: r.ok,
    status,
    data,   // parsed JSON or null
    text    // raw text (useful for debugging HTML errors)
  };
}

function fillSelect(selectId,records){
  const sel=document.getElementById(selectId); if(!sel) return;
  const firstOption = sel.querySelector('option[value=""]');
  sel.innerHTML = ""; // kosongkan dulu
  // pastikan ada placeholder paling atas
  const ph = document.createElement('option');
  ph.value = ""; ph.textContent = "-- Pilih Kabupaten/Kota --";
  sel.appendChild(ph);
  const frag=document.createDocumentFragment();
  for(const r of records){
    const opt=document.createElement('option');
    opt.value = r.slug || r.label;          // slug kalau ada, fallback label
    opt.textContent = r.label;
    opt.dataset.label = r.label || "";
    opt.dataset.slug  = r.slug  || "";
    opt.dataset.entity= r.entity|| "";
    frag.appendChild(opt);
  }
  sel.appendChild(frag);
}
function wireSearch(){
  const search=document.getElementById('search-kab');
  const sel=document.getElementById('kabupaten');
  if(!search||!sel) return;
  search.addEventListener('input',()=>{
    const q=search.value.toLowerCase();
    for(const opt of sel.options){
      if(!opt.value) continue;
      const label=(opt.dataset.label||opt.textContent).toLowerCase();
      opt.hidden=!label.includes(q);
    }
  });
}
function selectedSlugOrLabel(selectEl){
  if(!selectEl) return "";
  const opt = selectEl.options[selectEl.selectedIndex];
  return (opt?.dataset?.slug || opt?.dataset?.label || opt?.value || "").trim().toLowerCase();
}
// function getSelectedCityLabel(){ // dipakai mode tren single lama
//   const sel=document.getElementById('kabupaten');
//   const opt=sel?.options[sel.selectedIndex];
//   return opt?.dataset?.label || sel?.value || "";
// }

// ===== Inisialisasi Cities (semua dropdown yang ada)
async function initCities(){
  const targetIds = ['kabupaten','kabupaten2','kabupaten_a','ev_city']
    .filter(id => document.getElementById(id));
  if (targetIds.length === 0) return;

  try{
    let res=await fetch('/api/cities_full'); 
    if(!res.ok) throw new Error("fallback");
    const data=await res.json(); // [{entity, slug, label}]
    for (const id of targetIds) {
      fillSelect(id, data.map(d=>({label:d.label, slug:d.slug, entity:d.entity})));
    }
  }catch(_){
    try{ 
      const r2=await fetch('/api/cities'); 
      const arr=await r2.json(); 
      for (const id of targetIds) fillSelect(id, arr.map(l=>({label:l})));
    }
    catch(e){ console.error("Gagal memuat daftar kota/kab:", e); }
  }
  wireSearch(); // search hanya untuk #kabupaten
}

// ===== Islands (Beranda)
(async function initIslands(){
  try{
    const res=await fetch('/api/islands'); 
    const islands=await res.json();
    const sel=document.getElementById('pulau'); if(!sel) return;
    islands.forEach(n=>{
      const opt=document.createElement('option'); 
      opt.value=n; opt.textContent=n; sel.appendChild(opt);
    });
  }catch(e){ console.error(e); }
})();

// Enable minggu (global + beranda)
const bulanSel=document.getElementById('bulan');
const mingguSel=document.getElementById('minggu');
bulanSel?.addEventListener('change',()=>{
  if(!bulanSel.value){mingguSel.value="";mingguSel.disabled=true;}
  else {mingguSel.disabled=false;}
});
const b_bulanSel=document.getElementById('b_bulan');
const b_mingguSel=document.getElementById('b_minggu');
b_bulanSel?.addEventListener('change',()=>{
  if(!b_bulanSel.value){b_mingguSel.value="";b_mingguSel.disabled=true;}
  else {b_mingguSel.disabled=false;}
});

// ===== MAP (Beranda)
let __PROV_GJ=null;
async function getProvGeoJSON(){ 
  if(__PROV_GJ) return __PROV_GJ; 
  const r=await fetch('/static/indonesia_provinces.geojson?v=1',{cache:'no-store'}); 
  if(!r.ok) throw new Error('GeoJSON provinsi tidak ditemukan'); 
  __PROV_GJ=await r.json(); 
  return __PROV_GJ; 
}
let map,geoLayer;
function ensureMap(){ 
  if(map) return map; 
  map=L.map('map',{scrollWheelZoom:true}).setView([-2.5,118],5); 
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:10, attribution:'&copy; OpenStreetMap'}).addTo(map); 
  return map; 
}

async function showMapBeranda(){
  const island=document.getElementById('pulau').value || 'Semua Pulau';
  const tahun=document.getElementById('b_tahun').value;
  const bulan=document.getElementById('b_bulan').value;
  const minggu=document.getElementById('b_minggu').value;
  const loading=document.getElementById('b_loading');
  if(!tahun){ alert('Pilih tahun dulu.'); return; }
  loading.style.display='inline-block';
  try{
    const url=`/api/choropleth?island=${encodeURIComponent(island)}&year=${encodeURIComponent(tahun)}&month=${encodeURIComponent(bulan)}&week=${encodeURIComponent(minggu)}`;
    const res=await fetch(url); const js=await res.json(); if(!res.ok) throw new Error(js.error||'Server error');
    const m=ensureMap(); const gj=await getProvGeoJSON();
    const vmap=Object.fromEntries(js.data.map(d=>[normProv(d.province),{val:d.value,cat:d.category,label:d.province}]));
    if(geoLayer) geoLayer.remove();
    geoLayer=L.geoJSON(gj,{
      style: f=>{
        const raw=f.properties.Propinsi||f.properties.PROVINSI||f.properties.provinsi||f.properties.name||f.properties.NAMOBJ||"";
        const rec=vmap[normProv(raw)];
        const fill=rec? (rec.cat==='low'?'#2ecc71':rec.cat==='mid'?'#f1c40f':'#e74c3c') : '#bdc3c7';
        return {color:'#fff',weight:1,fillColor:fill,fillOpacity:.85};
      },
      onEachFeature:(feature,layer)=>{
        const raw=feature.properties.Propinsi||feature.properties.PROVINSI||feature.properties.provinsi||feature.properties.name||feature.properties.NAMOBJ||"—";
        const rec=vmap[normProv(raw)]; const val=rec? Math.round(rec.val):null; const cat=rec? rec.cat:'no-data';
        layer.bindPopup(`<b>${raw}</b><br/>${val?('Rp '+rupiah(val)):'—'}<br/>Kategori: ${cat}`);
      }
    }).addTo(m);
    try{ m.fitBounds(geoLayer.getBounds(), {padding:[20,20]}); }catch(e){}
    document.getElementById('statPulau').textContent=island||'Semua Pulau';
    let scope=`Tahun ${tahun}`; 
    if(bulan) scope=`Bulan ${monthsID[+bulan]} ${tahun}`; 
    if(bulan&&minggu) scope=`Minggu ke-${minggu}, ${monthsID[+bulan]} ${tahun}`;
    document.getElementById('statScope').textContent=scope;
    document.getElementById('statLast').textContent=js.last_actual||'-';
  }catch(e){ console.error(e); alert('Gagal menampilkan peta: '+e.message); }
  finally{ loading.style.display='none'; }
}
window.showMapBeranda = showMapBeranda;






// ===== TINJAUAN TREN (single & compare)
let trendChart=null;
function renderTrendChart(series,granularity,title){
  const placeholder=document.getElementById('trendPlaceholder');
  const canvas=document.getElementById('trendChart'); if(!canvas) return;
  const ctx=canvas.getContext('2d');
  const labels=series.map(s=>s.label);
  const values=series.map(s=>s.value);
  if(!series.length){ placeholder.style.display='block'; if(trendChart){trendChart.destroy();trendChart=null;} return; }
  placeholder.style.display='none';
  const datasetLabel= granularity==='yearly'?'Rata-rata per Bulan': (granularity==='monthly'?'Rata-rata per Minggu':'Harian (Minggu terpilih)');
  const chartData={labels, datasets:[{label:datasetLabel, data: values, tension:.25, fill:true, pointRadius:3, borderWidth:2}]};
  const options={responsive:true, maintainAspectRatio:false, plugins:{title:{display:true,text:title}, tooltip:{callbacks:{label:(c)=>` ${datasetLabel}: Rp ${rupiah(c.parsed.y)}`}}}, scales:{y:{ticks:{callback:v=>'Rp '+rupiah(v)}}, x:{ticks:{autoSkip:true,maxRotation:0}}}};
  if(trendChart){ trendChart.destroy(); }
  trendChart=new Chart(ctx,{type:'line',data:chartData,options});
}
function _seriesMap(series){ const m=new Map(); for(const s of (series||[])) m.set(s.label, s.value); return m; }
function renderTrendChartCompare(labels, valuesA, valuesB, nameA, nameB, granularity) {
  const placeholder=document.getElementById('trendPlaceholder');
  const canvas=document.getElementById('trendChart'); if(!canvas) return;
  const ctx=canvas.getContext('2d');
  if (!labels.length) { 
    placeholder.style.display='block'; 
    if (trendChart){ trendChart.destroy(); trendChart=null; } 
    return; 
  }
  placeholder.style.display='none';
  const options={
    responsive:true, maintainAspectRatio:false,
    plugins:{
      title:{display:true,text:`Tren Harga • Perbandingan`},
      tooltip:{callbacks:{label:(c)=>` ${c.dataset.label}: Rp ${rupiah(c.parsed.y)}`}}
    },
    scales:{ y:{ticks:{callback:v=>'Rp '+rupiah(v)}}, x:{ticks:{autoSkip:true,maxRotation:0}} }
  };
  const data = {
    labels,
    datasets:[
      { label: nameA||'Kota 1', data: valuesA, tension:.25, pointRadius:2, borderWidth:2 },
      { label: nameB||'Kota 2', data: valuesB, tension:.25, pointRadius:2, borderWidth:2 }
    ]
  };
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, { type:'line', data, options });
}


function updateTrendStats(s) {
  // s diharapkan punya: min, max, mean, vol_pct (aman kalau null)
  const fmt = n => (n==null || Number.isNaN(n)) ? '-' : rupiah(Math.round(n));
  const fmtPct = n => (n==null || Number.isNaN(n)) ? '-' : n.toFixed(2);

  const elMin  = document.getElementById('hargaMin');
  const elMax  = document.getElementById('hargaMax');
  const elAvg  = document.getElementById('hargaRata');
  const elVol  = document.getElementById('volatilitas');

  if (!elMin || !elMax || !elAvg || !elVol) return;

  elMin.textContent = fmt(s?.min);
  elMax.textContent = fmt(s?.max);
  elAvg.textContent = fmt(s?.mean);
  elVol.textContent = fmtPct(s?.vol_pct);
}

function resetTrendStats() {
  const ids = ['hargaMin','hargaMax','hargaRata','volatilitas'];
  ids.forEach(id => { const el=document.getElementById(id); if (el) el.textContent='-'; });
}

let _trendChartA = null;
let _trendChartB = null;

function _seriesToXY(series){
  const labels = (series||[]).map(s=>s.label);
  const values = (series||[]).map(s=>s.value);
  return {labels, values};
}

function renderTrendChartTo(canvasId, placeholderId, series, granularity, title){
  const ph = document.getElementById(placeholderId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const {labels, values} = _seriesToXY(series);

  if (!labels.length){
    if (ph) ph.style.display = 'block';
    if (canvasId==='trendChartA' && _trendChartA){ _trendChartA.destroy(); _trendChartA=null; }
    if (canvasId==='trendChartB' && _trendChartB){ _trendChartB.destroy(); _trendChartB=null; }
    return;
  } else {
    if (ph) ph.style.display = 'none';
  }

  const datasetLabel = granularity==='yearly' ? 'Rata-rata per Bulan'
                        : granularity==='monthly' ? 'Rata-rata per Minggu'
                        : 'Harian (minggu terpilih)';

  const data = {
    labels,
    datasets: [{ label: datasetLabel, data: values, tension:.25, fill:true, pointRadius:2, borderWidth:2 }]
  };
  const options = {
    responsive:true, maintainAspectRatio:false,
    plugins:{ title:{ display:true, text: title || 'Tren Harga' },
              tooltip:{ callbacks:{ label:(c)=>` ${datasetLabel}: Rp ${rupiah(c.parsed.y)}` } } },
    scales:{ y:{ ticks:{ callback:v=>'Rp '+rupiah(v) }}, x:{ ticks:{ autoSkip:true, maxRotation:0 }}}
  };

  if (canvasId==='trendChartA'){
    if (_trendChartA) _trendChartA.destroy();
    _trendChartA = new Chart(ctx, { type:'line', data, options });
  } else {
    if (_trendChartB) _trendChartB.destroy();
    _trendChartB = new Chart(ctx, { type:'line', data, options });
  }
}

let _trendChartCombined = null;

function renderTrendChartCombined(labels, valuesA, valuesB, nameA, nameB){
  const canvas = document.getElementById('trendChartCombined');
  const ph     = document.getElementById('trendPlaceholderCombined');
  if (!canvas) return;

  if (!labels || labels.length === 0){
    if (ph) ph.style.display = 'block';
    if (_trendChartCombined){ _trendChartCombined.destroy(); _trendChartCombined = null; }
    return;
  } else {
    if (ph) ph.style.display = 'none';
  }

  const ctx = canvas.getContext('2d');
  if (_trendChartCombined) _trendChartCombined.destroy();

  _trendChartCombined = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: nameA || 'Kota 1', data: valuesA, tension:.25, pointRadius:2, borderWidth:2 },
        { label: nameB || 'Kota 2', data: valuesB, tension:.25, pointRadius:2, borderWidth:2 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: true, text: 'Tren Harga • Grafik Gabungan' } },
      scales: {
        y: { ticks: { callback: v => 'Rp ' + rupiah(v) } },
        x: { ticks: { autoSkip: true, maxRotation: 0 } }
      }
    }
  });
}

async function loadData(){
  const selA = document.getElementById('kabupaten');
  const selB = document.getElementById('kabupaten2');
  const cityA = selectedSlugOrLabel(selA);
  const cityB = selectedSlugOrLabel(selB);

  const tahun  = document.getElementById('tahun').value;
  const bulan  = document.getElementById('bulan').value;
  const minggu = document.getElementById('minggu').value;
  const loading = document.getElementById('loadingSpinner');

  if (!cityA || !tahun) { alert('Pilih minimal Kota/Kab 1 dan Tahun.'); return; }

  const q = (c)=> `/api/trend?city=${encodeURIComponent(c)}&year=${encodeURIComponent(tahun)}&month=${encodeURIComponent(bulan)}&week=${encodeURIComponent(minggu)}`;

  loading.style.display='inline-block';
  try{
    // --- KOTA A
    const resA = await fetch(q(cityA)); 
    const trA  = await resA.json();
    if (!resA.ok) throw new Error(trA.error || 'Server error (A)');

    const cityLabelA = (trA.entity||trA.city||'Kota A').replace(/_/g,' ');
    updateTrendStatsTo('A', trA.stats || null, cityLabelA);
    {
      const scopeA = (bulan&&minggu) ? `Minggu ke-${minggu}, ${monthsID[+bulan]} ${tahun}`
                    : bulan ? `${monthsID[+bulan]} ${tahun}` : `Tahun ${tahun}`;
      const titleA = `Tren Harga • ${cityLabelA} • ${scopeA}`;
      renderTrendChartTo('trendChartA','trendPlaceholderA', trA.series||[], trA.granularity, titleA);
    }

    // --- KOTA B (opsional)
    if (cityB){
      const resB = await fetch(q(cityB)); 
      const trB  = await resB.json();
      if (!resB.ok) throw new Error(trB.error || 'Server error (B)');

      const cityLabelB = (trB.entity||trB.city||'Kota B').replace(/_/g,' ');
      updateTrendStatsTo('B', trB.stats || null, cityLabelB);
      {
        const scopeB = (bulan&&minggu) ? `Minggu ke-${minggu}, ${monthsID[+bulan]} ${tahun}`
                      : bulan ? `${monthsID[+bulan]} ${tahun}` : `Tahun ${tahun}`;
        const titleB = `Tren Harga • ${cityLabelB} • ${scopeB}`;
        renderTrendChartTo('trendChartB','trendPlaceholderB', trB.series||[], trB.granularity, titleB);
      }

      // ====== GRAFIK GABUNGAN 2 LINE (A & B) ======
      {
        const labels = Array.from(new Set([
          ...(trA.series||[]).map(s=>s.label),
          ...(trB.series||[]).map(s=>s.label),
        ])).sort((a,b)=>{
          const na = parseInt(a.replace(/\D+/g,''))||0;
          const nb = parseInt(b.replace(/\D+/g,''))||0;
          return (na-nb) || a.localeCompare(b);
        });

        const mA = new Map((trA.series||[]).map(s=>[s.label, s.value]));
        const mB = new Map((trB.series||[]).map(s=>[s.label, s.value]));
        const vA = labels.map(l => mA.has(l) ? mA.get(l) : null);
        const vB = labels.map(l => mB.has(l) ? mB.get(l) : null);

        if (typeof renderTrendChartCombined === 'function') {
          renderTrendChartCombined(
            labels, vA, vB,
            cityLabelA, cityLabelB
          );
        } else {
          // fallback: tampilkan placeholder combined bila renderer belum ada
          const phC = document.getElementById('trendPlaceholderCombined');
          if (phC) phC.style.display = labels.length ? 'none' : 'block';
        }
      }

    } else {
      // jika kota B tak dipilih: kosongkan blok B
      updateTrendStatsTo('B', null, '-');
      const phB = document.getElementById('trendPlaceholderB'); 
      if (phB) phB.style.display='block';
      if (window._trendChartB){ window._trendChartB.destroy(); window._trendChartB=null; }

      // reset grafik gabungan
      const phC = document.getElementById('trendPlaceholderCombined');
      if (phC) phC.style.display = 'block';
      if (window._trendChartCombined){ window._trendChartCombined.destroy(); window._trendChartCombined = null; }
    }

    // fokus ke tab tren
    document.querySelector('.nav-link[data-section="tren"]')?.click();

    // ringkasan singkat ambil dari kota A
    const s = trA.stats || {};
    const fmt = n => (n==null || Number.isNaN(n)) ? '-' : rupiah(Math.round(n));
    const scope = (bulan&&minggu) ? `Minggu ke-${minggu}, ${monthsID[+bulan]} ${tahun}`
                 : bulan ? `${monthsID[+bulan]} ${tahun}` : `Tahun ${tahun}`;
    document.getElementById('analisisTren').textContent =
      `Analisis ${scope} • ${(trA.entity||'Kota 1').replace(/_/g,' ')}: n=${s.n||0}, rata-rata Rp ${fmt(s.mean)}, rentang Rp ${fmt(s.min)}–Rp ${fmt(s.max)}, vol ${(s.vol_pct==null?'-':s.vol_pct.toFixed(2))}%.`;

  } catch(e){
    console.error(e);
    alert('Gagal memuat tren: ' + e.message);
    resetTrendStatsDual();
    // tampilkan placeholder kedua grafik A/B
    document.getElementById('trendPlaceholderA')?.style && (document.getElementById('trendPlaceholderA').style.display='block');
    document.getElementById('trendPlaceholderB')?.style && (document.getElementById('trendPlaceholderB').style.display='block');
    if (window._trendChartA){ window._trendChartA.destroy(); window._trendChartA=null; }
    if (window._trendChartB){ window._trendChartB.destroy(); window._trendChartB=null; }
    // reset grafik gabungan
    const phC = document.getElementById('trendPlaceholderCombined');
    if (phC) phC.style.display = 'block';
    if (window._trendChartCombined){ window._trendChartCombined.destroy(); window._trendChartCombined = null; }
  } finally {
    loading.style.display='none';
  }
}



window.loadData = loadData;

// ===== PREDIKSI (compare 2 kota)
function _fmtNum(n){ return (n==null || Number.isNaN(n)) ? '-' : rupiah(Math.round(n)); }
function _fmtPct(n){ return (n==null || Number.isNaN(n)) ? '-' : Number(n).toFixed(2); }

function updateTrendStatsTo(prefix /* 'A' | 'B' */, stats, cityLabel){
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('statCity'+prefix, cityLabel || '-');
  set('hargaMin'+prefix,  _fmtNum(stats?.min));
  set('hargaMax'+prefix,  _fmtNum(stats?.max));
  set('hargaRata'+prefix, _fmtNum(stats?.mean));
  set('volatilitas'+prefix, _fmtPct(stats?.vol_pct));
}

function resetTrendStatsDual(){
  updateTrendStatsTo('A', null, '-');
  updateTrendStatsTo('B', null, '-');
}
function fullDateRange(start,end){ 
  const s=new Date(start), e=new Date(end); const out=[]; 
  if(isNaN(s)||isNaN(e)||s>e) return out; 
  for(let d=new Date(s); d<=e; d.setDate(d.getDate()+1)){ out.push(new Date(d).toISOString().slice(0,10)); } 
  return out; 
}
const VIZ_ROLL = 30; // rolling non-centered, min_periods≈15 (server)

// ...

function _niceDate(idStr){
  // idStr 'YYYY-MM-DD' -> 'DD MMM YYYY'
  try{
    const d = new Date(idStr+"T00:00:00");
    const opt = {day:'2-digit', month:'short', year:'numeric'};
    return d.toLocaleDateString('id-ID', opt);
  }catch(_){ return idStr || '-'; }
}

function _statsFromSeries(labels, values){
  // labels: ['YYYY-MM-DD', ...]
  // values: [number|null,...] (prediksi atau aktual yang sudah di-mapped ke labels)
  const pts = [];
  for (let i=0;i<labels.length;i++){
    const v = values[i];
    if (v==null || Number.isNaN(v)) continue;
    pts.push({date: labels[i], value: Number(v)});
  }
  if (pts.length===0) {
    return { n:0, avg:null, min:null, min_date:null, max:null, max_date:null, start:null, end:null, change_pct:null };
  }
  const n = pts.length;
  let sum=0, min=Infinity, max=-Infinity, min_date=null, max_date=null;
  for(const p of pts){
    sum += p.value;
    if (p.value < min){ min = p.value; min_date = p.date; }
    if (p.value > max){ max = p.value; max_date = p.date; }
  }
  const avg = sum / n;
  const start = pts[0].value;
  const end   = pts[pts.length-1].value;
  const change_pct = (start===0? null : ((end-start)/start*100));
  return { n, avg, min, min_date, max, max_date, start, end, change_pct };
}

function _renderPredSummary(labels, seriesPred){
  const box = document.getElementById('predSummary');
  if (!box) return;

  const s = _statsFromSeries(labels, seriesPred);
  console.log('pred-summary stats =', s);

  if (s.n === 0){
    box.style.display = 'none';
    return;
  }
  // tampilkan paksa (hapus inline style)
  box.style.removeProperty('display');

  const fmt = n => (n==null || Number.isNaN(n))? '-' : 'Rp '+new Intl.NumberFormat('id-ID').format(Math.round(n));
  const fmtPct = n => (n==null || Number.isNaN(n))? '-' : (n>=0? '+' : '') + n.toFixed(2) + '%';

  document.getElementById('predMaxVal').textContent  = fmt(s.max);
  document.getElementById('predMaxDate').textContent = s.max_date ? _niceDate(s.max_date) : '—';
  document.getElementById('predMinVal').textContent  = fmt(s.min);
  document.getElementById('predMinDate').textContent = s.min_date ? _niceDate(s.min_date) : '—';
  document.getElementById('predAvgVal').textContent  = fmt(s.avg);
  document.getElementById('predCount').textContent   = `n = ${s.n} hari`;
  document.getElementById('predChangePct').textContent = fmtPct(s.change_pct);
  document.getElementById('predChangeNote').textContent = (s.start!=null && s.end!=null)
      ? `${fmt(s.start)} → ${fmt(s.end)}`
      : '—';
}


let _predChart = null;

// === Plugin: paksa jumlah tick X ===
// === Plugin: paksa jumlah tick X jadi persis N ===
const forceXTicksPlugin = {
  id: 'forceXTicks',
  afterBuildTicks(chart, args, opts) {
    const scale = args.scale;               // <— ambil scalenya di sini
    if (scale.axis !== 'x') return;

    const desired = Number(opts?.count) || 12;
    const ticks = scale.ticks || [];
    const total = ticks.length;
    if (total <= desired || desired < 2) return;

    // ambil 'desired' titik merata (termasuk awal & akhir)
    const step = (total - 1) / (desired - 1);
    const keep = [];
    for (let i = 0; i < desired; i++) {
      const idx = Math.round(i * step);
      keep.push(ticks[idx]);
    }

    // timpa ticks hasil build
    args.ticks = keep;      // <- penting di v3/v4
    scale.ticks = keep;     //   (aman juga diset untuk konsistensi)
  }
};
if (window.Chart) Chart.register(forceXTicksPlugin);

// Pastikan plugin ter-register

function renderPredChart(labels, actual, predicted, cityLabel) {
  const canvas = document.getElementById('predChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (_predChart) _predChart.destroy();

  _predChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: `Aktual • ${cityLabel}`,
          data: actual,
          borderWidth: 2,
          tension: .25,            // <-- smooth ringan (seperti Matplotlib)
          stepped: false,          // <-- DULUNYA 'before'
          pointRadius: 0,
          spanGaps: true           // gap di awal (hasil rolling) jangan dihubungkan
        },
        {
          label: `Prediksi • ${cityLabel}`,
          data: predicted,
          borderWidth: 2,
          tension: .25,
          stepped: false,          // <-- DULUNYA 'before'
          pointRadius: 0,
          spanGaps: true,
          borderDash: [6,6]        // opsional: biar keliatan dashed seperti plot
        }
      ]
    },
    options: {
      plugins: {
        title: { display: true, text: 'Aktual vs Prediksi (Gradient Boosting)' },
        forceXTicks: { count: 12 }
      },
      scales: {
        x: {
          ticks: {
            autoSkip: false,
            maxRotation: 0,
            callback(value, index, ticks) {
              const raw = this?.chart?.data?.labels?.[value] ?? (ticks?.[index]?.label ?? value);
              if (typeof raw === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(raw)) {
                const [y,m,d] = raw.split('-');
                const mon = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'][+m-1];
                return `${d} ${mon}`;
              }
              return String(raw ?? '');
            }
          },
          grid: { drawTicks: true }
        },
        y: { ticks: { count: 8 } }
      }
    }
  });
}

function _renderPredSummary(labels, seriesPred){
  const box = document.getElementById('predSummary');
  if (!box) return;

  const s = _statsFromSeries(labels, seriesPred);
  console.log('pred-summary stats =', s);     // <- cek di console

  if (s.n === 0) {
    box.style.display = 'none';
    return;
  }

  // tampilkan paksa: hapus inline style supaya class .stats-grid (display:grid) berlaku
  box.style.removeProperty('display');        // << ganti dari `box.style.display = ''`
  // … (lanjutan set isi kartu seperti sudah kamu tulis) …
}

function _renderPredSummaryFromAPI(predictedArray){
  const labels = predictedArray.map(p => p.date);
  const values = predictedArray.map(p => p.value ?? p.pred ?? null);
  _renderPredSummary(labels, values);
}
function _renderPredSummaryFromServer(s){
  const box = document.getElementById('predSummary');
  if (!box) return;
  box.style.removeProperty('display');

  const fmt = n => (n==null)? '-' : 'Rp '+new Intl.NumberFormat('id-ID').format(Math.round(n));
  const fmtPct = n => (n==null)? '-' : (n>=0? '+' : '') + Number(n).toFixed(2) + '%';

  document.getElementById('predMaxVal').textContent  = fmt(s.max);
  document.getElementById('predMaxDate').textContent = s.max_date || '—';
  document.getElementById('predMinVal').textContent  = fmt(s.min);
  document.getElementById('predMinDate').textContent = s.min_date || '—';
  document.getElementById('predAvgVal').textContent  = fmt(s.avg);
  document.getElementById('predCount').textContent   = `n = ${s.n||0} hari`;
  document.getElementById('predChangePct').textContent = fmtPct(s.change_pct);
  document.getElementById('predChangeNote').textContent =
    (s.start!=null && s.end!=null) ? `${fmt(s.start)} → ${fmt(s.end)}` : '—';
}
async function fetchPredictRange(citySlug, startISO, endISO) {
  const params = new URLSearchParams({
    city: citySlug,
    start: startISO,
    end: endISO,
    future_only: '0',
    hide_actual: '0',
    naive_fallback: '1',
    viz_roll: String(VIZ_ROLL)   // <-- DULUNYA '0'
  });
  const r = await fetch(`/api/predict_range?${params.toString()}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

(function dropdownKabupaten(){
  const btn    = document.getElementById('kabupatenBtn');
  const panel  = document.getElementById('kabupatenPanel');
  const inp    = document.getElementById('kabupatenSearch');
  const listEl = document.getElementById('kabupatenList');
  const metaEl = document.getElementById('kabupatenMeta');
  const hidden = document.getElementById('kabupaten_a');

  if (!btn || !panel || !inp || !listEl || !hidden) return;

  // --- 1) Ambil & cache data (sekali saja)
  let cities = window.__CITIES_CACHE || null;
  let lastRender = []; // untuk diff minimal
  const MAX_RENDER = 200;          // batasi render supaya lincah
  const DEBOUNCE_MS = 120;         // filter debounce

  const norm = s => (s||'').toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g,'');
  const slugify = s => norm(s).replace(/\s+/g,'_').replace(/[^\w_]/g,'');

  async function loadCities(){
    if (cities) return cities;
    try{
      const r = await fetch('/api/cities_full', {cache:'no-store'});
      const arr = await r.json();
      cities = arr.map(x=>({
        label: x.label || x.name,
        slug : x.slug  || slugify(x.label || x.name),
        prov : x.province || x.prov || '',
        island: x.island || ''
      }));
    }catch{
      const r2 = await fetch('/api/cities', {cache:'no-store'});
      const arr2 = await r2.json();
      cities = arr2.map(l=>({ label:l, slug:slugify(l), prov:'', island:'' }));
    }
    // Precompute utk cepat cari
    cities.forEach(c=>{ c._l = norm(c.label); });
    window.__CITIES_CACHE = cities;
    return cities;
  }

  // --- 2) Filter cepat: prioritas startsWith, lalu includes; limit MAX_RENDER
  function filterCities(q){
    if (!q) return cities.slice(0, MAX_RENDER);
    const n = norm(q);
    const starts = [];
    const contains = [];
    for (const c of cities){
      const i = c._l.indexOf(n);
      if (i === 0) starts.push(c);
      else if (i > 0) contains.push(c);
      if (starts.length + contains.length >= MAX_RENDER) break;
    }
    return starts.concat(contains).slice(0, MAX_RENDER);
  }

  // --- 3) Render hemat (pakai DocumentFragment, no reflow besar)
  function render(items){
    // quick bail: kalau list sama (panjang & slug sama), jangan render ulang
    if (items.length === lastRender.length && items.every((it, i)=> it.slug === lastRender[i].slug)) return;

    const frag = document.createDocumentFragment();
    for (const c of items){
      const div = document.createElement('div');
      div.className = 'dropdown-item';
      div.setAttribute('role', 'option');
      div.dataset.slug = c.slug;
      div.innerHTML = `<span>${c.label}</span>${c.prov ? `<span class="sub">${c.prov}</span>` : ''}`;
      frag.appendChild(div);
    }
    listEl.innerHTML = '';
    listEl.appendChild(frag);
    metaEl.textContent = `${items.length} hasil`;
    lastRender = items;
  }

  // --- 4) Open/close tanpa kedip
  let insidePointer = false;
  function openPanel(){
    if (panel.style.display === 'block') return;
    panel.style.display = 'block';
    btn.setAttribute('aria-expanded','true');
    // fokus setelah terbuka
    setTimeout(()=> inp.focus(), 0);
  }
  function closePanel(){
    if (panel.style.display === 'none') return;
    panel.style.display = 'none';
    btn.setAttribute('aria-expanded','false');
  }

  btn.addEventListener('click', async ()=>{
    if (panel.style.display === 'block'){ closePanel(); return; }
    await loadCities();
    render(cities.slice(0, MAX_RENDER));
    openPanel();
  });

  // cegah close saat scroll/klik di dalam panel
  panel.addEventListener('pointerdown', ()=>{ insidePointer = true; }, {passive:true});
  document.addEventListener('pointerdown', (e)=>{
    if (insidePointer){ insidePointer = false; return; }
    if (!panel.contains(e.target) && e.target !== btn) closePanel();
  });

  // --- 5) Debounced search (supaya gak “bergetar”)
  let t = null;
  inp.addEventListener('input', ()=>{
    clearTimeout(t);
    const q = inp.value.trim();
    t = setTimeout(()=>{
      const items = filterCities(q);
      render(items);
    }, DEBOUNCE_MS);
  });

  // --- 6) Pilih item (pakai mousedown supaya gak kehilangan fokus saat click)
  listEl.addEventListener('mousedown', (e)=>{
    const item = e.target.closest('.dropdown-item');
    if (!item) return;
    e.preventDefault(); // biar gak trigger blur/close duluan
    const slug = item.dataset.slug;
    const label = item.querySelector('span')?.textContent || item.textContent;
    hidden.value = slug;
    btn.textContent = label;
    closePanel();
  });

  // --- 7) Keyboard nav (↑/↓/Enter/Esc)
  let activeIdx = -1;
  function highlight(idx){
    const nodes = listEl.querySelectorAll('.dropdown-item');
    nodes.forEach(n=> n.removeAttribute('aria-selected'));
    if (idx >=0 && idx < nodes.length){
      nodes[idx].setAttribute('aria-selected','true');
      nodes[idx].scrollIntoView({block:'nearest'});
    }
  }
  inp.addEventListener('keydown', (e)=>{
    const nodes = listEl.querySelectorAll('.dropdown-item');
    if (e.key === 'ArrowDown'){ e.preventDefault(); activeIdx = Math.min(nodes.length-1, activeIdx+1); highlight(activeIdx); }
    else if (e.key === 'ArrowUp'){ e.preventDefault(); activeIdx = Math.max(0, activeIdx-1); highlight(activeIdx); }
    else if (e.key === 'Enter'){ e.preventDefault(); if (activeIdx>=0 && nodes[activeIdx]) nodes[activeIdx].dispatchEvent(new MouseEvent('mousedown')); }
    else if (e.key === 'Escape'){ closePanel(); btn.focus(); }
  });

  // preload ringan (biar cepat saat pertama kali klik)
  loadCities();
})();

async function loadPrediksi(){
  // fallback ke #kabupaten kalau #kabupaten_a tidak ada
  console.log("loadPrediksi called");
  // const sel = document.getElementById('citySearch') || document.getElementById('kabupaten');
  const hidden = document.getElementById('kabupaten_a');
  const citySlug =hidden ? hidden.value : '';
  const start = document.getElementById('startDate').value;
  const end   = document.getElementById('endDate').value;
  const ph = document.getElementById('predPlaceholder');

  if (!citySlug) { alert('Pilih Kabupaten/Kota.'); return; }
  if (!start || !end) { alert('Tanggal mulai & akhir wajib diisi.'); return; }

  try {
    ph.style.display = 'none';

    const r = await fetchPredictRange(citySlug, start, end);
    const labels = fullDateRange(start, end);

    // r.actual & r.predicted: [{date:'YYYY-MM-DD', value:number}, ...]
    const mapActual = new Map((r.actual || []).map(p => [p.date, p.value]));
    const mapPred   = new Map((r.predicted || []).map(p => [p.date, p.value]));

    const dsActual  = labels.map(d => mapActual.has(d) ? mapActual.get(d) : null);
    const dsPred    = labels.map(d => mapPred.has(d)   ? mapPred.get(d)   : null);

    const cityLabel = (r.entity || r.city || citySlug).replace(/_/g,' ');
    renderPredChart(labels, dsActual, dsPred, cityLabel);

    // --- RINGKASAN ---
    // 1) kalau server sudah kirim summary, pakai itu
    if (r.summary && r.summary.predicted){
      _renderPredSummaryFromServer(r.summary.predicted);
    } else {
      // 2) kalau mapping ke labels kosong/null semua, hitung langsung dari payload r.predicted
      const nonNullCount = dsPred.filter(v => v != null && !Number.isNaN(v)).length;
      if (nonNullCount === 0 && Array.isArray(r.predicted) && r.predicted.length){
        _renderPredSummaryFromAPI(r.predicted);
      } else {
        _renderPredSummary(labels, dsPred);
      }
    }

    document.querySelector('.nav-link[data-section="prediksi"]')?.click();
    await fetchAndRenderEvalForCity(citySlug);

  } catch (e) {
  console.error(e);
  ph.style.display = '';
  ph.innerHTML = `❌ Gagal memuat prediksi: <small>${e.message}</small>`;
}
}
window.loadPrediksi = loadPrediksi;


async function loadEvaluasi(){
  const citySel = document.getElementById('ev_city');
  const city    = selectedSlugOrLabel(citySel);
  const gran    = document.getElementById('ev_gran').value || 'weekly';
  const ph      = document.getElementById('evPlaceholder');

  if(!city){ alert('Pilih Kabupaten/Kota.'); return; }

  // tentukan rentang otomatis: N hari terakhir
  const endDate   = new Date();
  const startDate = new Date(); startDate.setDate(endDate.getDate() - EVAL_DEFAULT_DAYS);
  const toISO = d => d.toISOString().slice(0,10);
  const start = toISO(startDate);
  const end   = toISO(endDate);

  try{
    ph.style.display='none';

    // 1) metrik dari Excel (via API /api/eval_summary)
    const resSummary = await fetch(`/api/eval_summary?city=${encodeURIComponent(city)}`);
    let cityLabel = city;
    if (resSummary.ok) {
      const js = await resSummary.json();
      cityLabel = (js.city || city).replace(/_/g,' ');
      const m = js.metrics || {};
      const fmt = (x,dec=2)=> (x==null || Number.isNaN(x))? '-' : (dec==0? Math.round(x).toString() : x.toFixed(dec));
      document.getElementById('evMAE').textContent  = m.mae  != null ? fmtNum(m.mae, 0) : '-';
      document.getElementById('evMAPE').textContent = m.mape != null ? fmtNum(m.mape * 100, 2) : '-';
      document.getElementById('evMSE').textContent  = m.mse  != null ? fmtNum(m.mse, 0) : '-';
      document.getElementById('evR2').textContent   = m.r2   != null ? fmtNum(m.r2, 3) : '-';
    } else {
      // fallback kosong
      document.getElementById('evMAE').textContent  = '-';
      document.getElementById('evMAPE').textContent = '-';
      document.getElementById('evMSE').textContent  = '-';
      document.getElementById('evR2').textContent   = '-';
    }

    // 2) grafik: ambil aktual vs prediksi untuk rentang default
    const r = await fetchPredictRange(city, start, end);
    const labelsDaily = fullDateRange(start, end);

    const mapAct = new Map((r.actual||[]).map(p=>[p.date, p.value]));
    const mapPred= new Map((r.predicted||[]).map(p=>[p.date, p.value]));
    const dailyActual = labelsDaily.map(d => mapAct.has(d)? mapAct.get(d): null);
    const dailyPred   = labelsDaily.map(d => mapPred.has(d)? mapPred.get(d): null);

    const dailyActualPoints = labelsDaily.map((d,i)=> ({date:d, value: dailyActual[i]})).filter(p=>p.value!=null);
    const dailyPredPoints   = labelsDaily.map((d,i)=> ({date:d, value: dailyPred[i]})).filter(p=>p.value!=null);

    const aggAct = aggregate(dailyActualPoints, gran);
    const aggPred= aggregate(dailyPredPoints, gran);

    const unionLabels = Array.from(new Set([...aggAct.labels, ...aggPred.labels])).sort();
    const mA = new Map(aggAct.labels.map((l,i)=>[l, aggAct.values[i]]));
    const mP = new Map(aggPred.labels.map((l,i)=>[l, aggPred.values[i]]));
    const seriesA = unionLabels.map(l => (mA.has(l)? mA.get(l) : null));
    const seriesP = unionLabels.map(l => (mP.has(l)? mP.get(l) : null));

    const prettyCity = (r.entity||r.city||cityLabel).replace(/_/g,' ');
    const gText = gran==='weekly'?'Mingguan': (gran==='monthly'?'Bulanan':'Harian');
    const title = `Aktual vs Prediksi • ${prettyCity} • ${gText} • ${EVAL_DEFAULT_DAYS} hari terakhir`;
    renderEvaluasiChart(unionLabels, seriesA, seriesP, title);

    document.querySelector('.nav-link[data-section="evaluasi"]')?.click();
  }catch(e){
    console.error(e);
    ph.style.display='';
    ph.innerHTML = `❌ Gagal memuat evaluasi: <small>${e.message}</small>`;
  }
};

// ===== Default tanggal Prediksi (opsional 30 hari terakhir)
document.addEventListener('DOMContentLoaded', () => {
  const end = new Date();
  const start = new Date(); start.setDate(end.getDate() - 30);
  const toISO = d => d.toISOString().slice(0,10);

  const sEl=document.getElementById('startDate'); 
  const eEl=document.getElementById('endDate');
  if (sEl && eEl && !sEl.value && !eEl.value) {
    sEl.value = toISO(start); eEl.value = toISO(end);
  }

  // evaluasi: default 90 hari biar kurva lebih halus
  const es=document.getElementById('ev_start');
  const ee=document.getElementById('ev_end');
  if (es && ee && !es.value && !ee.value) {
    const evStart = new Date(); evStart.setDate(end.getDate() - 90);
    es.value = toISO(evStart); ee.value = toISO(end);
  }
});

function getWeekKey(dateStr){
  const d = new Date(dateStr+"T00:00:00");
  // ISO-like (Senin awal pekan)
  const day = (d.getDay()+6)%7; // 0=Senin
  const monday = new Date(d); monday.setDate(d.getDate()-day);
  const y = monday.getFullYear();
  const oneJan = new Date(y,0,1);
  // Perkiraan nomor pekan
  const week = Math.floor(((monday - oneJan)/(24*3600*1000) + (oneJan.getDay()+6)%7)/7)+1;
  return `${y}-W${String(week).padStart(2,'0')}`;
}
function getMonthKey(dateStr){
  const d=new Date(dateStr+"T00:00:00");
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}`;
}
// points: [{date:'YYYY-MM-DD', value:number}]
function aggregate(points, mode){
  if(mode==='daily') {
    const labels = points.map(p=>p.date);
    const values = points.map(p=>p.value);
    return {labels, values};
  }
  const groups = new Map();
  for(const p of points){
    if(p.value==null || Number.isNaN(p.value)) continue;
    const key = mode==='monthly' ? getMonthKey(p.date) : getWeekKey(p.date);
    const ent = groups.get(key) || {sum:0, n:0};
    ent.sum += p.value; ent.n += 1;
    groups.set(key, ent);
  }
  const labels = Array.from(groups.keys()).sort();
  const values = labels.map(k => groups.get(k).sum / groups.get(k).n);
  return {labels, values};
}

let _evChart=null;
function renderEvaluasiChart(labels, actualAvg, predAvg, title){
  const canvas=document.getElementById('evChart'); if(!canvas) return;
  const ctx=canvas.getContext('2d');
  if(_evChart) _evChart.destroy();
  _evChart = new Chart(ctx, {
    type:'line',
    data:{
      labels,
      datasets:[
        { label:'Aktual (avg)',   data: actualAvg, borderWidth:2, tension:.25, pointRadius:0 },
        { label:'Prediksi (avg)', data: predAvg,   borderWidth:2, tension:.25, pointRadius:0 }
      ]
    },
    options:{
      responsive:true,
      interaction:{mode:'index',intersect:false},
      plugins:{ title:{display:true, text: title || 'Aktual vs Prediksi (rata-rata)'} },
      scales:{ y:{ ticks:{ callback:v=>'Rp '+rupiah(v) } } }
    }
  });
}


// boot
initCities();

/* 1) Animated ink underline for nav */
(function navInk(){
  const nav = document.querySelector('.top-nav');
  if (!nav) return;
  let ink = nav.querySelector('.ink');
  if (!ink){
    ink = document.createElement('span');
    ink.className = 'ink';
    nav.appendChild(ink);
  }
  const moveInk = (el)=>{
    const rect = el.getBoundingClientRect();
    const parent = nav.getBoundingClientRect();
    ink.style.width = rect.width + 'px';
    ink.style.left  = (rect.left - parent.left) + 'px';
  };
  const active = nav.querySelector('.nav-link.active') || nav.querySelector('.nav-link');
  if (active) moveInk(active);
  nav.addEventListener('click', (e)=>{
    const a = e.target.closest('.nav-link');
    if (!a) return;
    requestAnimationFrame(()=> moveInk(a));
  });
  window.addEventListener('resize', ()=>{
    const current = nav.querySelector('.nav-link.active');
    if (current) moveInk(current);
  });
})();

/* 2) Auto Lite Mode (ringankan animasi di device low-spec / prefers-reduced-motion) */
(function enableLiteMode(){
  try{
    const dm = navigator.deviceMemory || 0;
    const cores = navigator.hardwareConcurrency || 0;
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const lowMem = dm && dm <= 4;
    const lowCore = cores && cores <= 4;
    if (prefersReduced || lowMem || lowCore){
      document.documentElement.classList.add('lite');
    }
  }catch(_){}
})();

/* 3) Scroll Reveal (lean) */
(function scrollRevealLean(){
  if (!('IntersectionObserver' in window)) {
    document.querySelectorAll('.content-section,.chart-container,#map,.stat-card')
      .forEach(el=> el.classList.add('is-visible'));
    return;
  }
  const lite = document.documentElement.classList.contains('lite');
  const observer = new IntersectionObserver((entries, obs)=>{
    for (const ent of entries){
      if (ent.isIntersecting){
        ent.target.classList.add('is-visible');
        obs.unobserve(ent.target);
      }
    }
  }, { rootMargin: lite ? '0px 0px -20% 0px' : '0px 0px -10% 0px', threshold: 0.06 });

  document.querySelectorAll('.content-section, .chart-container, #map')
    .forEach(el=> observer.observe(el));
})();

/* 4) Parallax hero (disable on lite) */
(function heroParallaxLite(){
  const htmlLite = document.documentElement.classList.contains('lite');
  const hero = document.querySelector('.hero');
  if (!hero || htmlLite) return;
  let ticking = false;
  const onScroll = ()=>{
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(()=>{
      const sy = window.scrollY || window.pageYOffset;
      const t = Math.min(30, sy * 0.06);
      hero.style.transform = `translateY(${t}px)`;
      ticking = false;
    });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
})();

/* 5) Ripple ringan (disabled on lite) */
(function rippleButtonsLite(){
  const lite = document.documentElement.classList.contains('lite');
  if (lite) return; // disable ripple in lite
  document.querySelectorAll('.btn, .btn-telusuri').forEach(btn=>{
    btn.classList.add('btn--ripple');
    let t;
    btn.addEventListener('pointerdown', (e)=>{
      const rect = btn.getBoundingClientRect();
      const rx = ((e.clientX - rect.left)/rect.width)*100;
      const ry = ((e.clientY - rect.top)/rect.height)*100;
      btn.style.setProperty('--rx', rx + '%');
      btn.style.setProperty('--ry', ry + '%');
      btn.classList.add('is-pressed');
      clearTimeout(t);
      t = setTimeout(()=> btn.classList.remove('is-pressed'), 180);
    });
  });
})();

/* 6) Chart.js default animation (respect lite) */
(function chartDefaults(){
  if (!window.Chart) return;
  const lite = document.documentElement.classList.contains('lite');
  Chart.defaults.animation.duration = lite ? 300 : 900;
  Chart.defaults.animation.easing = 'easeOutQuart';
  Chart.defaults.elements.line.tension = 0.25;
  Chart.defaults.elements.point.radius = 0;
})();

/* 7) Scroll tombol Telusuri tetap bekerja */

  // ================= TOP-N dari Excel =================
function fmtID(x, dec=0){
  if (x==null || Number.isNaN(x)) return '-';
  return new Intl.NumberFormat('id-ID', {maximumFractionDigits: dec, minimumFractionDigits: dec}).format(x);
}

async function loadTopN(){
  const tahun = document.getElementById('b_top_tahun')?.value || '';
  const order = document.getElementById('b_top_order')?.value || 'desc';
  const limit = document.getElementById('b_top_limit')?.value || '5';
  const spinner = document.getElementById('top5_loading');
  const tbody = document.querySelector('#top5Table tbody');
  if (!tbody) return;

  if (!tahun){
    alert('Pilih tahun dulu.'); 
    return;
  }

  spinner && (spinner.style.display = 'inline-block');
  try{
    const url = `/api/top_cities?year=${encodeURIComponent(tahun)}&order=${encodeURIComponent(order)}&limit=${encodeURIComponent(limit)}`;
    const res = await fetch(url, {cache:'no-store'});
    const js  = await res.json();
    if (!res.ok) throw new Error(js.error || 'Server error');

    // render table
    const rows = js.data || [];
    if (!rows.length){
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--muted)">Tidak ada data untuk tahun ${tahun}.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td style="text-align:center">${r.rank ?? ''}</td>
        <td>${r.city || '-'}</td>
        <td>${r.province || '-'}</td>
        <td style="text-align:right">Rp ${fmtID(r.avg,0)}</td>
        <td style="text-align:right">${r.min==null?'-':('Rp '+fmtID(r.min,0))}</td>
        <td style="text-align:right">${r.max==null?'-':('Rp '+fmtID(r.max,0))}</td>
        <td style="text-align:right">${r.n==null?'-':fmtID(r.n,0)}</td>
      </tr>
    `).join('');
  }catch(e){
    console.error(e);
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#b00020">Gagal memuat: ${e.message}</td></tr>`;
  }finally{
    spinner && (spinner.style.display = 'none');
  }
}

// sinkron default tahun (opsional) & bind tombol
document.addEventListener('DOMContentLoaded', ()=>{
  const tMaster = document.getElementById('b_tahun');
  const tTop    = document.getElementById('b_top_tahun');
  if (tMaster && tTop && !tTop.value && tMaster.value){
    tTop.value = tMaster.value;
  }
  document.getElementById('btnTop5')?.addEventListener('click', loadTopN);
});


// ================= Regional Correlation (Pulau/Provinsi) =================
document.addEventListener('DOMContentLoaded', () => {
  const rcModeSel   = document.getElementById('rc_mode');    // 'island' | 'province'
  const rcValueSel  = document.getElementById('rc_value');   // dropdown dinamis
  const rcBtn       = document.getElementById('rc_btn');     // tombol Tampilkan (opsional untuk isi dropdown)
  const rcSpinner   = document.getElementById('rc_loading');
  const rcTableBody = document.querySelector('#rc_table tbody');

  // kalau dropdown utamanya nggak ada, sudahi (view lain)
  if (!rcModeSel || !rcValueSel) return;

  // util kecil
  
  const MONTHS_ID_SHORT = ["","Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"];
  const fmtIDnum = (x, dec = 0) =>
    (x == null || Number.isNaN(x))
      ? '-'
      : new Intl.NumberFormat('id-ID', { maximumFractionDigits: dec, minimumFractionDigits: dec }).format(x);

  const showSpin = (yes) => { if (rcSpinner) rcSpinner.style.display = yes ? 'inline-block' : 'none'; };
  const rcRenderEmpty = (msg) => {
    if (!rcTableBody) return;
    rcTableBody.innerHTML = `
      <tr>
        <td colspan="12" style="text-align:center;color:var(--muted)">${msg || 'Tidak ada data.'}</td>
      </tr>
    `;
  };
  const rcPeriod = (m, y) => {
    if (m == null || y == null) return '—';
    const mm = Number(m) || 0, yy = Number(y) || 0;
    if (!mm || !yy) return '—';
    return `${MONTHS_ID_SHORT[mm] || mm}-${yy}`;
  };
  const rcDatePretty = (iso) => {
    if (!iso) return '—';
    try { return new Date(iso+"T00:00:00").toLocaleDateString('id-ID',{day:'2-digit',month:'short',year:'numeric'}); }
    catch { return iso; }
  };

  async function loadIslandsList() {
    try {
      const r = await fetch('/api/islands', { cache: 'no-store' });
      if (!r.ok) throw 0;
      const arr = await r.json();
      return (arr || []).filter(n => n && n.toLowerCase() !== 'semua pulau');
    } catch {
      return ["Jawa","Sumatra","Kalimantan","Sulawesi","Bali–NT","Maluku","Papua"];
    }
  }
  async function loadProvincesList() {
    try {
      const r = await fetch('/api/provinces', { cache: 'no-store' });
      if (!r.ok) throw 0;
      return await r.json();
    } catch {
      return [
        "Aceh","Sumatera Utara","Sumatera Barat","Riau","Kepulauan Riau","Jambi",
        "Sumatera Selatan","Kepulauan Bangka Belitung","Bengkulu","Lampung",
        "Banten","DKI Jakarta","Jawa Barat","Jawa Tengah","DI Yogyakarta","Jawa Timur",
        "Bali","Nusa Tenggara Barat","Nusa Tenggara Timur",
        "Kalimantan Barat","Kalimantan Tengah","Kalimantan Selatan","Kalimantan Timur","Kalimantan Utara",
        "Sulawesi Utara","Gorontalo","Sulawesi Tengah","Sulawesi Barat","Sulawesi Selatan","Sulawesi Tenggara",
        "Maluku","Maluku Utara","Papua","Papua Barat","Papua Barat Daya","Papua Tengah","Papua Pegunungan","Papua Selatan"
      ];
    }
  }

  async function fillRcValueOptions() {
    rcValueSel.innerHTML = '<option value="">-- Pilih --</option>';
    const mode = rcModeSel.value; // 'island' | 'province'
    const list = (mode === 'island') ? await loadIslandsList() : await loadProvincesList();
    for (const name of list) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      rcValueSel.appendChild(opt);
    }
    // log kecil biar kelihatan
    console.log('[RC] filled', mode, '=>', rcValueSel.options.length, 'opsi');
  }

  async function fetchRegionSummary(mode, value) {
    const u = `/api/region_summary?mode=${encodeURIComponent(mode)}&value=${encodeURIComponent(value)}`;
    const r = await fetch(u, { cache: 'no-store' });
    if (!r.ok) {
      const js = await r.json().catch(() => ({}));
      throw new Error(js.error || 'Server error');
    }
    return r.json();
  }

  function renderRegionRows(rows) {
    if (!rcTableBody) return;
    if (!rows || !rows.length) {
      rcRenderEmpty('Tidak ada data untuk pilihan ini.');
      return;
    }
    rcTableBody.innerHTML = rows.map((r, i) => `
      <tr>
        <td style="text-align:center">${r.no ?? (i + 1)}</td>
        <td>${r.city || '-'}</td>
        <td>${r.province || '-'}</td>
        <td>${r.island || '-'}</td>
        <td class="num">Rp ${fmtIDnum(r.min_value, 0)}</td>
        <td class="num dim">${rcDatePretty(r.min_date)}</td>
        <td class="num">Rp ${fmtIDnum(r.max_value, 0)}</td>
        <td class="num dim">${rcDatePretty(r.max_date)}</td>
        <td class="num">Rp ${fmtIDnum(r.avg_month_high, 0)}</td>
        <td class="num dim">${rcPeriod(r.avg_month_high_month, r.avg_month_high_year)}</td>
        <td class="num">Rp ${fmtIDnum(r.avg_month_low, 0)}</td>
        <td class="num dim">${rcPeriod(r.avg_month_low_month, r.avg_month_low_year)}</td>
      </tr>
    `).join('');
  }

  async function runRegional() {
    const mode  = rcModeSel.value || 'island';
    const value = rcValueSel.value || '';
    if (!value) { alert('Pilih nilai pada dropdown kedua.'); return; }
    showSpin(true);
    try {
      const js = await fetchRegionSummary(mode, value);
      renderRegionRows(js.rows || []);
    } catch (e) {
      console.error(e);
      rcRenderEmpty(`Gagal memuat: ${e.message}`);
    } finally {
      showSpin(false);
    }
  }

  // wiring
  rcModeSel.addEventListener('change', fillRcValueOptions);
  rcBtn && rcBtn.addEventListener('click', runRegional);

  // init pertama kali (pasti dipanggil setelah DOM siap)
  fillRcValueOptions();
});

function updateEvalCards(metrics) {
  const fmt0 = x => (x==null || Number.isNaN(x)) ? '-' : new Intl.NumberFormat('id-ID').format(Math.round(x));
  const fmt2 = x => (x==null || Number.isNaN(x)) ? '-' : Number(x).toFixed(2);

  const mse  = metrics?.mse ?? null;
  const rmse = metrics?.rmse ?? (mse!=null ? Math.sqrt(mse) : null);
  const mape = metrics?.mape != null ? (metrics.mape * 100) : null; // asumsikan server kirim 0..1
  const r2   = metrics?.r2 ?? null;

  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('mseValue',  mse==null ? '-' : `Rp ${fmt0(mse)}`);
  set('rmseValue', rmse==null ? '-' : `Rp ${fmt0(rmse)}`);
  set('mapeValue', fmt2(mape));
  set('r2Value',   r2==null ? '-' : Number(r2).toFixed(3));

  // kalau pakai grade:
  // const grade = r2==null ? '-' : (r2>=0.90?'A':r2>=0.85?'A-':r2>=0.80?'B+':r2>=0.75?'B':r2>=0.70?'B-':'C');
  // set('performanceGrade', grade);
}

async function fetchAndRenderEvalForCity(city){
  try{
    const res = await fetch(`/api/eval_summary?city=${encodeURIComponent(city)}`);
    if (!res.ok) throw 0;
    const js = await res.json();
    updateEvalCards(js?.metrics || {});
  } catch {
    updateEvalCards({});
  }
}

async function loadUpload() {
  console.log("loadUpload called");
  const fileInput = document.getElementById('fileInput');
  const uploadPrompt = document.getElementById('uploadPrompt');
  const uploadSuccess = document.getElementById('uploadSuccess');
  const uploadResults = document.getElementById('uploadResults');
  const ph = document.getElementById('uploadPlaceholder'); // placeholder error/loading (opsional)
  const saveBtn = document.getElementById('saveUploadBtn');

  // Reset UI state
  if (ph) ph.style.display = 'none';
  if (uploadPrompt) uploadPrompt.style.display = 'block';
  if (uploadSuccess) uploadSuccess.style.display = 'none';
  if (uploadResults) uploadResults.style.display = 'none';
  if (saveBtn) saveBtn.disabled = true;

  // If a file already selected (page refresh / revisit), tampilkan detail & enable save
  if (fileInput && fileInput.files && fileInput.files[0]) {
    const file = fileInput.files[0];
    if (uploadPrompt) uploadPrompt.style.display = 'none';
    if (uploadSuccess) {
      uploadSuccess.style.display = 'block';
      const fn = document.getElementById('fileName');
      const fs = document.getElementById('fileSize');
      if (fn) fn.textContent = file.name;
      if (fs) fs.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    }
    if (saveBtn) saveBtn.disabled = false;
    // note: jangan auto-proses kecuali memang diinginkan
  }

  // **JANGAN** memanggil navLink.click() di sini — navigasi (menunjukkan section)
  // **ditangani oleh delegated nav handler**. Jika kamu ingin memastikan section terlihat
  // ketika fungsi ini dipanggil dari tempat lain, panggil showSectionById('upload')
  // atau simpan logika showSection di satu tempat.
}

window.loadUpload = loadUpload;
/* =========================
   Dropdown Search Optimized
   ========================= */
// dipanggil oleh <input type="file" onchange="handleFileSelect(event)">
function handleFileSelect(evt) {
  const file = evt.target.files[0];
  if (!file) return;
  const uploadPrompt = document.getElementById('uploadPrompt');
  const uploadSuccess = document.getElementById('uploadSuccess');
  const fn = document.getElementById('fileName');
  const fs = document.getElementById('fileSize');
  if (uploadPrompt) uploadPrompt.style.display = 'none';
  if (uploadSuccess) uploadSuccess.style.display = 'block';
  if (fn) fn.textContent = file.name;
  if (fs) fs.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  document.getElementById('saveUploadBtn').disabled = false;
}

// simple helpers for downloading templates (kamu bisa generate file server-side too)
function downloadTemplateCSV() {
  const csv = "date,City A,City B\n2023-01-01,0,0\n";
  const blob = new Blob([csv], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'template_prices.csv';
  a.click();
}
function downloadTemplateExcel() {
  // buat placeholder: arahkan ke template di server jika ada. Untuk demo, fallback ke CSV:
  downloadTemplateCSV();
}

// utama: kirim file ke backend
// GANTI fungsi saveAndPredict dengan ini
// --- replace saveAndPredict with this defensive version ---
async function saveAndPredict() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput || !fileInput.files || !fileInput.files[0]) return alert('Pilih file dulu');
  const file = fileInput.files[0];
  const saveBtn = document.getElementById('saveUploadBtn');
  const loading = document.getElementById('uploadLoading');

  saveBtn.disabled = true;
  loading.style.display = 'inline-block';
  loading.textContent = 'Memproses... (ini bisa lama jika mode full)';

  try {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('mode', 'full'); // ubah ke 'quick' bila ingin cepat

    const resp = await fetchJsonSafe('/api/upload_file', {
      method: 'POST',
      body: fd
    });

    console.log('upload response (safe):', resp);

    // Jika status non-OK, tampilkan detail (server bisa ngirim 400 dengan JSON)
    if (!resp.ok) {
      const reason = resp.data ? JSON.stringify(resp.data) : (resp.text || `HTTP ${resp.status}`);
      throw new Error(reason);
    }

    // Ambil parsed JSON kalau ada; kalau tidak ada fallback ke text (dan coba parse)
    let data = resp.data;
    if (!data && resp.text) {
      try { data = JSON.parse(resp.text); }
      catch (e) { /* tetap null */ }
    }

    if (!data) {
      console.warn('Server returned no JSON body; raw text:', resp.text);
      throw new Error('Server memberikan response tanpa JSON. Periksa Network tab / server log.');
    }

    console.log('upload data parsed:', data);

    // HANDLE QUICK
    if (data.mode === 'quick' || data.stats) {
      renderResults(data);
      return;
    }

    // HANDLE FULL (server kamu mengembalikan single-city 'full' object)
    if (data.mode === 'full') {
      // render same UI as quick, plus show pack/metrics etc
      renderResults(data);

      // optional: show detailed results list if response included `results` array
      const ul = document.getElementById('uploadFullResultsList');
      if (ul && Array.isArray(data.results)) {
        ul.innerHTML = data.results.map(r => {
          if (r.ok) return `<li><b>${r.city}</b>: OK — best_r2=${r.best_r2 ?? '-'}</li>`;
          return `<li><b>${r.city}</b>: FAIL — ${r.reason ?? JSON.stringify(r)}</li>`;
        }).join('');
        document.getElementById('uploadFullSummary')?.style.removeProperty('display');
      }

      // show uploadResults section
      document.getElementById('uploadResults')?.style.removeProperty('display');
      return;
    }

    // fallback: jika ada stats/predictions di bentuk lain
    if (data.stats || data.predictions || data.trend || data.pred_series) {
      renderResults(data);
      return;
    }

    console.warn('Unexpected upload response shape', data);
    alert('Server returned unexpected response. Check console / Network.');

  } catch (err) {
    console.error(err);
    alert('Terjadi error: ' + (err.message || err));
  } finally {
    loading.style.display = 'none';
    saveBtn.disabled = false;
  }
}



// render hasil: update stats dan chart
// global chart holders (safe)
window._uploadTrendChart = window._uploadTrendChart || null;
window._uploadPredChart  = window._uploadPredChart  || null;

function fmtID(n){
  if (n == null || Number.isNaN(n)) return '-';
  return 'Rp ' + new Intl.NumberFormat('id-ID', {maximumFractionDigits:0}).format(Math.round(n));
}

function renderResults(payload){
  // payload bisa shape quick OR full
  const container = document.getElementById('uploadResults');
  if (!container) {
    console.warn('renderResults: #uploadResults tidak ditemukan');
    return;
  }
  // show container
  container.style.removeProperty('display');

  // --- stats: support payload.stats OR payload.metrics / payload.n_total ---
  const stats = payload.stats || payload.metrics || {};
  const nPoints = stats.n_points ?? stats.n ?? payload.n_total ?? '-';
  const avg = stats.avg ?? stats.mean ?? '-';
  const min = stats.min ?? '-';
  const max = stats.max ?? '-';

  const elN = document.getElementById('uploadDataPoints');
  const elAvg = document.getElementById('uploadAvgPrice');
  const elMin = document.getElementById('uploadMinPrice');
  const elMax = document.getElementById('uploadMaxPrice');

  if (elN) elN.textContent = (nPoints === '-' ? '-' : String(nPoints));
  if (elAvg) elAvg.textContent = (avg === '-' ? '-' : fmtID(avg));
  if (elMin) elMin.textContent = (min === '-' ? '-' : fmtID(min));
  if (elMax) elMax.textContent = (max === '-' ? '-' : fmtID(max));

  // --- predictions: accept keys '1','7','10','30' safe ---
  const preds = payload.predictions || payload.predictions_full || {};
  const pick = (k)=> {
    if (!preds) return null;
    if (preds[k]) return preds[k];
    // sometimes backend returns numbers => try numeric key
    return preds[String(k)] || null;
  };
  const p1 = pick('1') || pick(1) || null;
  const p7 = pick('7') || pick(7) || null;
  const p30 = pick('30') || pick(30) || pick('10') || pick(10) || null; // be permissive

  if (document.getElementById('pred1Day')) document.getElementById('pred1Day').textContent = p1?.value == null ? '-' : fmtID(p1.value);
  if (document.getElementById('pred1DayDate')) document.getElementById('pred1DayDate').textContent = p1?.date ?? '—';
  if (document.getElementById('pred7Day')) document.getElementById('pred7Day').textContent = p7?.value == null ? '-' : fmtID(p7.value);
  if (document.getElementById('pred7DayDate')) document.getElementById('pred7DayDate').textContent = p7?.date ?? '—';
  if (document.getElementById('pred30Day')) document.getElementById('pred30Day').textContent = p30?.value == null ? '-' : fmtID(p30.value);
  if (document.getElementById('pred30DayDate')) document.getElementById('pred30DayDate').textContent = p30?.date ?? '—';

  // --- trend chart (actual series); support payload.trend (dates/values) OR payload.trend_series shape ---
  const trend = payload.trend || (payload.trend_series && { dates: payload.trend_series.dates, values: payload.trend_series.actual }) || {dates: [], values: []};
  try {
    const ctx = document.getElementById('uploadTrendChart')?.getContext?.('2d');
    if (ctx) {
      if (window._uploadTrendChart) window._uploadTrendChart.destroy();
      window._uploadTrendChart = new Chart(ctx, {
        type: 'line',
        data: { labels: trend.dates || [], datasets: [{ label: 'Harga Aktual', data: trend.values || [], tension: 0.2, pointRadius: 0 }] },
        options: { responsive: true, plugins: { legend: { display:false } } }
      });
    }
  } catch (e) {
    console.error('renderResults: trend chart error', e);
  }

  // --- pred vs actual chart: payload.pred_series or payload.predicted + payload.actual (predict_range style) ---
  let predChartLabels = [];
  let predActual = [];
  let predPred = [];

  if (payload.pred_series && payload.pred_series.dates) {
    predChartLabels = payload.pred_series.dates;
    predActual = payload.pred_series.actual || [];
    predPred = payload.pred_series.pred || [];
  } else if (payload.trend && payload.predictions_series) {
    // fallback if weird shape
    predChartLabels = payload.trend.dates || [];
    predActual = payload.trend.values || [];
    predPred = payload.predictions_series.values || [];
  } else if (payload.actual && payload.predicted) {
    predChartLabels = (payload.actual || []).map(p=>p.date);
    predActual = (payload.actual || []).map(p=>p.value);
    predPred = (payload.predicted || []).map(p=>p.value);
  } else {
    // try using trend for actual and shifted actual as pred
    predChartLabels = trend.dates || [];
    predActual = trend.values || [];
    predPred = (trend.values || []).slice(0); predPred.unshift(null); predPred.pop();
  }

  try {
    const ctx2 = document.getElementById('uploadPredChart')?.getContext?.('2d');
    if (ctx2) {
      if (window._uploadPredChart) window._uploadPredChart.destroy();
      window._uploadPredChart = new Chart(ctx2, {
        type: 'line',
        data: {
          labels: predChartLabels || [],
          datasets: [
            { label: 'Actual', data: predActual || [], pointRadius:0, tension:.2 },
            { label: 'Prediction', data: predPred || [], pointRadius:0, tension:.2, borderDash:[6,4] }
          ]
        },
        options: { responsive: true }
      });
    }
  } catch (e) {
    console.error('renderResults: pred chart error', e);
  }

  // --- full summary UI (if payload has training info) ---
  const fullSummary = document.getElementById('uploadFullSummary');
  const ul = document.getElementById('uploadFullResultsList');
  if (payload.mode === 'full' || payload.best_r2 || payload.pack_path || payload.metrics) {
    if (fullSummary) fullSummary.style.removeProperty('display');
    if (ul) {
      ul.innerHTML = `
        <li><b>Kota:</b> ${payload.city ?? '-'}</li>
        <li><b>n_total:</b> ${payload.n_total ?? '-'}</li>
        <li><b>test_days:</b> ${payload.test_days ?? '-'}</li>
        <li><b>best_r2:</b> ${payload.best_r2 ?? '-'}</li>
        <li><b>pack:</b> ${payload.pack_path ? payload.pack_path.split('/').pop() : '-'}</li>
        <li><b>metrics:</b> <pre style="white-space:pre-wrap">${JSON.stringify(payload.metrics || payload.metrics || {}, null, 2)}</pre></li>
      `;
    }
  } else {
    if (fullSummary) fullSummary.style.display = 'none';
    if (ul) ul.innerHTML = '';
  }
}


function downloadTemplateCSV() {
  const csv = "date,City A,City B\n2023-01-01,0,0\n";
  const blob = new Blob([csv], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'template_prices.csv';
  a.click();
}
function downloadTemplateExcel() {
  // buat placeholder: arahkan ke template di server jika ada. Untuk demo, fallback ke CSV:
  downloadTemplateCSV();
}



// ===== Patch kecil: pastikan Upload link ada & kliknya memanggil loadUpload() + show section =====
// -----------------------------
// Single delegated nav handler
// -----------------------------
document.addEventListener('DOMContentLoaded', () => {
  const nav = document.querySelector('.top-nav') || document.querySelector('nav');
  if (!nav) return;

  // Ensure Upload link exists (only create if missing)
  if (!nav.querySelector('.nav-link[data-section="upload"]')) {
    const uploadLink = document.createElement('a');
    uploadLink.href = '#';
    uploadLink.className = 'nav-link';
    uploadLink.dataset.section = 'upload';
    uploadLink.textContent = 'Upload';
    nav.appendChild(uploadLink);
  }

  // Delegated click handler untuk semua nav-link
  nav.addEventListener('click', (ev) => {
    const a = ev.target.closest('.nav-link');
    if (!a) return;
    ev.preventDefault();

    const targetId = a.getAttribute('data-section');
    if (!targetId) return;

    // update nav active state
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    a.classList.add('active');

    // hide semua section dulu
    document.querySelectorAll('.content-section').forEach(s => {
      s.classList.remove('active','is-visible');
      s.style.display = 'none';
    });

    // show target section
    const sec = document.getElementById(targetId);
    if (sec) {
      sec.style.removeProperty('display');
      sec.classList.add('active','is-visible');
      try { sec.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch(_) {}
    }

    // khusus: kalau buka upload => panggil loadUpload()
    if (targetId === 'upload' && typeof window.loadUpload === 'function') {
      try { window.loadUpload(); } catch (e) { console.error('loadUpload error', e); }
    }

    // resize map/chart safe-guards
    setTimeout(() => {
      try { if (window.map && typeof window.map.invalidateSize === 'function') window.map.invalidateSize(); } catch(_) {}
      try { if (window.trendChart && typeof window.trendChart.resize === 'function') window.trendChart.resize(); } catch(_) {}
      try { if (window._predChart && typeof window._predChart.resize === 'function') window._predChart.resize(); } catch(_) {}
      try { if (window._evChart && typeof window._evChart.resize === 'function') window._evChart.resize(); } catch(_) {}
    }, 300);
  });

  // Move ink underline to active (if you use the ink UI)
  const ink = nav.querySelector('.ink');
  if (ink) {
    const active = nav.querySelector('.nav-link.active') || nav.querySelector('.nav-link');
    if (active) {
      const rect = active.getBoundingClientRect(), parent = nav.getBoundingClientRect();
      ink.style.width = rect.width + 'px';
      ink.style.left  = (rect.left - parent.left) + 'px';
    }
  }
});
