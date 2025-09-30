// ===== NAV (toggle section) =====
document.querySelectorAll('.nav-link').forEach(link=>{
  link.addEventListener('click',e=>{
    e.preventDefault();
    document.querySelectorAll('.nav-link').forEach(l=>l.classList.remove('active'));
    document.querySelectorAll('.content-section').forEach(s=>s.classList.remove('active'));
    link.classList.add('active');
    document.getElementById(link.getAttribute('data-section')).classList.add('active');
  });
});

// ===== Scroll tombol Telusuri =====
document.getElementById('scroll-to-beranda')
  ?.addEventListener('click', ()=>{
    document.querySelector('.nav-link[data-section="beranda"]')?.click();
    document.getElementById('beranda')?.scrollIntoView({behavior:'smooth'});
  });

// ===== Utils =====
const monthsID=["","Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November","Desember"];
const rupiah=n=>new Intl.NumberFormat('id-ID').format(n);
const EVAL_DEFAULT_DAYS = 120; // jangka waktu otomatis untuk grafik evaluasi
const fmtNum = (x, dec=0) =>
  (x==null || Number.isNaN(x)) ? '-' :
  new Intl.NumberFormat('id-ID', { maximumFractionDigits: dec, minimumFractionDigits: dec }).format(x);

function normProv(s){return (s||"").toString().toLowerCase().replace(/provinsi|propinsi|prov\./g,"").replace(/\s+/g,"").replace(/[^\w]/g,"");}
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


// NAV (toggle section) — keep this one
document.querySelectorAll('.nav-link').forEach(link=>{
  link.addEventListener('click', e=>{
    e.preventDefault();
    document.querySelectorAll('.nav-link').forEach(l=>l.classList.remove('active'));
    document.querySelectorAll('.content-section').forEach(s=>s.classList.remove('active'));
    link.classList.add('active');
    document.getElementById(link.getAttribute('data-section')).classList.add('active');
  });
});

// Scroll tombol Telusuri — keep this one
document.getElementById('scroll-to-beranda')
  ?.addEventListener('click', ()=>{
    document.querySelector('.nav-link[data-section="beranda"]')?.click();
    document.getElementById('beranda')?.scrollIntoView({behavior:'smooth'});
  });

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
async function fetchPredictRange(citySlug, startISO, endISO) {
  const params = new URLSearchParams({
    city: citySlug, start: startISO, end: endISO,
    future_only: '0', hide_actual: '0', naive_fallback: '1'
  });
  const r = await fetch(`/api/predict_range?${params.toString()}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
let _predChart=null;
function renderPredChart(labels, actual, predicted, cityLabel) {
  const canvas = document.getElementById('predChart'); if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (_predChart) _predChart.destroy();

  _predChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: `Aktual • ${cityLabel}`,   data: actual,    borderWidth: 2, tension: 0.25, pointRadius: 0 },
        { label: `Prediksi • ${cityLabel}`, data: predicted, borderWidth: 2, tension: 0.25, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { title: { display: true, text: 'Aktual vs Prediksi (Gradient Boosting)' } },
      scales: { x: { ticks: { autoSkip: true, maxTicksLimit: 12 } }, y: {} }
    }
  });
}
async function loadPrediksi(){
const sel = document.getElementById('kabupaten_a');
  const citySlug = selectedSlugOrLabel(sel);
  const start = document.getElementById('startDate').value;
  const end   = document.getElementById('endDate').value;
  const ph = document.getElementById('predPlaceholder');

  if (!citySlug) { alert('Pilih Kabupaten/Kota.'); return; }
  if (!start || !end) { alert('Tanggal mulai & akhir wajib diisi.'); return; }

  try {
    ph.style.display = 'none';

    const r = await fetchPredictRange(citySlug, start, end);
    const labels = fullDateRange(start, end);

    // r.actual & r.predicted format: [{date: 'YYYY-MM-DD', value: number}, ...]
    const mapActual = new Map((r.actual || []).map(p => [p.date, p.value]));
    const mapPred   = new Map((r.predicted || []).map(p => [p.date, p.value]));

    const dsActual  = labels.map(d => mapActual.has(d) ? mapActual.get(d) : null);
    const dsPred    = labels.map(d => mapPred.has(d)   ? mapPred.get(d)   : null);

    const cityLabel = (r.entity || r.city || citySlug).replace(/_/g,' ');
    renderPredChart(labels, dsActual, dsPred, cityLabel);

    document.querySelector('.nav-link[data-section="prediksi"]')?.click();
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
/* =========================================================
   FUTURISTIC UI ENHANCEMENTS (append-only; non-breaking)
   ========================================================= */

/* 0) Smooth nav switch (kode lama kamu sudah handle active state) */
document.querySelectorAll('.nav-link').forEach(link=>{
  link.addEventListener('click',e=>{
    e.preventDefault();
    document.querySelectorAll('.nav-link').forEach(l=>l.classList.remove('active'));
    document.querySelectorAll('.content-section').forEach(s=>s.classList.remove('active'));
    link.classList.add('active');
    document.getElementById(link.getAttribute('data-section')).classList.add('active');
  });
});

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
document.getElementById('scroll-to-beranda')
  ?.addEventListener('click', ()=>{
    document.querySelector('.nav-link[data-section="beranda"]')?.click();
    document.getElementById('beranda')?.scrollIntoView({behavior:'smooth'});
  });

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
