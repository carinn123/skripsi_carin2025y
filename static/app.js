
function _safeSetText(id, txt) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = txt;
}
async function populateQuickSelect(){ /* ... */ }
async function quickPredictFetchAndRender(entity, opts){ /* ... */ }

// --- lainnya, fungsi render/chart, dll ---

// --- akhirnya: hook DOM ready ---
document.addEventListener('DOMContentLoaded', () => {
  populateQuickSelect();
  const sel = document.getElementById('quick_kabupaten');
  if (sel) {
    sel.addEventListener('change', (e) => {
      const entity = (e.target.value || '').trim();
      if (!entity) {
        document.getElementById('quickPredResult').style.display = 'none';
        return;
      }
      quickPredictFetchAndRender(entity, { mode: 'test' });
    });
  }
});
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


// --- replace existing fetchJsonSafe with this robust helper ---
async function fetchJsonSafe(url, opts = {}) {
  // opts passed directly to fetch (method, body, headers, ...)
  let res;
  try {
    res = await fetch(url, opts);
  } catch (networkErr) {
    // network failure (DNS, offline, CORS, etc)
    return {
      ok: false,
      status: 0,
      statusText: networkErr && networkErr.message ? String(networkErr.message) : 'network error',
      headers: {},
      json: null,
      text: null,
      isFile: false,
      error: networkErr
    };
  }

  const status = res.status;
  const statusText = res.statusText || '';
  const headers = {};
  res.headers.forEach((v, k) => headers[k.toLowerCase()] = v);

  // detect file-like responses (Excel, octet-stream, etc)
  const ctype = (headers['content-type'] || '').toLowerCase();
  const disposition = (headers['content-disposition'] || '').toLowerCase();
  const isFile = ctype.includes('application/octet-stream') ||
                 ctype.includes('application/vnd.ms-excel') ||
                 ctype.includes('application/vnd.openxmlformats-officedocument') ||
                 disposition.includes('attachment');

  // try to read as text first (safe), but if file -> return blob
  let text = null;
  let json = null;
  try {
    if (isFile) {
      // caller likely expects a download blob (Excel). Return minimal info and blob separately.
      const blob = await res.blob();
      return {
        ok: res.ok,
        status,
        statusText,
        headers,
        json: null,
        text: null,
        isFile: true,
        blob
      };
    } else {
      text = await res.text();
      if (text) {
        // prefer parsing when content-type claims json
        if (ctype.includes('application/json') || ctype.includes('text/json')) {
          try { json = JSON.parse(text); } catch (e) { json = null; }
        } else {
          // be permissive: try parse anyway
          try { json = JSON.parse(text); } catch (e) { json = null; }
        }
      }
    }
  } catch (readErr) {
    return {
      ok: res.ok,
      status,
      statusText,
      headers,
      json: null,
      text: null,
      isFile,
      error: readErr
    };
  }

  return {
    ok: res.ok,
    status,
    statusText,
    headers,
    json,   // parsed JSON or null
    text,   // raw text or null
    isFile  // boolean
  };
}


function goToDetailPrediksi() {
  // Pindah ke tab "Prediksi" saja tanpa set apa pun
  const predNav = document.querySelector('.nav-link[data-section="prediksi"]');
  if (predNav) {
    predNav.click();
  } else {
    console.warn("Tab 'Prediksi' tidak ditemukan.");
  }

  // Optional: scroll ke atas tab prediksi biar user langsung lihat
  setTimeout(() => {
    const section = document.getElementById('prediksi');
    if (section) {
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, 200);
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
function normProv(s){
  if(!s) return "";
  // trim + ke lowercase
  s = s.toString().trim().toLowerCase();
  // normalisasi unicode (hilangkan diakritik)
  if(s.normalize) s = s.normalize('NFKD').replace(/[\u0300-\u036f]/g,'');
  // hilangkan token umum yang bikin mismatch (provinsi, propinsi, prov., kab., kota, kep.)
  s = s.replace(/\b(provinsi|propinsi|prov\.|kabupaten|kab\.|kota|kepulauan|kep\.)\b/g,'');
  // hapus semua non-alphanumeric
  s = s.replace(/[^a-z0-9]+/g,'');
  // trim lagi
  return s.trim();
}
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

function showMapValidation(msg){
  const el = document.getElementById('mapValidation');
  if(!el) return;
  el.innerHTML = `<small><strong style="color:#b45309">⚠️ ${msg}</strong></small>`;
  el.style.display = 'block';
}
function hideMapValidation(){
  const el = document.getElementById('mapValidation');
  if(!el) return;
  el.style.display = 'none';
}


// === showMapBeranda (update versi final) ===
async function showMapBeranda(event){
  const islandInput = document.getElementById('pulau');
  const island = islandInput ? islandInput.value.trim() : '';
  const tahun = document.getElementById('b_tahun').value;
  const bulan = document.getElementById('b_bulan').value;
  const minggu = document.getElementById('b_minggu').value;
  const loading = document.getElementById('b_loading');

  // NEW: mode (actual | predicted)
  const mode = (document.getElementById('b_mode') && document.getElementById('b_mode').value) || 'actual';

  // validasi dasar
  if(!island){
    showMapValidation('Pilih Pulau terlebih dahulu.');
    return;
  }
  if(!tahun){
    showMapValidation('Pilih Tahun terlebih dahulu.');
    return;
  }
  if(mode === 'predicted' && !bulan){
    showMapValidation('Untuk Predicted, silakan pilih Bulan terlebih dahulu.');
    return;
  }
  hideMapValidation();

  loading.style.display='inline-block';

  try{
    const url = `/api/choropleth?island=${encodeURIComponent(island)}&year=${encodeURIComponent(tahun)}&month=${encodeURIComponent(bulan)}&week=${encodeURIComponent(minggu)}&mode=${encodeURIComponent(mode)}`;
    const res = await fetch(url, {cache:'no-store'});
    const js = await res.json().catch(()=>({}));
    if(!res.ok) throw new Error(js.error || ('HTTP '+res.status));

    const m = ensureMap();
    const gj = await getProvGeoJSON();

    // buat mapping provinsi -> nilai
    const vmap = Object.fromEntries((js.data||[]).map(d=>[
      normProv(d.province),
      { val:d.value, cat:d.category, label:d.province }
    ]));

    if(geoLayer) geoLayer.remove();
    geoLayer = L.geoJSON(gj, {
      style: f => {
        const raw = f.properties.Propinsi || f.properties.PROVINSI || f.properties.provinsi || f.properties.name || f.properties.NAMOBJ || "";
        const rec = vmap[normProv(raw)];
        const fill = rec
          ? (rec.cat==='low' ? '#2ecc71' : rec.cat==='mid' ? '#f1c40f' : '#e74c3c')
          : '#bdc3c7';
        return { color:'#fff', weight:1, fillColor:fill, fillOpacity:.85 };
      },
      onEachFeature: (feature, layer) => {
        const raw = feature.properties.Propinsi || feature.properties.PROVINSI || feature.properties.provinsi || feature.properties.name || feature.properties.NAMOBJ || "—";
        const rec = vmap[normProv(raw)];
        const val = rec ? Math.round(rec.val) : null;
        const cat = rec ? rec.cat : 'no-data';

        let meta = '';
        if(js.mode === 'predicted' || js.generated_at || js.model_version){
          const gen = js.generated_at ? `Generated: ${js.generated_at}` : '';
          const mv = js.model_version ? `Model: ${js.model_version}` : '';
          meta = `<br/><small style="color:#6b7280">${[gen,mv].filter(Boolean).join(' • ')}</small>`;
        }

        layer.bindPopup(`<b>${raw}</b><br/>${val ? ('Rp '+rupiah(val)) : '—'}<br/>Kategori: ${cat}${meta}`);
        if (!layer._provClickTimer) layer._provClickTimer = null;
          layer.on('click', function(e){
            const provName = (feature.properties.Propinsi || feature.properties.PROVINSI || feature.properties.provinsi || feature.properties.name || feature.properties.NAMOBJ || "").trim();
            if(!provName) return;
            if(layer._provClickTimer) clearTimeout(layer._provClickTimer);
            layer._provClickTimer = setTimeout(()=>{
              // non-blocking call to load table for this province
              if(typeof loadRegionSummaryForProvince === 'function'){
                loadRegionSummaryForProvince(provName).catch(err => console.error('loadRegionSummaryForProvince', err));
              } else {
                console.warn('loadRegionSummaryForProvince not defined');
              }
              layer._provClickTimer = null;
            }, 300);
          });
        // use a distinct variable name and a tiny debounce to avoid spam
        let _provClickTimer = null;
        layer.on('click', function(e){
          const provName = (feature.properties.Propinsi || feature.properties.PROVINSI || feature.properties.provinsi || feature.properties.name || feature.properties.NAMOBJ || "").trim();
          if(!provName) return;
          // debounce 300ms to avoid double-click storms
          if(_provClickTimer) clearTimeout(_provClickTimer);
          _provClickTimer = setTimeout(()=>{
            // call loader for province (implementasi di bawah)
            loadRegionSummaryForProvince(provName);
            _provClickTimer = null;
          }, 300);
});
      }
    }).addTo(m);

    try { m.fitBounds(geoLayer.getBounds(), {padding:[20,20]}); } catch(e){}

    // === 🔥 trigger tabel eksplor kab/kota ===
    loadRegionSummaryFromMap(island).catch(err => {
      console.error('loadRegionSummaryFromMap failed', err);
    });

    // === update informasi statistik ===
    document.getElementById('statPulau').textContent = island || 'Semua Pulau';

    let scope = `Tahun ${tahun}`;
    if(bulan) scope = `Bulan ${monthsID[+bulan]} ${tahun}`;
    if(bulan && minggu) scope = `Minggu ke-${minggu}, ${monthsID[+bulan]} ${tahun}`;
    document.getElementById('statScope').textContent = scope;

    if(js.generated_at){
      try{
        const d = new Date(js.generated_at);
        const opt = { year:'numeric', month:'short', day:'numeric' };
        document.getElementById('statLast').textContent = 
          `Predicted: ${d.toLocaleDateString('id-ID', opt)}`;
      }catch(e){
        document.getElementById('statLast').textContent = `Predicted: ${js.generated_at.split('T')[0]}`;
      }
    } else {
      document.getElementById('statLast').textContent = js.last_actual || '-';
    }

  }catch(e){
    console.error(e);
    alert('Gagal menampilkan peta: '+(e.message||e));
  }finally{
    loading.style.display='none';
  }
}

window.showMapBeranda = showMapBeranda;


// >>> ADDED >>> START helper functions for region table (paste after showMapBeranda)

// render rows into #rc_table tbody
function fillRegionTable(rows){
  const tbody = document.querySelector('#rc_table tbody');
  if(!tbody){
    console.warn('fillRegionTable: tbody not found');
    return;
  }
  if(!rows || rows.length === 0){
    tbody.innerHTML = `<tr><td colspan="12" style="text-align:center;color:var(--muted)">Tidak ada data untuk pilihan ini.</td></tr>`;
    return;
  }

  let html = '';
  rows.forEach(r => {
    const highPeriod = (r.avg_month_high_month && r.avg_month_high_year) ? `${r.avg_month_high_month}/${r.avg_month_high_year}` :
                       (r.avg_month_high_month ? `${r.avg_month_high_month}` : '—');
    const lowPeriod  = (r.avg_month_low_month && r.avg_month_low_year) ? `${r.avg_month_low_month}/${r.avg_month_low_year}` :
                       (r.avg_month_low_month ? `${r.avg_month_low_month}` : '—');

    html += `<tr>
      <td>${r.no ?? ''}</td>
      <td>${r.city ?? ''}</td>
      <td>${r.province ?? ''}</td>
      <td>${r.island ?? ''}</td>
      <td class="num">${(r.min_value!=null) ? rupiah(Math.round(r.min_value)) : '—'}</td>
      <td class="num">${r.min_date ?? '—'}</td>
      <td class="num">${(r.max_value!=null) ? rupiah(Math.round(r.max_value)) : '—'}</td>
      <td class="num">${r.max_date ?? '—'}</td>
      <td class="num">${(r.avg_month_high!=null) ? rupiah(Math.round(r.avg_month_high)) : '—'}</td>
      <td class="num">${highPeriod}</td>
      <td class="num">${(r.avg_month_low!=null) ? rupiah(Math.round(r.avg_month_low)) : '—'}</td>
      <td class="num">${lowPeriod}</td>
    </tr>`;
  });

  tbody.innerHTML = html;
}

// load by island (called automatically after showMapBeranda)
const __RC_CACHE = {}; // simple in-memory cache
const RC_CACHE_TTL = 1000 * 60 * 5; // 5 minutes

async function loadRegionSummaryFromMap(island){
  const tbody = document.querySelector('#rc_table tbody');
  const loader = document.getElementById('rc_loading');
  if(loader) loader.style.display = 'inline-block';

  try{
    if(!island){
      fillRegionTable([]);
      return;
    }

    const key = `island:${island}`;
    const now = Date.now();
    if(__RC_CACHE[key] && (now - __RC_CACHE[key].ts) < RC_CACHE_TTL){
      fillRegionTable(__RC_CACHE[key].rows);
      return;
    }

    const url = `/api/region_summary?mode=island&value=${encodeURIComponent(island)}`;
    const res = await fetch(url, {cache:'no-store'});
    const js = await res.json().catch(()=>({}));

    if(!res.ok){
      console.error('region_summary error', js);
      fillRegionTable([]);
      return;
    }

    const rows = js.rows || [];
    fillRegionTable(rows);
    __RC_CACHE[key] = { ts: now, rows };
  }catch(err){
    console.error('loadRegionSummaryFromMap error', err);
    fillRegionTable([]);
  }finally{
    if(loader) loader.style.display = 'none';
  }
}

// load by province (called when user clicks a province on the map)
async function loadRegionSummaryForProvince(prov){
  const tbody = document.querySelector('#rc_table tbody');
  const loader = document.getElementById('rc_loading');
  if(loader) loader.style.display = 'inline-block';

  try{
    if(!prov){
      fillRegionTable([]);
      return;
    }
    const key = `province:${prov}`;
    const now = Date.now();
    if(__RC_CACHE[key] && (now - __RC_CACHE[key].ts) < RC_CACHE_TTL){
      fillRegionTable(__RC_CACHE[key].rows);
      return;
    }

    const url = `/api/region_summary?mode=province&value=${encodeURIComponent(prov)}`;
    const res = await fetch(url, {cache:'no-store'});
    const js = await res.json().catch(()=>({}));

    if(!res.ok){
      console.error('region_summary (province) error', js);
      fillRegionTable([]);
      return;
    }

    const rows = js.rows || [];
    fillRegionTable(rows);
    __RC_CACHE[key] = { ts: now, rows };
  }catch(err){
    console.error('loadRegionSummaryForProvince error', err);
    fillRegionTable([]);
  }finally{
    if(loader) loader.style.display = 'none';
  }
}

// >>> ADDED >>> END helper functions

// tambahan: bila user ganti mode, update UI (hanya untuk UX)
const bmode = document.getElementById('b_mode');
if(bmode){
  bmode.addEventListener('change', function(){
    // jika predicted -> show small note (and encourage select month)
    const note = document.getElementById('pred_info');
    if(note){
      note.textContent = (this.value === 'predicted') ? 'Mode: Predicted (pilih Bulan untuk hasil precomputed)' : 'Mode: Actual';
    }
  });
}





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

  // label untuk sumbu X: sesuaikan dengan granularity
  const xAxisLabel = granularity==='yearly' ? 'Bulan' :
                     granularity==='monthly' ? 'Minggu' : 'Tanggal';

  const data = {
    labels,
    datasets: [{ 
      label: datasetLabel, 
      data: values, 
      tension: .25, 
      fill: true, 
      pointRadius: 2, 
      borderWidth: 2,
      borderColor: '#4A90E2',
      backgroundColor: 'rgba(74,144,226,0.08)'
    }]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: { 
        display: true, 
        text: title || 'Tren Harga',
        color: '#111',
        font: { size: 14, weight: '600' }
      },
      tooltip: {
        callbacks: {
          label: (c) => ` ${datasetLabel}: Rp ${rupiah(c.parsed.y)}`
        }
      },
      legend: {
        labels: { color: '#333' }
      }
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Harga (Rp)',
          color: '#111',
          font: { size: 12, weight: '600' }
        },
        ticks: {
          callback: v => 'Rp ' + rupiah(v),
          color: '#444'
        },
        grid: {
          color: '#eee'
        }
      },
      x: {
        title: {
          display: true,
          text: xAxisLabel,
          color: '#111',
          font: { size: 12, weight: '600' }
        },
        ticks: {
          autoSkip: true,
          maxRotation: 0,
          color: '#444',
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
        grid: {
          color: '#f7f7f7'
        }
      }
    }
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
        {
          label: nameA || 'Kota 1',
          data: valuesA,
          tension: .25,
          pointRadius: 2,
          borderWidth: 2,
          borderColor: '#4A90E2',
          backgroundColor: 'rgba(74,144,226,0.06)',
          spanGaps: true
        },
        {
          label: nameB || 'Kota 2',
          data: valuesB,
          tension: .25,
          pointRadius: 2,
          borderWidth: 2,
          borderColor: '#FF8A65',
          backgroundColor: 'rgba(255,138,101,0.06)',
          spanGaps: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { 
          display: true,
          text: 'Tren Harga • Grafik Gabungan',
          color: '#111',
          font: { size: 14, weight: '600' }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const lab = ctx.dataset.label || '';
              const val = ctx.parsed.y;
              return ` ${lab}: Rp ${rupiah(val)}`;
            }
          }
        },
        legend: {
          labels: { color: '#333' }
        }
      },
      scales: {
        y: {
          title: {
            display: true,
            text: 'Harga (Rp)',
            color: '#111',
            font: { size: 12, weight: '600' }
          },
          ticks: {
            callback: v => 'Rp ' + rupiah(v),
            color: '#444'
          },
          grid: { color: '#eee' }
        },
        x: {
          title: {
            display: true,
            text: 'Tanggal',
            color: '#111',
            font: { size: 12, weight: '600' }
          },
          ticks: {
            autoSkip: true,
            maxRotation: 0,
            color: '#444',
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
          grid: { color: '#f7f7f7' }
        }
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
function getTestCutoff() {
  try {
    const el = document.getElementById('prediksi');
    if (el && el.dataset && el.dataset.testCutoff) return el.dataset.testCutoff;
  } catch(e){}
  return '2025-07-01'; // fallback
}

// ---------- modal helper (sama seperti sebelumnya) ----------
function showTestCutoffModal(cutoffIso) {
  return new Promise((resolve) => {
    const modal = document.getElementById('test-cutoff-modal');
    if (!modal) {
      // fallback simple confirm
      const ok = confirm(`Mode 'test' tidak boleh melewati ${cutoffIso}.\nOK = ganti ke 'real', Cancel = batal.`);
      return resolve(ok ? 'switch_mode' : 'cancel');
    }
    modal.style.display = 'block';
    modal.setAttribute('aria-hidden','false');
    const title = modal.querySelector('#tm-title');
    if (title) title.innerHTML = `Mode <code>test</code> melebihi cutoff (${cutoffIso})`;
    const btnSwitch = document.getElementById('tm-btn-switch');
    const btnSetCut = document.getElementById('tm-btn-setcut');
    const btnCancel = document.getElementById('tm-btn-cancel');

    function cleanup(){ modal.style.display='none'; modal.setAttribute('aria-hidden','true'); btnSwitch.removeEventListener('click',onSwitch); btnSetCut.removeEventListener('click',onSet); btnCancel.removeEventListener('click',onCancel); }
    function onSwitch(){ cleanup(); resolve('switch_mode'); }
    function onSet(){ cleanup(); resolve('set_cutoff'); }
    function onCancel(){ cleanup(); resolve('cancel'); }

    btnSwitch.addEventListener('click', onSwitch);
    btnSetCut.addEventListener('click', onSet);
    btnCancel.addEventListener('click', onCancel);
  });
}

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
          borderColor: '#4A90E2',     // 💙 warna garis aktual
          backgroundColor: '#4A90E233',
          borderWidth: 2,
          tension: 0.25,
          pointRadius: 0,
          spanGaps: true
        },
        {
          label: `Prediksi • ${cityLabel}`,
          data: predicted,
          borderColor: '#FF6384',     // 💗 warna garis prediksi
          borderDash: [6, 6],
          backgroundColor: '#FF638433',
          borderWidth: 2,
          tension: 0.25,
          pointRadius: 0,
          spanGaps: true
        }
      ]
    },
    options: {
      plugins: {
        title: {
          display: true,
          text: 'Aktual vs Prediksi (Gradient Boosting)',
          color: '#000',
          font: { size: 14, weight: 'bold' }
        },
        legend: {
          labels: {
            color: '#333', // warna teks legend
            font: { size: 12 }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Tanggal',      // 🕓 label sumbu X
            color: '#111',
            font: { size: 13, weight: 'bold' }
          },
          ticks: {
            color: '#444',         // warna teks tanggal
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
          grid: {
            drawTicks: true,
            color: '#eee'           // warna garis grid X
          }
        },
        y: {
          title: {
            display: true,
            text: 'Harga (Rp)',     // 💰 label sumbu Y
            color: '#111',
            font: { size: 13, weight: 'bold' }
          },
          ticks: {
            color: '#444',          // warna angka di sumbu Y
            count: 8
          },
          grid: {
            color: '#eee'           // warna garis horizontal
          }
        }
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

function _renderPredSummaryFromServer(summaryObj){
  const box = document.getElementById('predSummary');
  if (!box) return;

  // summaryObj expected like: { n, avg, min, min_date, max, max_date, start, end, change_pct }
  const s = summaryObj || {};
  if (!s || (s.n === 0)) {
    box.style.display = 'none';
    return;
  }
  box.style.removeProperty('display');

  const fmt = n => (n==null || Number.isNaN(n))? '-' : 'Rp '+new Intl.NumberFormat('id-ID').format(Math.round(n));
  const fmtPct = n => (n==null || Number.isNaN(n))? '-' : (n>=0? '+' : '') + Number(n).toFixed(2) + '%';

  _safeSetText('predMaxVal',  fmt(s.max));
  _safeSetText('predMaxDate', s.max_date ? _niceDate(s.max_date) : '—');
  _safeSetText('predMinVal',  fmt(s.min));
  _safeSetText('predMinDate', s.min_date ? _niceDate(s.min_date) : '—');
  _safeSetText('predAvgVal',  fmt(s.avg));
  _safeSetText('predCount',   `n = ${s.n} hari`);

  // optional fields (may not be present in HTML)
  const changeText = (s.start!=null && s.end!=null) ? `${fmt(s.start)} → ${fmt(s.end)}` : '—';
  _safeSetText('predChangePct', fmtPct(s.change_pct));
  _safeSetText('predChangeNote', changeText);
}
async function fetchPredictRange(citySlug, startISO, endISO, mode = 'test') {
   // normalisasi mode
  const modeNorm = (mode || 'test').toString().toLowerCase();
  const allowed = new Set(['test','real']);
  const modeToSend = allowed.has(modeNorm) ? modeNorm : 'test';
  const params = new URLSearchParams({
    city: citySlug,
    start: startISO,
    end: endISO,
    mode: modeToSend,
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

async function quickPredictFetchAndRender(entitySlug, opts = { mode: 'test' }) {
  if (!entitySlug) return;
  const loading = document.getElementById('quickPredLoading');
  const resultBox = document.getElementById('quickPredResult');

  const elTodayVal = document.getElementById('quickTodayValue');
  const elTodayDate = document.getElementById('quickTodayDate');
  const elTomorrowVal = document.getElementById('quickTomorrow');
  const elTomorrowDate = document.getElementById('quickTomorrowDate');
  const el7Val = document.getElementById('quick7Days');
  const el7Date = document.getElementById('quick7DaysDate');
  const el30Val = document.getElementById('quick30Days');
  const el30Date = document.getElementById('quick30DaysDate');
  const chartCanvas = document.getElementById('quickPredChart');

  const fmtMoney = n => (n == null || Number.isNaN(n)) ? '-' : 'Rp ' + new Intl.NumberFormat('id-ID').format(Math.round(n));
  const niceDate = iso => {
    try { const d = new Date(iso); return d.toLocaleDateString('id-ID', { weekday:'long', year:'numeric', month:'long', day:'numeric' }); }
    catch { return iso; }
  };
  const shortDate = iso => {
    try { const d = new Date(iso); return d.toLocaleDateString('id-ID', { day:'2-digit', month:'short', year:'numeric' }); }
    catch { return iso; }
  };

  // UI prepare
  if (loading) loading.style.display = '';
  if (resultBox) resultBox.style.display = 'none';

  try {
    const url = `/api/quick_predict?city=${encodeURIComponent(entitySlug)}&mode=${encodeURIComponent(opts.mode||'test')}`;
    const res = await fetch(url);
    if (!res.ok) {
      const t = await res.text().catch(()=>res.statusText);
      throw new Error(`${res.status} ${t}`);
    }
    const j = await res.json();
    if (!j.ok) throw new Error(j.error || 'no ok');

    // fill main today
    if (elTodayVal) elTodayVal.textContent = fmtMoney(j.last_value);
    if (elTodayDate) elTodayDate.textContent = niceDate(j.last_actual);

    // fill horizons
    if (j.predictions && j.predictions["1"]) {
      elTomorrowVal.textContent = fmtMoney(j.predictions["1"].value);
      elTomorrowDate.textContent = shortDate(j.predictions["1"].date);
    } else {
      elTomorrowVal.textContent = '-';
      elTomorrowDate.textContent = '';
    }
    if (j.predictions && j.predictions["7"]) {
      el7Val.textContent = fmtMoney(j.predictions["7"].value);
      el7Date.textContent = shortDate(j.predictions["7"].date);
    } else {
      el7Val.textContent = '-'; el7Date.textContent = '';
    }
    if (j.predictions && j.predictions["10"]) {
      el30Val.textContent = fmtMoney(j.predictions["10"].value);
      el30Date.textContent = shortDate(j.predictions["10"].date);
    } else {
      el30Val.textContent = '-'; el30Date.textContent = '';
    }

    // render small chart: history + predicted points
    if (chartCanvas && (Array.isArray(j.history) || j.predictions)) {
      const labels = [];
      const dataVals = [];

      // history
      if (Array.isArray(j.history)) {
        for (const p of j.history) {
          labels.push(p.date);
          dataVals.push(Number(p.value));
        }
      }

      // append predicted up to 10 days (preserve label order)
      const predsArr = [];
      for (const k of ['1','7','10']) {
        if (j.predictions && j.predictions[k]) predsArr.push(j.predictions[k]);
      }
      // to show fuller curve, try to request longer preds if desired — for now add predsArr in chronological order
      predsArr.sort((a,b)=> (new Date(a.date)) - (new Date(b.date)) );
      for (const p of predsArr) {
        labels.push(p.date);
        dataVals.push(Number(p.value));
      }

      try {
        if (window.quickChartRef && window.quickChartRef.destroy) {
          window.quickChartRef.destroy();
          window.quickChartRef = null;
        }
        if (typeof Chart !== 'undefined') {
          window.quickChartRef = new Chart(chartCanvas.getContext('2d'), {
            type: 'line',
            data: { labels: labels, datasets: [{ label: 'Harga (Rp/L)', data: dataVals, tension: 0.2, fill:false }]},
            options: { plugins:{legend:{display:false}}, scales:{ x:{display:true}, y:{display:true} }, responsive:true, maintainAspectRatio:false }
          });
        }
      } catch (err) { console.warn("quick chart error", err); }
    }

    // show result
    if (loading) loading.style.display = 'none';
    if (resultBox) resultBox.style.removeProperty('display');

    return j;
  } catch (err) {
    console.error("quickPredict error:", err);
    if (loading) loading.style.display = 'none';
    if (resultBox) {
      resultBox.style.removeProperty('display');
      document.getElementById('quickTodayValue').textContent = 'Gagal memuat';
      document.getElementById('quickTodayDate').textContent = '';
    }
    return null;
  }
}

// Isi select #quick_kabupaten dari endpoint /api/cities_full
async function populateQuickSelect() {
  const sel = document.getElementById('quick_kabupaten');
  if (!sel) {
    console.warn("populateQuickSelect: #quick_kabupaten not found");
    return;
  }

  try {
    const resp = await fetch('/api/cities_full', { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const arr = await resp.json();

    // Keep the first placeholder option if present
    const firstOpt = sel.querySelector('option') ? sel.querySelector('option').outerHTML : '<option value="">-- Pilih --</option>';
    sel.innerHTML = firstOpt;

    arr.forEach(item => {
      // server returns objects like { entity, slug, label }
      const opt = document.createElement('option');
      opt.value = item.entity || item.slug || (item.label || '').toLowerCase().replace(/\s+/g,'_');
      opt.textContent = item.label || item.entity || opt.value;
      sel.appendChild(opt);
    });

    console.log(`populateQuickSelect: loaded ${arr.length} cities`);
  } catch (err) {
    console.warn('populateQuickSelect failed, keeping existing options:', err);
    // leave existing static options in place as fallback
  }
}


async function loadPrediksi(){
  // fallback ke #kabupaten kalau #kabupaten_a tidak ada
  console.log("loadPrediksi called");
  // const sel = document.getElementById('citySearch') || document.getElementById('kabupaten');
  const hidden = document.getElementById('kabupaten_a');
  const citySlug =hidden ? hidden.value : '';
  const start = document.getElementById('startDate').value;
  const end   = document.getElementById('endDate').value;
  const ph = document.getElementById('predPlaceholder');
  const predModeEl = document.getElementById('predMode');
  const mode = predModeEl ? (predModeEl.value || 'test') : 'test';

  if (!citySlug) { alert('Pilih Kabupaten/Kota.'); return; }
  if (!start || !end) { alert('Tanggal mulai & akhir wajib diisi.'); return; }

  try {
    // only enforce cutoff for mode === 'test'
    if ((mode || '').toString().toLowerCase() === 'test') {
      const cutoffIso = getTestCutoff();
      const endDate = new Date(end + 'T00:00:00');
      const cutoffDate = new Date(cutoffIso + 'T00:00:00');
      if (endDate > cutoffDate) {
        const action = await showTestCutoffModal(cutoffIso);
        if (action === 'cancel') return;                   // user abort
        if (action === 'switch_mode') {                    // switch to real
          if (predModeEl) { predModeEl.value = 'real'; mode = 'real'; }
        } else if (action === 'set_cutoff') {              // clamp inputs
          if (endInput) endInput.value = cutoffIso;
          if (new Date(start + 'T00:00:00') > cutoffDate && startInput) startInput.value = cutoffIso;
        }
      }
    }
  } catch(err){
    console.warn("cutoff validation error", err);
  }
  try {
    ph.style.display = 'none';

    const r = await fetchPredictRange(citySlug, start, end, mode);
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
// Helper: render region table rows
function fillRegionTable(rows){
  const tbody = document.querySelector('#rc_table tbody');
  if(!tbody){
    console.warn('fillRegionTable: tbody not found');
    return;
  }

  if(!rows || rows.length === 0){
    tbody.innerHTML = `<tr><td colspan="12" style="text-align:center;color:var(--muted)">Tidak ada data untuk pilihan ini.</td></tr>`;
    return;
  }

  let html = '';
  rows.forEach(r => {
    // format periode bulan/year tampil rapi
    const highPeriod = (r.avg_month_high_month && r.avg_month_high_year) ? `${r.avg_month_high_month}/${r.avg_month_high_year}` : (r.avg_month_high_month ? `${r.avg_month_high_month}` : '—');
    const lowPeriod  = (r.avg_month_low_month && r.avg_month_low_year) ? `${r.avg_month_low_month}/${r.avg_month_low_year}` : (r.avg_month_low_month ? `${r.avg_month_low_month}` : '—');

    html += `<tr>
      <td>${r.no ?? ''}</td>
      <td>${r.city ?? ''}</td>
      <td>${r.province ?? ''}</td>
      <td>${r.island ?? ''}</td>
      <td class="num">${(r.min_value!=null) ? rupiah(Math.round(r.min_value)) : '—'}</td>
      <td class="num">${r.min_date ?? '—'}</td>
      <td class="num">${(r.max_value!=null) ? rupiah(Math.round(r.max_value)) : '—'}</td>
      <td class="num">${r.max_date ?? '—'}</td>
      <td class="num">${(r.avg_month_high!=null) ? rupiah(Math.round(r.avg_month_high)) : '—'}</td>
      <td class="num">${highPeriod}</td>
      <td class="num">${(r.avg_month_low!=null) ? rupiah(Math.round(r.avg_month_low)) : '—'}</td>
      <td class="num">${lowPeriod}</td>
    </tr>`;
  });

  tbody.innerHTML = html;
}

// simple in-memory cache to avoid repeated calls in short time
const __RC_CACHE = {}; // key -> {ts, rows}
const RC_CACHE_TTL = 1000 * 60 * 5; // 5 minutes

async function loadRegionSummaryFromMap(island){
  const tbody = document.querySelector('#rc_table tbody');
  const loader = document.getElementById('rc_loading');
  if(loader) loader.style.display = 'inline-block';

  try{
    // ambil filter waktu dari UI
    const year = document.getElementById('b_tahun')?.value || '';
    const month = document.getElementById('b_bulan')?.value || '';
    const week = document.getElementById('b_minggu')?.value || '';

    // skip kalau ga ada pulau
    if(!island){
      fillRegionTable([]);
      return;
    }

    const params = new URLSearchParams();
    params.set('mode', 'island');
    params.set('value', island);
    if(year)  params.set('year', year);
    if(month) params.set('month', month);
    if(week && month) params.set('week', week);

    // optional: tambahkan ?predict=1 untuk predicted mode otomatis
    const modeSel = document.getElementById('b_mode');
    if(modeSel && modeSel.value === 'predicted') {
      params.set('predict', '1');
    }

    const url = `/api/region_summary?${params.toString()}`;
    const res = await fetch(url, {cache:'no-store'});
    const js = await res.json().catch(()=>({}));

    if(!res.ok){
      console.error('region_summary error', js);
      fillRegionTable([]);
      return;
    }

    fillRegionTable(js.rows || []);
  }catch(err){
    console.error('loadRegionSummaryFromMap error', err);
    fillRegionTable([]);
  }finally{
    if(loader) loader.style.display = 'none';
  }
}



  // async function runRegional() {
  //   const mode  = rcModeSel.value || 'island';
  //   const value = rcValueSel.value || '';
  //   if (!value) { alert('Pilih nilai pada dropdown kedua.'); return; }
  //   showSpin(true);
  //   try {
  //     const js = await fetchRegionSummary(mode, value);
  //     renderRegionRows(js.rows || []);
  //   } catch (e) {
  //     console.error(e);
  //     rcRenderEmpty(`Gagal memuat: ${e.message}`);
  //   } finally {
  //     showSpin(false);
  //   }
  // }

  // wiring
  // rcModeSel.addEventListener('change', fillRcValueOptions);
  // rcBtn && rcBtn.addEventListener('click', runRegional);

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
  set('mseValue',  mse == null ? '-' : fmt0(mse));
  set('rmseValue', rmse == null ? '-' : fmt0(rmse));
  set('mapeValue', fmt2(mape));
  set('r2Value',   r2 == null ? '-' : Number(r2).toFixed(3));

  // kalau pakai grade:
  // const grade = r2==null ? '-' : (r2>=0.90?'A':r2>=0.85?'A-':r2>=0.80?'B+':r2>=0.75?'B':r2>=0.70?'B-':'C');
  // set('performanceGrade', grade);
}

/**
 * Try several slug variants and call /api/eval_summary?city=...
 * On success: call updateEvalCards(metrics) and optionally show label.
 * On fail : set cards to '-' (via updateEvalCards with null) and log.
 */

// Replace earlier multi-candidate function with this single-call version
async function fetchAndRenderEvalForCity(cityInput) {
  if (!cityInput) {
    updateEvalCards(null);
    return null;
  }

  try {
    console.debug("fetch eval single call for:", cityInput);
    const url = `/api/eval_summary?city=${encodeURIComponent(cityInput)}`;
    const resp = await fetch(url, { cache: 'no-store' });

    if (resp.status === 404) {
      console.info("eval not found (404) for:", cityInput);
      updateEvalCards(null);
      return null;
    }
    if (!resp.ok) {
      const t = await resp.text().catch(()=>resp.statusText);
      console.warn("eval fetch failed:", resp.status, t);
      updateEvalCards(null);
      return null;
    }

    const j = await resp.json();
    if (j && j.ok && j.metrics) {
      updateEvalCards(j.metrics);
      const labelEl = document.getElementById('evalCityLabel');
      if (labelEl && j.city) labelEl.textContent = j.city;
      return j;
    } else {
      // server might respond with metrics directly
      if (j && j.metrics) {
        updateEvalCards(j.metrics);
        return j;
      }
      updateEvalCards(null);
      return null;
    }
  } catch (err) {
    console.warn("fetchAndRenderEvalForCity error:", err);
    updateEvalCards(null);
    return null;
  }
}

// --- helper tambahan untuk HTML safety (supaya tidak error di innerHTML) ---
function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// --- fix: normalisasi data upload dari Flask agar chart bisa muncul ---
function normalizeFullResultForRender(result) {
  const predsObj = result.predictions || {};
  const predsArr = Object.values(predsObj)
    .filter(p => p && p.date && p.value != null)
    .sort((a,b) => new Date(a.date) - new Date(b.date));

  const pred_dates = predsArr.map(p => p.date);
  const pred_values = predsArr.map(p => Number(p.value));

  // fallback: kalau tidak ada trend (aktual), isi dengan prediksi juga
  const trend_values = result.trend?.values || pred_values;

  return {
    mode: 'full',
    city: result.city || '',
    n_total: result.n_total ?? null,
    test_days: result.test_days ?? null,
    metrics: result.metrics || {},
    pack_path: result.pack_path || null,
    predictions: predsObj,
    pred_series: {
      dates: pred_dates,
      actual: trend_values, // bisa pakai aktual kalau tersedia
      pred: pred_values
    },
    trend: {
      dates: pred_dates,
      values: trend_values
    },
    actual: pred_dates.map((d,i)=>({date:d,value:trend_values[i]})),
    predicted: pred_dates.map((d,i)=>({date:d,value:pred_values[i]}))
  };
}


// helper: ubah result (single) jadi payload yang renderResults paham
function normalizeFullResultForRender(result) {
  // result.predictions: { "1": {date, value}, "7": {...}, ... } (misal)
  const predsObj = result.predictions || {};
  // convert predsObj => arrays sorted by date
  const predsArr = Object.values(predsObj).filter(Boolean).slice().sort((a,b) => new Date(a.date) - new Date(b.date));
  const pred_dates = predsArr.map(p => p.date);
  const pred_values = predsArr.map(p => (p.value == null ? null : Number(p.value)));

  // construct a payload acceptable for renderResults
  const payload = {
    mode: 'full',
    city: result.city || '',
    n_total: result.n_total ?? null,
    test_days: result.test_days ?? null,
    metrics: result.metrics || {},
    pack_path: result.pack_path || null,
    // put predictions as object (same shape) so renderResults can pick them
    predictions: predsObj,
    // create pred_series/trend (trend = actual historical; here we don't have actual so use empty)
    pred_series: {
      dates: pred_dates,
      pred: pred_values
    },
    // For compatibility, also provide 'trend' as empty or same as pred_dates with nulls
    trend: {
      dates: pred_dates,
      values: Array(pred_values.length).fill(null)
    },
    // also provide 'predicted' & 'actual' arrays to match other branches
    actual: [], 
    predicted: predsArr.map(p => ({ date: p.date, value: Number(p.value) })),
  };
  return payload;
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


async function saveAndPredict() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput || !fileInput.files || !fileInput.files[0])
    return alert('Pilih file dulu');

  const file = fileInput.files[0];
  const saveBtn = document.getElementById('saveUploadBtn');
  const spinner = document.getElementById('uploadSpinner');
  const label = document.getElementById('uploadLabel');

  saveBtn.disabled = true;
  spinner.style.display = 'inline-block';
  label.textContent = 'Memproses...';

  try {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('mode', 'full'); // ubah ke 'quick' bila ingin cepat

    const resp = await fetchJsonSafe('/api/upload_file', {
      method: 'POST',
      body: fd
    });

    console.log('upload response (safe):', resp);

    if (!resp.ok) {
      const reason = resp.data ? JSON.stringify(resp.data) : (resp.text || `HTTP ${resp.status}`);
      throw new Error(reason);
    }

    // Parse JSON body
    let data = resp.data || null;
    if (!data && resp.text) {
      try { data = JSON.parse(resp.text); }
      catch (e) {}
    }

    // 🔹 Normalize nested shape
    if (data && data.data && typeof data.data === 'object') {
      data = { ...data, ...data.data };
    }

    if (!data) {
      console.warn('Server returned no JSON body; raw text:', resp.text);
      throw new Error('Server memberikan response tanpa JSON. Periksa Network tab / server log.');
    }

    console.log('upload data parsed:', data);

    // === HANDLERS ===
    if (data.mode === 'quick' || data.stats) {
      renderResults(data);
      return;
    }

    if (Array.isArray(data.results) && data.results.length > 0) {
  const first = data.results[0];
  console.log('upload first result:', first);

  // if server already provided trend or stats, render it directly
  if ((first.trend && Array.isArray(first.trend.dates) && Array.isArray(first.trend.values)) ||
      (first.stats && (first.stats.n_points || first.stats.n_points === 0))) {
    renderResults(first);
  } else if (typeof normalizeFullResultForRender === 'function') {
    // fallback: use normalizer if available
    const payloadForRender = normalizeFullResultForRender(first);
    renderResults(payloadForRender);
  } else {
    // last-resort: render root data (may still work)
    renderResults(data);
  }

    const ul = document.getElementById('uploadFullResultsList');
  if (ul) {
    ul.innerHTML = data.results.map(r => {
      const city = escapeHTML ? escapeHTML(r.city || '—') : (r.city || '—');
      if (r.ok) {
        // jika backend menambahkan r.pdf_url, tampilkan tombol unduh
        const reportButton = r.pdf_url
          ? `<a class="btn-download-report" href="${r.pdf_url}" target="_blank" rel="noopener">📥 Unduh Laporan</a>`
          : '';
        return `<li style="margin-bottom:0.4rem;">
          <b>${city}</b>: OK — best_r2=${r.best_r2 ?? '-'} ${reportButton}
        </li>`;
      } else {
        const reason = escapeHTML ? escapeHTML(r.reason ?? JSON.stringify(r)) : (r.reason ?? JSON.stringify(r));
        return `<li style="margin-bottom:0.4rem;"><b>${city}</b>: FAIL — ${reason}</li>`;
      }
    }).join('');
    document.getElementById('uploadFullSummary')?.style.removeProperty('display');
  }


  document.getElementById('uploadResults')?.style.removeProperty('display');
  return;
} else {
  renderResults(data);
  return;
}




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
   spinner.style.display = 'none';
  label.textContent = 'Simpan & Analisis';
  saveBtn.disabled = false;
  }
}



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
    // --- PREDICTION-ONLY CHART (safe: no date adapter required) ---
  let predChartLabels = [];
  let predValues = [];

  if (payload?.pred_series && Array.isArray(payload.pred_series.dates)) {
    predChartLabels = payload.pred_series.dates.slice();
    if (Array.isArray(payload.pred_series.pred)) {
      predValues = payload.pred_series.pred.map(v => (v == null ? null : Number(v)));
    } else {
      predValues = (payload.pred_series.actual || []).map(v => (v == null ? null : Number(v)));
    }
  } else if (payload?.predictions && typeof payload.predictions === 'object') {
    const predsArr = Object.values(payload.predictions || {})
      .filter(Boolean)
      .map(p => ({ date: p.date, value: (p.value == null ? null : Number(p.value)) }))
      .sort((a,b)=> new Date(a.date) - new Date(b.date));
    predChartLabels = predsArr.map(p => p.date);
    predValues = predsArr.map(p => p.value);
  } else if (Array.isArray(payload?.predicted)) {
    const arr = payload.predicted.map(p => ({ date: p.date, value: (p.value == null ? null : Number(p.value)) }))
      .sort((a,b)=> new Date(a.date) - new Date(b.date));
    predChartLabels = arr.map(p => p.date);
    predValues = arr.map(p => p.value);
  } else {
    // fallback 7 hari dari akhir trend
    const tdates = (trend && Array.isArray(trend.dates)) ? trend.dates.slice() : [];
    const tvals  = (trend && Array.isArray(trend.values)) ? trend.values.slice() : [];
    if (tdates.length) {
      const lastDate = new Date(tdates[tdates.length-1] + 'T00:00:00');
      for (let i=1;i<=7;i++){
        const d = new Date(lastDate);
        d.setDate(lastDate.getDate() + i);
        predChartLabels.push(d.toISOString().slice(0,10));
        predValues.push(tvals.length ? Number(tvals[tvals.length-1]) : null);
      }
    }
  }

  try {
    const canvas = document.getElementById('uploadPredChart');
    const ctx2 = canvas?.getContext?.('2d');
    if (!ctx2) throw new Error('Canvas #uploadPredChart tidak ditemukan');

    // Hancurkan chart lama (cara aman)
    try {
      // Chart.getChart tersedia di Chart.js v3+
      const existing = (typeof Chart.getChart === 'function') ? Chart.getChart(canvas) : window._uploadPredChart;
      if (existing && typeof existing.destroy === 'function') existing.destroy();
    } catch(e){
      // fallback: jika kita menyimpan global reference
      if (window._uploadPredChart && typeof window._uploadPredChart.destroy === 'function') {
        window._uploadPredChart.destroy();
      }
    }
    window._uploadPredChart = null;

    // Buat chart baru — pakai 'category' untuk sumbu X (tanggal string),
    // sehingga tidak membutuhkan adapter waktu eksternal.
    window._uploadPredChart = new Chart(ctx2, {
      type: 'line',
      data: {
        labels: predChartLabels || [],
        datasets: [{
          label: 'Prediksi',
          data: predValues || [],
          borderWidth: 2,
          tension: 0.25,
          pointRadius: 3,
          spanGaps: true
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                const v = ctx.parsed.y;
                if (v == null || Number.isNaN(Number(v))) return 'Tidak ada data';
                return 'Rp ' + new Intl.NumberFormat('id-ID').format(Math.round(Number(v)));
              }
            }
          }
        },
        scales: {
          x: {
            type: 'category', // <- penting: tidak perlu date adapter
            title: { display: true, text: 'Tanggal' },
            ticks: { maxRotation: 45, autoSkip: true }
          },
          y: {
            title: { display: true, text: 'Harga (Rp)' },
            ticks: {
              callback: (v) => v == null ? '' : 'Rp ' + new Intl.NumberFormat('id-ID').format(v)
            }
          }
        },
        interaction: { mode: 'index', intersect: false }
      }
    });
  } catch (e) {
    console.error('renderResults: pred-only chart error', e);
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
    // --- small: show download button for current payload if pdf_url present ---
  try {
    const resultsContainer = document.getElementById('uploadResults');
    const existing = document.getElementById('downloadEvalBtn');
    if (existing) existing.remove();

    if (payload && payload.pdf_url) {
      const a = document.createElement('a');
      a.id = 'downloadEvalBtn';
      a.href = payload.pdf_url;
      a.target = '_blank';
      a.rel = 'noopener';
      a.className = 'btn-download-report';
      a.textContent = '📥 Unduh Laporan Evaluasi (PDF)';
      // append di atas konten result (ubah jika mau posisi lain)
      resultsContainer.insertBefore(a, resultsContainer.firstChild);
    }
  } catch (e) {
    console.warn('Failed to render PDF download button', e);
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

// ====== SCROLL TO BERANDA FIX (universal, works meskipun 1 file) ======
document.addEventListener('click', (e) => {
  const btn = e.target.closest('#scroll-to-beranda');
  if (!btn) return;
  e.preventDefault();

  console.log('[scroll-to-beranda] clicked!');

  document.body.classList.add('hero-collapsed');
  document.querySelector('.nav-link[data-section="beranda"]')?.click();
  document.getElementById('beranda')?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  try { history.replaceState(null, '', '#beranda'); } catch (_) {}
});
