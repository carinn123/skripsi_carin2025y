
function _safeSetText(id, txt) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = txt;
}
async function populateQuickSelect(){ /* ... */ }
async function quickPredictFetchAndRender(entity, opts){ /* ... */ }

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
      quickPredictFetchAndRender(entity, { mode: 'real' });
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

let MAP_INSTANCE = null;
let GEOJSON_CACHE = null;
let geoLayer = null;

function ensureMap(){
  if(MAP_INSTANCE) return MAP_INSTANCE;
  MAP_INSTANCE = L.map('map', { preferCanvas: true }).setView([-2.5, 118], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OSM contributors' }).addTo(MAP_INSTANCE);
  return MAP_INSTANCE;
}
async function getProvGeoJSON(){
  if(GEOJSON_CACHE) return GEOJSON_CACHE;
  const url = '/static/indonesia_provinces.geojson'; // <- gunakan file yang kamu punya
  try {
    const res = await fetch(url, {cache:'no-store'});
    if(!res.ok){
      console.warn('getProvGeoJSON: failed to load', url, res.status);
      GEOJSON_CACHE = { type: "FeatureCollection", features: [] };
      return GEOJSON_CACHE;
    }
    GEOJSON_CACHE = await res.json();
    return GEOJSON_CACHE;
  } catch (e) {
    console.error('getProvGeoJSON error', e);
    GEOJSON_CACHE = { type: "FeatureCollection", features: [] };
    return GEOJSON_CACHE;
  }
}

let KABKOTA_CACHE = null;
async function getKabkotaPoints(){
  if(KABKOTA_CACHE) return KABKOTA_CACHE;
  const url = '/static/kabkota_points.geojson';
  try {
    const res = await fetch(url, {cache:'no-store'});
    if(!res.ok){ console.warn('getKabkotaPoints failed', res.status); KABKOTA_CACHE = null; return null; }
    KABKOTA_CACHE = await res.json();
    return KABKOTA_CACHE;
  } catch (e) { console.error('getKabkotaPoints error', e); KABKOTA_CACHE = null; return null; }
}

function setTableHeader(){
  const thead = document.getElementById('rc_table_head') || document.querySelector('#rc_table thead');
  if(!thead) return;
  thead.innerHTML = `<tr>
   <th style="width:56px">No</th>
    <th>Kabupaten/Kota</th>
    <th>Provinsi</th>
    <th>Pulau</th>
    <th class="num">Harga Terendah</th>
    <th class="num">Tanggal Terendah</th>
    <th class="num">Harga Tertinggi</th>
    <th class="num">Tanggal Tertinggi</th>
    <th class="num">Rata² Kota</th>
    <th class="num">Rata² Mode</th>
  </tr>`;
}

function populateFilterDropdownsFromRows(rows){
  const provSel = document.getElementById('rc_filter_province');
  const islSel = document.getElementById('rc_filter_island');
  if(!provSel || !islSel) return;
  const provSet = new Set(), islSet = new Set();
  rows.forEach(r => { if(r.province) provSet.add(String(r.province).trim()); if(r.island) islSet.add(String(r.island).trim()); });
  function fill(sel, arr, placeholder){
    sel.innerHTML = '';
    const ph = document.createElement('option'); ph.value=''; ph.textContent = placeholder; sel.appendChild(ph);
    arr.sort().forEach(v=>{ const o = document.createElement('option'); o.value=v; o.textContent=v; sel.appendChild(o); });
  }
  fill(provSel, Array.from(provSet), '— Semua Provinsi —');
  fill(islSel, Array.from(islSet), '— Semua Pulau —');
}

function applyTableFilters(){
  const q = (document.getElementById('rc_filter_text')?.value || '').trim().toLowerCase();
  const prov = (document.getElementById('rc_filter_province')?.value || '').trim().toLowerCase();
  const isl = (document.getElementById('rc_filter_island')?.value || '').trim().toLowerCase();
  const rows = document.querySelectorAll('#rc_table tbody tr');
  rows.forEach(tr=>{
    const city = (tr.dataset.city || '').toLowerCase();
    const p = (tr.dataset.prov || '').toLowerCase();
    const i = (tr.dataset.island || '').toLowerCase();
    let visible = true;
    if(q) visible = visible && (city.includes(q) || p.includes(q) || i.includes(q));
    if(prov) visible = visible && (p === prov);
    if(isl) visible = visible && (i === isl);
    tr.style.display = visible ? '' : 'none';
  });
  const shownEl = document.getElementById('rc_table_shown_count');
  if(shownEl) shownEl.textContent = Array.from(document.querySelectorAll('#rc_table tbody tr')).filter(r=> r.style.display !== 'none').length;
}

function attachTableFilters(){
  const q = document.getElementById('rc_filter_text');
  const prov = document.getElementById('rc_filter_province');
  const isl = document.getElementById('rc_filter_island');
  const clear = document.getElementById('rc_filter_clear');
  if(q) q.addEventListener('input', applyTableFilters);
  if(prov) prov.addEventListener('change', applyTableFilters);
  if(isl) isl.addEventListener('change', applyTableFilters);
  if(clear) clear.addEventListener('click', ()=>{
    if(q) q.value = '';
    if(prov) prov.value = '';
    if(isl) isl.value = '';
    applyTableFilters();
  });
}

function fillRegionTable(rows){
  console.log('fillRegionTable called, rows:', Array.isArray(rows) ? rows.length : typeof rows, rows && rows[0]);
  const tableEls = document.querySelectorAll('#rc_table');
  if(tableEls.length === 0){
    console.error('fillRegionTable: #rc_table not found in DOM!');
    return;
  }
  if(tableEls.length > 1){
    console.warn('fillRegionTable: multiple #rc_table found! Removing duplicates may help.', tableEls);
  }

  let tbody = document.querySelector('#rc_table tbody');
  if(!tbody){
    console.warn('fillRegionTable: tbody not found — creating one.');
    const t = document.getElementById('rc_table');
    const newTbody = document.createElement('tbody');
    t.appendChild(newTbody);
    tbody = document.querySelector('#rc_table tbody');
    if(!tbody){
      console.error('fillRegionTable: gagal membuat tbody');
      return;
    }
  }

  // defensive: if rows not array, clear and show message
  if(!rows || !Array.isArray(rows) || rows.length === 0){
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:var(--muted)">Tidak ada data untuk pilihan ini.</td></tr>`;
    const shownEl = document.getElementById('rc_table_shown_count');
    const totalEl = document.getElementById('rc_table_total_count');
    if(shownEl) shownEl.textContent = 0;
    if(totalEl && window.__LAST_RC_META) totalEl.textContent = window.__LAST_RC_META.count || 0;
    return;
  }

  // determine whether we need to compute island/national references on client
  // server may provide island_mean OR national_mean OR ref_mean; be flexible
  let needComputeRef = rows.some(r => (r.island_mean == null && r.national_mean == null && r.ref_mean == null));
  let islandMeans = {};
  let nationalMean = null;
  if(needComputeRef){
    const accum = Object.create(null);
    let sum = 0, count = 0;
    rows.forEach(r => {
      // row.mean was normalized earlier in many places; support mean, mean_value
      const meanRaw = (r.mean != null && !Number.isNaN(Number(r.mean))) ? Number(r.mean)
                     : (r.mean_value != null && !Number.isNaN(Number(r.mean_value))) ? Number(r.mean_value)
                     : null;
      const islandKey = (r.island != null && String(r.island).trim() !== '') ? String(r.island).trim() : 'unknown';

      if(meanRaw !== null){
        sum += meanRaw;
        count += 1;
        if(!accum[islandKey]) accum[islandKey] = { sum: 0, n: 0 };
        accum[islandKey].sum += meanRaw;
        accum[islandKey].n += 1;
      }
    });
    Object.keys(accum).forEach(k => {
      islandMeans[k] = (accum[k].n > 0) ? (accum[k].sum / accum[k].n) : null;
    });
    nationalMean = (count > 0) ? (sum / count) : null;
  }

  // Build HTML rows. Columns: no, kota, prov, pulau, min, max, mean_kota, ref_mean
  let html = '';
  rows.forEach((r, idx) => {
    const city = r.city ?? r.name ?? r.entity ?? '';
    const province = r.province ?? '';
    const island = r.island ?? 'unknown';

    const min_value = (r.min != null) ? rupiah(Math.round(Number(r.min))) : '—';
    const min_date = r.min_date ?? '—';
    const max_value = (r.max != null) ? rupiah(Math.round(Number(r.max))) : '—';
    const max_date = r.max_date ?? '—';

    // Prefer server-provided mean variants; support mean, mean_value
    const mean_raw = (r.mean != null && !Number.isNaN(Number(r.mean))) ? Number(r.mean)
                    : (r.mean_value != null && !Number.isNaN(Number(r.mean_value))) ? Number(r.mean_value)
                    : null;
    const mean_value = (mean_raw != null) ? rupiah(Math.round(mean_raw)) : '—';

    // reference value priority:
    // 1) island_mean (server)
    // 2) national_mean (server)
    // 3) ref_mean (legacy server field)
    // 4) fallback computed islandMeans/nationalMean on client
    let refNum = null;
    if (r.island_mean != null && !Number.isNaN(Number(r.island_mean))) {
      refNum = Number(r.island_mean);
    } else if (r.national_mean != null && !Number.isNaN(Number(r.national_mean))) {
      refNum = Number(r.national_mean);
    } else if (r.ref_mean != null && !Number.isNaN(Number(r.ref_mean))) {
      refNum = Number(r.ref_mean);
    } else {
      // use computed fallback if available
      const islKey = (island && String(island).trim() !== '') ? String(island).trim() : 'unknown';
      if (Object.prototype.hasOwnProperty.call(islandMeans, islKey) && islandMeans[islKey] != null) {
        refNum = islandMeans[islKey];
      } else if (nationalMean != null) {
        refNum = nationalMean;
      } else {
        refNum = null;
      }
    }
    const ref_value = (refNum != null) ? rupiah(Math.round(Number(refNum))) : '—';

    const no = r.no ?? (window.__LAST_RC_META ? ((window.__LAST_RC_META.page - 1) * window.__LAST_RC_META.page_size + idx + 1) : idx + 1);

    html += `<tr data-city="${escapeHtml(city).toLowerCase()}" data-prov="${escapeHtml(province).toLowerCase()}" data-island="${escapeHtml(island).toLowerCase()}">
      <td>${no}</td>
      <td>${escapeHtml(city)}</td>
      <td>${escapeHtml(province)}</td>
      <td>${escapeHtml(island)}</td>
      <td class="num">${min_value}</td>
      <td class="num">${min_date}</td>
      <td class="num">${max_value}</td>
      <td class="num">${max_date}</td>
      <td class="num">${mean_value}</td>
      <td class="num">${ref_value}</td>
    </tr>`;
  });
  tbody.innerHTML = html;

  // update count note (use meta cached globally if available)
  const shownEl = document.getElementById('rc_table_shown_count');
  const totalEl = document.getElementById('rc_table_total_count');
  if(window.__LAST_RC_META){
    if(shownEl) shownEl.textContent = (window.__LAST_RC_META.rows_shown || rows.length);
    if(totalEl) totalEl.textContent = (window.__LAST_RC_META.count || 0);
  } else {
    if(shownEl) shownEl.textContent = rows.length;
    if(totalEl) totalEl.textContent = rows.length;
  }

  // repopulate filter dropdowns and apply filters
  populateFilterDropdownsFromRows(rows);
  applyTableFilters();
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
function escapeHtml(s){
  if(!s && s !== 0) return '';
  return String(s).replace(/[&<>"']/g, (m)=> ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}
function renderPager(meta, onPage) {
  const container = document.getElementById('rc_table_pager');
  const shownEl = document.getElementById('rc_table_shown_count');
  const totalEl = document.getElementById('rc_table_total_count');
  if(!container) {
    console.warn('renderPager: container #rc_table_pager not found');
    return;
  }
  container.innerHTML = '';

  const total = meta && Number.isFinite(meta.count) ? meta.count : 0;
  const page = meta && meta.page ? Math.max(1, meta.page) : 1;
  const ps = meta && meta.page_size ? meta.page_size : 25;
  const totalPages = total > 0 ? Math.max(1, Math.ceil(total / ps)) : 1;
  const start = total === 0 ? 0 : ((page - 1) * ps) + 1;
  const end = Math.min(total, page * ps);

  if(shownEl) shownEl.textContent = (total === 0 ? 0 : (end - start + 1));
  if(totalEl) totalEl.textContent = total;

  // create buttons and assign classes
  const prev = document.createElement('button');
  prev.textContent = '◀ Prev';
  prev.disabled = page <= 1;
  prev.className = 'pager-btn';

  const next = document.createElement('button');
  next.textContent = 'Next ▶';
  next.disabled = page >= totalPages;
  next.className = 'pager-btn primary';

  const info = document.createElement('span');
  info.className = 'pager-info';
  info.style.margin = '0 8px';
  info.textContent = `Halaman ${page} / ${totalPages} • Menampilkan ${start}-${end} dari ${total}`;

  prev.addEventListener('click', () => onPage(Math.max(1, page - 1)));
  next.addEventListener('click', () => onPage(Math.min(totalPages, page + 1)));

  container.appendChild(prev);
  container.appendChild(info);
  container.appendChild(next);

  if(totalPages > 1){
    const gotoLabel = document.createElement('span');
    gotoLabel.style.marginLeft='8px';
    gotoLabel.textContent='Lompat ke:';

    const gotoInput = document.createElement('input');
    gotoInput.type='number';
    gotoInput.min=1;
    gotoInput.max=totalPages;
    gotoInput.value=page;
    gotoInput.className = 'pager-input';
    gotoInput.style.marginLeft='6px';

    const gotoBtn = document.createElement('button');
    gotoBtn.textContent='Go';
    gotoBtn.className = 'pager-btn';
    gotoBtn.addEventListener('click', () => {
      let v = parseInt(gotoInput.value || '1', 10);
      if(isNaN(v) || v < 1) v = 1;
      if(v > totalPages) v = totalPages;
      onPage(v);
    });

    container.appendChild(gotoLabel);
    container.appendChild(gotoInput);
    container.appendChild(gotoBtn);
  }
}

const CLIENT_PAGE_SIZE = 25;
const RC_CACHE_TTL = 1000 * 60 * 5;
const __RC_CACHE = {};



function normalizeProvName(s){
  if(!s && s !== 0) return '';
  let t = String(s).trim();
  t = t.replace(/\b(KAB\.?|KOTA\.?|KABUPATEN|KOTA)\b/ig, '').replace(/[.]/g,'').replace(/\s+/g,' ').trim();
  return t.toLowerCase();
}

async function showMapBeranda(event){
  const islandInput = document.getElementById('pulau');
  // allow empty island to mean "Semua Pulau"
  let island = islandInput ? islandInput.value.trim() : '';
  if(!island) island = 'Semua Pulau';

  const tahun = document.getElementById('b_tahun').value;
  const bulan = document.getElementById('b_bulan').value;
  const minggu = document.getElementById('b_minggu').value;
  const loading = document.getElementById('b_loading');
  const mode = (document.getElementById('b_mode') && document.getElementById('b_mode').value) || 'actual';

  // read bucket_scope UI (#b_scope), fallback auto:
  let bucket_scope = 'national';
  const scopeEl = document.getElementById('b_scope');
  if(scopeEl && scopeEl.value){
    bucket_scope = scopeEl.value;
  } else {
    bucket_scope = (island && island !== 'Semua Pulau') ? 'island' : 'national';
  }

  // basic validation: require tahun, but NOT pulau (we accept "Semua Pulau")
  if(!tahun){
    showMapValidation('Pilih Tahun terlebih dahulu.');
    return;
  }
  if(mode === 'predicted' && minggu && !bulan){
    showMapValidation('Jika memilih Minggu, silakan pilih Bulan terlebih dahulu.');
    return;
  }
  hideMapValidation();

  loading.style.display='inline-block';

  try{
    // include_table=1 so server will attempt to return per-city table for the same filters
    const url = `/api/choropleth?island=${encodeURIComponent(island)}&year=${encodeURIComponent(tahun)}&month=${encodeURIComponent(bulan)}&week=${encodeURIComponent(minggu)}&mode=${encodeURIComponent(mode)}&bucket_scope=${encodeURIComponent(bucket_scope)}&include_table=1`;

    const res = await fetch(url, { cache: 'no-store' });

    // robust read + parse with helpful logs
    const raw = await res.text();
    console.groupCollapsed('choropleth fetch debug');
    console.log('requested URL:', url);
    console.log('HTTP status:', res.status, res.statusText);
    console.log('raw response length:', raw ? raw.length : 0);
    console.log('raw response (first 1200 chars):', raw ? raw.slice(0,1200) : '');
    console.groupEnd();

    let js;
    try {
      js = raw ? JSON.parse(raw) : null;
    } catch (err) {
      console.error('[choropleth] JSON parse failed', err);
      showMapValidation('Gagal membaca response dari server (invalid JSON). Periksa console/server log.');
      fillRegionTable([]);
      if (window.geoLayer) try { window.geoLayer.remove(); } catch(_) {}
      throw err;
    }

    if (!res.ok) {
      const errMsg = (js && js.error) ? js.error : `HTTP ${res.status}`;
      console.error('choropleth returned error status', res.status, js);
      showMapValidation(`Server error: ${errMsg}`);
      fillRegionTable([]);
      throw new Error(errMsg);
    }

    console.groupCollapsed('choropleth parsed debug');
    console.log('parsed keys:', js && Object.keys(js));
    console.log('data len:', Array.isArray(js && js.data) ? js.data.length : js && js.data);
    console.log('table len:', Array.isArray(js && js.table) ? js.table.length : js && js.table);
    console.groupEnd();

    if(!js || (typeof js === 'object' && Object.keys(js).length === 0)){
      console.warn('choropleth: parsed payload empty, aborting');
      showMapValidation('Server mengembalikan payload kosong untuk filter ini.');
      fillRegionTable([]);
      return;
    }

    
    // === tampilkan rata-rata nasional per kota (rata_ratanasional) ===
// === tampilkan rata-rata nasional per kota (rata_ratanasional) ===
console.log('DEBUG: js.rata_ratanasional =>', js && js.rata_ratanasional);
const rataValServer = js && (js.rata_ratanasional ?? js.rata_ratanasional_value ?? js.rata_rata_nasional);
if (rataValServer != null) {
  const el = document.getElementById('legend_national_mean');
  const formatted = (typeof rupiah === 'function')
    ? 'Rp ' + rupiah(Math.round(rataValServer))
    : 'Rp ' + Math.round(rataValServer).toLocaleString('id-ID');
  if (el) {
    el.textContent = `Rata-rata Nasional (per-kota): ${formatted}`;
  } else {
    console.warn('legend_national_mean element not found');
    // optionally create it dynamically:
    const container = document.querySelector('.legend') || document.body;
    const p = document.createElement('p');
    p.id = 'legend_national_mean';
    p.style.marginTop = '8px';
    p.style.fontSize = '0.95rem';
    p.style.color = '#214';
    p.style.fontWeight = '600';
    p.innerHTML = `Rata-rata Nasional (per-kota): ${formatted}<br><small style="font-weight:400;color:#666">(dihitung dari rata-rata kota sesuai filter)</small>`;
    container.insertAdjacentElement('afterend', p);
  }
} else {
  // no value returned: keep default text / clear previous
  const el = document.getElementById('legend_national_mean');
  if (el) el.textContent = 'Rata-rata Nasional (per-kota): - (tidak ada data untuk filter)';
}


    const m = ensureMap();
    const gj = await getProvGeoJSON();

    // build vmap for choropleth painting
    const vmap = Object.fromEntries((js.data||[]).map(d=>[normProv(d.province), { val:d.value, cat:d.category, label:d.province, n_cities: d.n_cities || d.n_kota || 0 }]));

    if(window.geoLayer) window.geoLayer.remove();
    window.geoLayer = L.geoJSON(gj, {
      style: f => {
        const rawn = f.properties.Propinsi || f.properties.PROVINSI || f.properties.provinsi || f.properties.name || f.properties.NAMOBJ || "";
        const rec = vmap[normProv(rawn)];
        const fill = rec ? (rec.cat==='T1' ? '#2ecc71' : rec.cat==='T2' ? '#f1c40f' : '#e74c3c') : '#bdc3c7';
        return { color:'#fff', weight:1, fillColor:fill, fillOpacity:.85 };
      },
      onEachFeature: (feature, layer) => {
        const rawn = feature.properties.Propinsi || feature.properties.PROVINSI || feature.properties.provinsi || feature.properties.name || feature.properties.NAMOBJ || "—";
        const rec = vmap[normProv(rawn)];
        const valueRaw = rec ? (('val' in rec) ? rec.val : (rec.value ?? null)) : null;
        const val = (valueRaw == null) ? null : Math.round(Number(valueRaw));
        const cat = rec ? (rec.cat || rec.category || 'no-data') : 'no-data';
        let meta = '';
        // if (js.mode === 'predicted' || js.generated_at || js.model_version) {
        //   // const gen = js.generated_at ? `Generated: ${js.generated_at}` : '';
        //   const mv  = js.model_version ? `Model: ${js.model_version}` : '';
        //   // meta = `<br/><small style="color:#6b7280">${[gen, mv].filter(Boolean).join(' • ')}</small>`;
        // }
        const nCitiesInfo = rec && rec.n_cities ? `<br/><small>Jumlah kota: ${rec.n_cities}</small>` : '';
        layer.bindPopup(`<b>${rawn}</b><br/>${val ? ('Rp '+rupiah(val)) : '—'}<br/>Kategori: ${escapeHtml(cat)}${nCitiesInfo}`);

        layer.on('click', function(e){
          const provName = (feature.properties.Propinsi || feature.properties.PROVINSI || feature.properties.provinsi || feature.properties.name || feature.properties.NAMOBJ || "").trim();
          if(!provName) return;
          if(typeof loadRegionSummaryForProvince === 'function'){
             const tahun = document.getElementById('b_tahun')?.value || '';
            const bulan = document.getElementById('b_bulan')?.value || '';
            const minggu = document.getElementById('b_minggu')?.value || '';
            const mode = (document.getElementById('b_mode') && document.getElementById('b_mode').value) || 'actual';
            loadRegionSummaryForProvince(provName, 1, { year: tahun, month: bulan, week: minggu, mode }).catch(err => console.error('loadRegionSummaryForProvince', err));

          }
        }
      
      );
      }
    }).addTo(m);

    try { m.fitBounds(window.geoLayer.getBounds(), {padding:[20,20]}); } catch(e){}

    // === TABLE: map server table rows to expected fillRegionTable format
    // If server didn't include ref_mean, compute fallback on client
    let tableRows = [];
   
    if(tableRows.length > 300){
      showMapValidation(`Menampilkan ${tableRows.length} baris untuk "Semua Pulau" — mungkin butuh beberapa detik untuk render.` );
      setTimeout(hideMapValidation, 4000);
    }

    fillRegionTable(tableRows);

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
        document.getElementById('statLast').textContent = `Predicted: ${d.toLocaleDateString('id-ID', opt)}`;
      }catch(e){
        document.getElementById('statLast').textContent = `Predicted: ${js.generated_at.split('T')[0]}`;
      }
    } else {
      document.getElementById('statLast').textContent = js.last_actual || '-';
    }

  }catch(e){
    console.error(e);
    alert('Gagal menampilkan peta & tabel: '+(e.message||e));
    fillRegionTable([]);
  }finally{
    loading.style.display='none';
  }
}
window.showMapBeranda = showMapBeranda;

/* ---------- init wiring (filters) ---------- */
document.addEventListener('DOMContentLoaded', ()=>{
  setTableHeader();
  attachTableFilters();
  const btn = document.getElementById('btn_show_map');
  if(btn) btn.addEventListener('click', showMapBeranda);

  // enable/disable minggu based on bulan
  const bulan = document.getElementById('b_bulan');
  const minggu = document.getElementById('b_minggu');
  if(bulan && minggu){
    bulan.addEventListener('change', ()=> { if(bulan.value) minggu.disabled = false; else { minggu.value=''; minggu.disabled=true; } });
  }
});

async function fetchJsonWithRetry(url, tries = 2) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, { cache: 'no-store' });
      const text = await res.text(); // ambil text mentah
      if (!text || !text.trim()) {
        console.warn(`fetchJsonWithRetry: Empty response body on try ${i + 1}`);
        if (i < tries - 1) await new Promise(r => setTimeout(r, 300));
        continue;
      }

      try {
        const data = JSON.parse(text);
        if (!res.ok) {
          console.warn(`fetchJsonWithRetry: HTTP ${res.status} but parsed JSON`, data);
        }
        return data; // sukses
      } catch (e) {
        console.warn(`fetchJsonWithRetry: JSON parse error on try ${i + 1}`, e, text.slice(0, 300));
        if (i < tries - 1) await new Promise(r => setTimeout(r, 300));
        continue;
      }
    } catch (e) {
      console.error(`fetchJsonWithRetry: Fetch failed on try ${i + 1}`, e);
      if (i < tries - 1) await new Promise(r => setTimeout(r, 300));
    }
  }

  console.error('fetchJsonWithRetry: All tries failed', url);
  return null;
}

async function loadRegionSummaryFromMap(island, page = 1){
  console.log('loadRegionSummaryFromMap called', { island, page });
  const tbody = document.querySelector('#rc_table tbody');
  const loader = document.getElementById('rc_loading');
  if(loader) loader.style.display = 'inline-block';
  try{
    if(!island){
      fillRegionTable([]);
      renderPager({count:0,page:1,page_size:25}, ()=>{});
      return;
    }

    const key = `island:${island}:page:${page}`;
    const now = Date.now();
    if(__RC_CACHE[key] && (now - __RC_CACHE[key].ts) < RC_CACHE_TTL){
      const js = __RC_CACHE[key].data;
      console.log('rc cache hit', key, js);
      // ensure global meta exists for fillRegionTable
      window.__LAST_RC_META = { count: js.count || 0, page: js.page || page, page_size: js.page_size || 25, rows_shown: (js.rows||[]).length };
      fillRegionTable(js.rows || []);
      renderPager({count: js.count || 0, page: js.page || page, page_size: js.page_size || 25}, (p)=> loadRegionSummaryFromMap(island, p));
      return;
    }

    // request page_size 25 (keep consistent)
    const req_page_size = 25;
    const url = `/api/region_summary?mode=island&value=${encodeURIComponent(island)}&page=${page}&page_size=${req_page_size}`;
    console.log('fetch region_summary', url);
  
    const js = await fetchJsonWithRetry(url);
    if (!js) {
      console.warn('fetchJsonWithRetry returned null for', url);
      fillRegionTable([]);
      renderPager({count:0,page:1,page_size:req_page_size}, ()=>{});
      return;
    }
    // js sudah berisi parsed JSON (atau object); log untuk debug
    console.log('region_summary response', js);

    const rows = js.rows || [];
    const total = Number.isFinite(js.count) ? js.count : (rows.length || 0);
    const page_ret = js.page || page;
    const page_size = js.page_size || req_page_size;

    // store global meta for fillRegionTable (important)
    window.__LAST_RC_META = { count: total, page: page_ret, page_size: page_size, rows_shown: rows.length };

    __RC_CACHE[key] = { ts: now, data: { rows, count: total, page: page_ret, page_size } };

    fillRegionTable(rows);
    renderPager({count: total, page: page_ret, page_size}, (p)=> loadRegionSummaryFromMap(island, p));

  }catch(err){
    console.error('loadRegionSummaryFromMap error', err);
    fillRegionTable([]);
    renderPager({count:0,page:1,page_size:25}, ()=>{});
  }finally{
    if(loader) loader.style.display = 'none';
  }
}
async function loadRegionSummaryForProvince(prov, page = 1, opts = {}){
  const tbody = document.querySelector('#rc_table tbody');
  const loader = document.getElementById('rc_loading');
  if(loader) loader.style.display = 'inline-block';

  try{
    if(!prov){
      fillRegionTable([]);
      renderPager({count:0,page:1,page_size:25}, ()=>{});
      return;
    }

    const tahun = opts.year ?? (document.getElementById('b_tahun')?.value || '');
    const bulan = opts.month ?? (document.getElementById('b_bulan')?.value || '');
    const minggu = opts.week ?? (document.getElementById('b_minggu')?.value || '');
    const mode = opts.mode ?? ((document.getElementById('b_mode') && document.getElementById('b_mode').value) || 'actual');

    const key = `province:${prov}:page:${page}:y${tahun}:m${bulan}:w${minggu}:mode${mode}`;
    const now = Date.now();
    if(__RC_CACHE[key] && (now - __RC_CACHE[key].ts) < RC_CACHE_TTL){
      const js = __RC_CACHE[key].data;
      console.log('rc cache hit', key, js);
      window.__LAST_RC_META = { count: js.count || 0, page: js.page || page, page_size: js.page_size || 25, rows_shown: (js.rows||[]).length };
      fillRegionTable(js.rows || []);
      renderPager({count: js.count || 0, page: js.page || page, page_size: js.page_size || 25}, (p)=> loadRegionSummaryForProvince(prov, p));
      return;
    }

    const req_page_size = 25;
    const url = `/api/region_summary?mode=province&value=${encodeURIComponent(prov)}&page=${page}&page_size=${req_page_size}`
                + `&year=${encodeURIComponent(tahun)}&month=${encodeURIComponent(bulan)}&week=${encodeURIComponent(minggu)}&predict=${mode === 'predicted' ? '1' : '0'}`;
    console.log('fetch region_summary province', url);

    // => replaced fetch(...) with fetchJsonWithRetry(...)
    const js = await fetchJsonWithRetry(url);
    if (!js) {
      console.warn('fetchJsonWithRetry returned null for', url);
      fillRegionTable([]);
      renderPager({count:0,page:1,page_size:req_page_size}, ()=>{});
      return;
    }
    console.log('region_summary(province) response (parsed)', js);

    // if backend returns an explicit error structure, handle it
    if (js && js.error) {
      console.error('region_summary (province) error from server', js.error);
      fillRegionTable([]);
      renderPager({count:0,page:1,page_size:req_page_size}, ()=>{});
      return;
    }

    const rows = js.rows || [];
    const total = Number.isFinite(js.count) ? js.count : (rows.length || 0);
    const page_ret = js.page || page;
    const page_size = js.page_size || req_page_size;

    window.__LAST_RC_META = { count: total, page: page_ret, page_size: page_size, rows_shown: rows.length };

    __RC_CACHE[key] = { ts: now, data: { rows, count: total, page: page_ret, page_size } };

    fillRegionTable(rows);
    renderPager({count: total, page: page_ret, page_size}, (p)=> loadRegionSummaryForProvince(prov, p));

  }catch(err){
    console.error('loadRegionSummaryForProvince error', err);
    fillRegionTable([]);
    renderPager({count:0,page:1,page_size:25}, ()=>{});
  }finally{
    if(loader) loader.style.display = 'none';
  }
}


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

function getTestCutoff() {
  try {
    const el = document.getElementById('prediksi');
    if (el && el.dataset && el.dataset.testCutoff) return el.dataset.testCutoff;
  } catch(e){}
  return '2025-07-01'; // fallback
}

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
      color: '#333',
      font: { size: 12 }
    }
  },
  tooltip: {
    enabled: true,
    backgroundColor: 'rgba(0,0,0,0.8)',
    titleFont: { size: 13, weight: 'bold' },
    bodyFont: { size: 12 },
    padding: 10,
    callbacks: {
      label: function(context) {
        const val = context.parsed.y;
        return 'Rp ' + new Intl.NumberFormat('id-ID').format(Math.round(val));
      },
      title: function(context) {
        const label = context[0].label;
        return `Tanggal: ${label}`;
      }
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


async function quickPredictFetchAndRender(entitySlug, opts = { mode: 'real' }) {
  if (!entitySlug) return;
  const loading = document.getElementById('quickPredLoading');
  const resultBox = document.getElementById('quickPredResult');

  const elTodayVal = document.getElementById('quickTodayValue');
  const elTodayDate = document.getElementById('quickTodayDate');
  const elTomorrowVal = document.getElementById('quickTomorrow');
  const elLusaVal = document.getElementById('quicklusa');
  const elLusaDate = document.getElementById('quicklusaDate');
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
    const url = `/api/quick_predict?city=${encodeURIComponent(entitySlug)}&mode=${encodeURIComponent(opts.mode||'real')}`;
    const res = await fetch(url);
    if (!res.ok) {
      const t = await res.text().catch(()=>res.statusText);
      throw new Error(`${res.status} ${t}`);
    }
    const j = await res.json();
    if (!j.ok) throw new Error(j.error || 'no ok');

    // fill main today (last actual)
    // === bagian fill main today (diperbaiki) ===
if (j.last_actual || j.last_value != null) {
  const actualDate = j.last_actual ? shortDate(j.last_actual) : '—';
  // tulis teks lengkap langsung ke quickTodayDate
  elTodayDate.textContent = `Harga aktual terakhir ${actualDate} : `;
  // lalu isi nilai ke span quickTodayValue
  elTodayVal.textContent = fmtMoney(j.last_value);
  // jika span belum berada di dalam elTodayDate, masukkan; kalau sudah, skip
  if (!elTodayDate.contains(elTodayVal)) elTodayDate.appendChild(elTodayVal);
} else {
  elTodayDate.textContent = 'Harga aktual terakhir — : ';
  elTodayVal.textContent = '-';
  if (!elTodayDate.contains(elTodayVal)) elTodayDate.appendChild(elTodayVal);
}


    // fill horizons
    if (j.predictions && j.predictions["1"]) {
      elTomorrowVal.textContent = fmtMoney(j.predictions["1"].value);
      elTomorrowDate.textContent = shortDate(j.predictions["1"].date);
    } else {
      if (elTomorrowVal) elTomorrowVal.textContent = '-';
      if (elTomorrowDate) elTomorrowDate.textContent = '';
    }

    if (j.predictions && j.predictions["2"]) {
      if (elLusaVal) elLusaVal.textContent = fmtMoney(j.predictions["2"].value);
      if (elLusaDate) elLusaDate.textContent = shortDate(j.predictions["2"].date);
    } else {
      if (elLusaVal) elLusaVal.textContent = '-';
      if (elLusaDate) elLusaDate.textContent = '';
    }

    if (j.predictions && j.predictions["7"]) {
      if (el7Val) el7Val.textContent = fmtMoney(j.predictions["7"].value);
      if (el7Date) el7Date.textContent = shortDate(j.predictions["7"].date);
    } else {
      if (el7Val) el7Val.textContent = '-';
      if (el7Date) el7Date.textContent = '';
    }

    if (j.predictions && j.predictions["10"]) {
      if (el30Val) el30Val.textContent = fmtMoney(j.predictions["10"].value);
      if (el30Date) el30Date.textContent = shortDate(j.predictions["10"].date);
    } else {
      if (el30Val) el30Val.textContent = '-';
      if (el30Date) el30Date.textContent = '';
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

      // append predicted points in chronological order (1,2,7,10)
      const predsArr = [];
      for (const k of ['1','2','7','10']) {
        if (j.predictions && j.predictions[k]) predsArr.push(j.predictions[k]);
      }
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
            options: {
              plugins: { legend: { display: false } },
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: {
                  display: true,
                  title: { display: true, text: 'Tanggal', color: '#111', font: { size: 12, weight: '600' } },
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
                },
                y: {
                  display: true,
                  title: { display: true, text: 'Harga (Rp)', color: '#111', font: { size: 12, weight: '600' } },
                  ticks: {
                    color: '#444',
                    callback: v => 'Rp ' + new Intl.NumberFormat('id-ID').format(Math.round(v))
                  },
                  grid: { color: '#eee' }
                }
              }
            }
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
      if (document.getElementById('quickTodayValue')) document.getElementById('quickTodayValue').textContent = 'Gagal memuat';
      if (document.getElementById('quickTodayDate')) document.getElementById('quickTodayDate').textContent = '';
    }
    return null;
  }
}

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

// Prefer runtime eval returned inside /api/predict_range (mode=real)
    try {
      if ((mode || '').toString().toLowerCase() === 'real' && r && r.eval && r.eval.runtime) {
        // r.eval.runtime is expected: { mae, mse, rmse, mape, r2, n }
        updateEvalCards(r.eval.runtime);
        const labelEl = document.getElementById('evalCityLabel');
        if (labelEl) labelEl.textContent = (r.entity || r.city || citySlug).replace(/_/g,' ');
      } else {
        // fallback: call eval_summary endpoint (will use mode param)
        await fetchAndRenderEvalForCity(citySlug, mode);
      }
    } catch (err) {
      console.warn("render runtime eval failed, falling back to /api/eval_summary:", err);
      await fetchAndRenderEvalForCity(citySlug, mode);
    }

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

function fmtID(x, dec=0){
  if (x==null || Number.isNaN(x)) return '-';
  return new Intl.NumberFormat('id-ID', {maximumFractionDigits: dec, minimumFractionDigits: dec}).format(x);
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

async function fetchAndRenderEvalForCity(cityInput, mode = 'test') {
  if (!cityInput) {
    updateEvalCards(null);
    return null;
  }

  try {
    console.debug("fetch eval single call for:", cityInput, "mode:", mode);
    let url = `/api/eval_summary?city=${encodeURIComponent(cityInput)}`;
    if (mode && mode.toString().toLowerCase() === 'real') {
      url += '&mode=real';
    }

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

    // Prioritize runtime evaluation if server returned it (various server shapes handled)
    // Possible server shapes:
    // 1) { ok: true, runtime: { entity..., runtime_eval: {...} }, excel: {...} }
    // 2) { ok: true, runtime: { runtime_eval: {...} }, ... }
    // 3) { ok: true, city: ..., slug: ..., metrics: {...} } (old Excel-only)
    // 4) { metrics: {...} } (other shape)
    let runtimeEval = null;
    if (j && j.mode === 'real' && j.runtime && j.runtime.runtime_eval) {
      runtimeEval = j.runtime.runtime_eval;
    } else if (j && j.runtime && j.runtime.runtime_eval) {
      runtimeEval = j.runtime.runtime_eval;
    } else if (j && j.runtime && j.runtime.runtime_eval === undefined && j.runtime.runtime_eval === null && j.runtime && j.runtime.runtime_eval) {
      // no-op: keep it defensive
    } else if (j && j.runtime && j.runtime.runtime_eval === undefined && j.runtime.runtime_eval === undefined && j.runtime) {
      // handle older variant where runtime is the eval object directly
      if (j.runtime && j.runtime.runtime_eval == null && j.runtime.runtime_eval !== undefined) {
        // nothing
      }
    }

    // Another possible server shape: { runtime: { entity, last_actual, runtime_eval } }
    if (!runtimeEval && j && j.runtime && j.runtime.runtime_eval) runtimeEval = j.runtime.runtime_eval;
    // Or server might return runtime directly as runtime object
    if (!runtimeEval && j && j.runtime && j.runtime.mae) runtimeEval = j.runtime;

    // If runtimeEval present, render it; else fallback to Excel metrics (j.metrics or j)
    if (runtimeEval) {
      updateEvalCards(runtimeEval);
      const labelEl = document.getElementById('evalCityLabel');
      if (labelEl && (j.city || (j.runtime && j.runtime.entity))) {
        labelEl.textContent = j.city || j.runtime.entity;
      }
      return j;
    }

    // fallback to Excel metrics (old behavior)
    if (j && j.ok && j.metrics) {
      updateEvalCards(j.metrics);
      const labelEl = document.getElementById('evalCityLabel');
      if (labelEl && j.city) labelEl.textContent = j.city;
      return j;
    } else if (j && j.metrics) {
      updateEvalCards(j.metrics);
      return j;
    } else if (j && j.excel && j.excel.metrics) {
      updateEvalCards(j.excel.metrics);
      return j;
    }

    updateEvalCards(null);
    return null;

  } catch (err) {
    console.warn("fetchAndRenderEvalForCity error:", err);
    updateEvalCards(null);
    return null;
  }
}


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
async function getNearestEntity(lat, lon) {
  // load JSON keyed object
  const res = await fetch('/static/city_coords.json');
  const data = await res.json();

  // convert jadi array of [slug, {lat, lng, label}]
  const entities = Object.entries(data);

  function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const toRad = (v) => v * Math.PI / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat/2)**2 + Math.cos(toRad(lat1))*Math.cos(toRad(lat2))*Math.sin(dLon/2)**2;
    return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  let best = null;
  let bestDist = Infinity;

  for (const [slug, e] of entities) {
    const d = haversineKm(lat, lon, e.lat, e.lng);
    if (d < bestDist) {
      bestDist = d;
      best = { slug, name: e.label, dist: d };
    }
  }

  return best;
}

document.getElementById('scroll-to-beranda')?.addEventListener('click', () => {
  // 1️⃣ collapse hero & scroll ke beranda
  document.body.classList.add('hero-collapsed');
  document.querySelector('.nav-link[data-section="beranda"]')?.click();
  document.getElementById('beranda')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  history.replaceState(null, '', '#beranda');

  // 2️⃣ minta izin lokasi user
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(async (pos) => {
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      console.log("Lokasi user:", lat, lon);

      // 3️⃣ load daftar koordinat kota
      const res = await fetch('/static/city_coords.json');
      const data = await res.json();
      const entities = Object.entries(data);

      // hitung kota terdekat
      function haversineKm(lat1, lon1, lat2, lon2) {
        const R = 6371;
        const toRad = (v) => v * Math.PI / 180;
        const dLat = toRad(lat2 - lat1);
        const dLon = toRad(lon2 - lon1);
        const a = Math.sin(dLat/2)**2 + Math.cos(toRad(lat1))*Math.cos(toRad(lat2))*Math.sin(dLon/2)**2;
        return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      }

      let best = null, bestDist = Infinity;
      for (const [slug, e] of entities) {
        const d = haversineKm(lat, lon, e.lat, e.lng);
        if (d < bestDist) {
          bestDist = d;
          best = { slug, label: e.label, dist: d };
        }
      }

      // 💥 di sini nih potongan kamu masuk
      const sel = document.getElementById('quick_kabupaten');
      const info = document.getElementById('auto-location');

      if (best) {
        console.log(`Kota terdekat: ${best.label} (${best.slug}) [${bestDist.toFixed(1)} km]`);

        // set dropdown
        if (sel) {
          sel.value = best.slug;
          sel.dispatchEvent(new Event('change'));
        }

        // tampilkan teks lokasi
        if (info) {
          info.textContent = `Lokasi terdekat Anda berdasarkan Daftar Kota terdeteksi di ${best.label}`;
        }

        // panggil API untuk prediksi otomatis (opsional)
        fetch(`/api/quick_predict?city=${best.slug}&mode=real`)
          .then(r => r.json())
          .then(data => console.log("Prediksi otomatis:", data))
          .catch(err => console.error("Error prediksi otomatis:", err));
      }
    }, (err) => {
      console.warn("User menolak lokasi:", err);
    });
  }
});


async function initQuickSelectTom() {
  const sel = document.getElementById('quick_kabupaten');
  if (!sel) return;

  // fetch city list
  let cities = [];
  try {
    const resp = await fetch('/api/cities_full', { cache: 'no-store' });
    cities = await resp.json();
  } catch (e) {
    console.warn('failed load cities_full', e);
    cities = [];
  }

  // fill select
  sel.innerHTML = '<option value="">-- Pilih --</option>';
  cities.forEach(item => {
    const slug = item.entity || item.slug || (item.label||'').toLowerCase().replace(/\s+/g,'_');
    const label = item.label || slug;
    const o = document.createElement('option'); o.value = slug; o.textContent = label;
    sel.appendChild(o);
  });

  // init TomSelect (search inside dropdown)
  let ts = null;
  try {
    ts = new TomSelect(sel, {
      allowEmptyOption: true,
      maxOptions: 100,
      hideSelected:true,
      searchField: ['text'],
      placeholder: '-- Pilih atau ketik untuk mencari --',
      // keep dropdown attached to body to avoid overflow issues (optional)
      // dropdownParent: 'body',
      render: {
        option: function(data, escape) {
          return '<div>' + escape(data.text) + '</div>';
        }
      }
    });
  } catch (e) {
    console.warn('TomSelect init failed', e);
  }

  // expose instance for programmatic set (if needed elsewhere)
  sel._ts = ts || null;
}

// call on DOM ready
document.addEventListener('DOMContentLoaded', initQuickSelectTom);

/* ---------- auto-detect location & programmatically set dropdown ---------- */

async function autoDetectLocationAndSelect() {
  const sel = document.getElementById('quick_kabupaten');
  if (!sel) return;

  // pastikan opsi sudah dimuat
  if (!sel.options.length || sel.options.length <= 1) {
    await initQuickSelectTom();
    await new Promise(r => setTimeout(r, 120));
  }

  // pastikan geolocation support
  if (!navigator.geolocation) {
    console.warn('Geolocation tidak didukung browser ini.');
    return;
  }

  // ambil posisi user
  navigator.geolocation.getCurrentPosition(async (pos) => {
    const lat = pos.coords.latitude;
    const lon = pos.coords.longitude;

    try {
      const resp = await fetch('/static/city_coords.json', { cache: 'no-store' });
      if (!resp.ok) throw new Error('File city_coords.json tidak ditemukan.');
      const data = await resp.json();

      // hitung jarak terdekat
      const toRad = deg => deg * Math.PI / 180;
      const hav = (lat1, lon1, lat2, lon2) => {
        const R = 6371;
        const dLat = toRad(lat2 - lat1);
        const dLon = toRad(lon2 - lon1);
        const a = Math.sin(dLat / 2) ** 2 +
                  Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
        return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      };

      let best = null;
      let bestDist = Infinity;
      for (const [slug, city] of Object.entries(data)) {
        if (!city.lat || !city.lng) continue;
        const d = hav(lat, lon, city.lat, city.lng);
        if (d < bestDist) {
          bestDist = d;
          best = { slug, label: city.label, dist: d };
        }
      }

      const info = document.getElementById('auto-location');

      // kalau lokasi ditemukan
      if (best) {
        // update info teks
        if (info) info.textContent = `📍 Lokasi terdekat Anda terdeteksi di ${best.label}`;

        // pastikan dropdown diisi otomatis
        if (sel._ts) {
          sel._ts.setValue(best.slug);
        } else {
          sel.value = best.slug;
        }

        // trigger event change biar langsung load prediksi
        sel.dispatchEvent(new Event('change'));
      } else {
        // kalau tidak ketemu kota terdekat
        if (info) info.textContent = "📍 Tidak dapat mendeteksi lokasi Anda. Silakan pilih manual.";
      }

    } catch (err) {
      console.warn("autoDetectLocationAndSelect error:", err);
      const info = document.getElementById('auto-location');
      if (info) info.textContent = "📍 Tidak dapat mendeteksi lokasi Anda. Silakan pilih manual.";
    }
  }, (err) => {
    console.warn('Gagal mendeteksi lokasi:', err);
    const info = document.getElementById('auto-location');
    if (info) info.textContent = "📍 Tidak dapat mendeteksi lokasi Anda. Silakan pilih manual.";
  }, { timeout: 10000 });
}


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
// Pastikan ini dijalankan setelah DOM siap (file Anda sudah memuat script dengan defer)
document.addEventListener('click', (e) => {
  // existing scroll-to-beranda handler (biarkan tetap)
  const btn = e.target.closest('#scroll-to-beranda');
  if (btn) {
    e.preventDefault();
    console.log('[scroll-to-beranda] clicked!');
    document.body.classList.add('hero-collapsed');
    document.querySelector('.nav-link[data-section="beranda"]')?.click();
    document.getElementById('beranda')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    try { history.replaceState(null, '', '#beranda'); } catch (_) {}
    return;
  }

  // BRAND -> kembali ke COVER
  const brand = e.target.closest('#brand-link');
  if (brand) {
    e.preventDefault();
    console.log('[brand-link] clicked! kembali ke cover');

    // 1) tutup hamburger jika terbuka (checkbox nav-toggle)
    const navToggle = document.getElementById('nav-toggle');
    if (navToggle && navToggle.checked) {
      navToggle.checked = false;
    }

    // 2) kembalikan state cover (hilangkan collapsed)
    document.body.classList.remove('hero-collapsed');

    // 3) reset active nav (opsional: hilangkan semua active)
    document.querySelectorAll('.top-nav .nav-link').forEach(link => link.classList.remove('active'));

    // 4) scroll ke #cover (jika ada)
    const coverEl = document.getElementById('cover');
    if (coverEl) {
      // beri sedikit delay jika ada animasi collapse yang perlu selesai
      setTimeout(() => {
        coverEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 50);
    } else {
      // fallback ke top
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // 5) update url hash (opsional)
    try { history.replaceState(null, '', '#cover'); } catch (_) {}

    return;
  }
});

