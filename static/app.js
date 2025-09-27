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
function getSelectedCityLabel(){ // dipakai mode tren single lama
  const sel=document.getElementById('kabupaten');
  const opt=sel?.options[sel.selectedIndex];
  return opt?.dataset?.label || sel?.value || "";
}

// ===== Inisialisasi Cities (semua dropdown yang ada)
async function initCities(){
  const targetIds = ['kabupaten','kabupaten2','kabupaten_a','kabupaten_b']
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

async function loadData(){
  const selA = document.getElementById('kabupaten');
  const selB = document.getElementById('kabupaten2');
  const cityA = selectedSlugOrLabel(selA);
  const cityB = selectedSlugOrLabel(selB);

  const tahun = document.getElementById('tahun').value;
  const bulan = document.getElementById('bulan').value;
  const minggu= document.getElementById('minggu').value;
  const loading = document.getElementById('loadingSpinner');

  if(!cityA || !tahun){ alert('Pilih minimal Kota/Kab 1 dan Tahun.'); return; }

  const q = (c)=> `/api/trend?city=${encodeURIComponent(c)}&year=${encodeURIComponent(tahun)}&month=${encodeURIComponent(bulan)}&week=${encodeURIComponent(minggu)}`;

  loading.style.display='inline-block';
  try{
    // fetch kota A (wajib)
    const resA = await fetch(q(cityA)); const trA = await resA.json();
    if(!resA.ok) throw new Error(trA.error||'Server error');

    if (cityB) {
      // fetch kota B (opsional)
      const resB = await fetch(q(cityB)); const trB = await resB.json();
      if(!resB.ok) throw new Error(trB.error||'Server error');

      // gabungkan label
      const labels = Array.from(new Set([
        ...(trA.series||[]).map(s=>s.label),
        ...(trB.series||[]).map(s=>s.label),
      ])).sort((a,b)=>{
        // coba urutkan numerik bila bisa (bulan/minggu), else lexicographic
        const na = parseInt(a.replace(/\D+/g,''))||0;
        const nb = parseInt(b.replace(/\D+/g,''))||0;
        return (na-nb) || a.localeCompare(b);
      });

      const mA = _seriesMap(trA.series||[]);
      const mB = _seriesMap(trB.series||[]);
      const vA = labels.map(l => mA.has(l) ? mA.get(l) : null);
      const vB = labels.map(l => mB.has(l) ? mB.get(l) : null);

      renderTrendChartCompare(
        labels, vA, vB,
        (trA.entity||trA.city||'Kota 1').replace(/_/g,' '),
        (trB.entity||trB.city||'Kota 2').replace(/_/g,' '),
        trA.granularity || trB.granularity
      );

      // ringkas analisis untuk kota A saja (biar singkat)
      const s = trA.stats||{}; const fmt=n=>(n==null||Number.isNaN(n))?"-":rupiah(Math.round(n));
      const scope = bulan&&minggu ? `Minggu ke-${minggu}, ${monthsID[+bulan]} ${tahun}`
                  : bulan ? `${monthsID[+bulan]} ${tahun}` : `Tahun ${tahun}`;
      document.getElementById('analisisTren').textContent =
        `Perbandingan ${scope}. ${ (trA.entity||'Kota 1').replace(/_/g,' ') }: n=${s.n||0}, rata-rata Rp ${fmt(s.mean)}, rentang Rp ${fmt(s.min)}–Rp ${fmt(s.max)}, vol ${(s.vol_pct==null?'-':s.vol_pct.toFixed(2))}%`;
    } else {
      // fallback: single
      const s=trA.stats||{}; const fmt=n=>(n==null||Number.isNaN(n))?"-":rupiah(Math.round(n));
      document.getElementById('hargaMin').textContent=fmt(s.min);
      document.getElementById('hargaMax').textContent=fmt(s.max);
      document.getElementById('hargaRata').textContent=fmt(s.mean);
      document.getElementById('volatilitas').textContent=(s.vol_pct==null?"-":s.vol_pct.toFixed(2));
      const scope= bulan&&minggu ? `Minggu ke-${minggu}, ${monthsID[+bulan]} ${tahun}` : (bulan ? `${monthsID[+bulan]} ${tahun}` : `Tahun ${tahun}`);
      const title=`Tren Harga • ${(trA.entity||'').replace(/_/g,' ')} • ${scope}`;
      renderTrendChart(trA.series||[], trA.granularity, title);
      document.getElementById('analisisTren').textContent=`Analisis ${scope}: n=${s.n||0}, rata-rata Rp ${fmt(s.mean)}, rentang Rp ${fmt(s.min)}–Rp ${fmt(s.max)}, volatilitas ${(s.vol_pct==null?'-':s.vol_pct.toFixed(2))}%.`;
    }

    document.querySelector('.nav-link[data-section="tren"]')?.click();
  }catch(e){ console.error(e); alert('Gagal memuat tren: '+e.message); }
  finally{ loading.style.display='none'; }
}
window.loadData = loadData;

// ===== PREDIKSI (compare 2 kota)
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
function renderPredChart(labels, dsA, dsB, labelA, labelB) {
  const canvas=document.getElementById('predChart'); if(!canvas) return;
  const ctx = canvas.getContext('2d');
  if (_predChart) _predChart.destroy();
  _predChart = new Chart(ctx, {
    type:'line',
    data:{
      labels,
      datasets:[
        { label: labelA || 'Kota 1', data: dsA, borderWidth:2, tension:0.25, pointRadius:0 },
        { label: labelB || 'Kota 2', data: dsB, borderWidth:2, tension:0.25, pointRadius:0 }
      ]
    },
    options:{
      responsive:true,
      interaction:{ mode:'index', intersect:false },
      plugins:{
        title:{display:true,text:'Aktual vs Forecast (Gradient Boosting)'}
      },
      scales:{ x:{ ticks:{ autoSkip:true, maxTicksLimit:12 } }, y:{} }
    }
  });
}
async function loadPrediksi(){
  const aSel=document.getElementById('kabupaten_a'); 
  const bSel=document.getElementById('kabupaten_b');
  const aSlug=selectedSlugOrLabel(aSel);
  const bSlug=selectedSlugOrLabel(bSel);
  const start=document.getElementById('startDate').value; 
  const end=document.getElementById('endDate').value;
  const ph=document.getElementById('predPlaceholder');

  if (!aSlug || !bSlug) { alert('Pilih Kota/Kab 1 dan 2.'); return; }
  if (!start || !end)   { alert('Tanggal mulai & akhir wajib diisi.'); return; }

  try{
    ph.style.display='none';
    const [rA, rB] = await Promise.all([
      fetchPredictRange(aSlug, start, end),
      fetchPredictRange(bSlug, start, end)
    ]);
    const labels = fullDateRange(start, end);
    const mapA = new Map([...(rA.actual||[]), ...(rA.predicted||[])].map(p=>[p.date, p.value]));
    const mapB = new Map([...(rB.actual||[]), ...(rB.predicted||[])].map(p=>[p.date, p.value]));
    const dsA = labels.map(d => mapA.has(d) ? mapA.get(d) : null);
    const dsB = labels.map(d => mapB.has(d) ? mapB.get(d) : null);
    const labelA = (rA.entity||rA.city||'Kota 1').replace(/_/g,' ');
    const labelB = (rB.entity||rB.city||'Kota 2').replace(/_/g,' ');
    renderPredChart(labels, dsA, dsB, labelA, labelB);
    document.querySelector('.nav-link[data-section="prediksi"]')?.click();
  }catch(e){
    console.error(e);
    ph.style.display = '';
    ph.innerHTML = `❌ Gagal memuat prediksi: <small>${e.message}</small>`;
  }
}
window.loadPrediksi = loadPrediksi;

// ===== Default tanggal Prediksi (opsional 30 hari terakhir)
document.addEventListener('DOMContentLoaded', () => {
  const end = new Date();
  const start = new Date(); start.setDate(end.getDate() - 30);
  const toISO = d => d.toISOString().slice(0,10);
  const sEl=document.getElementById('startDate'); const eEl=document.getElementById('endDate');
  if (sEl && eEl && !sEl.value && !eEl.value) {
    sEl.value = toISO(start); eEl.value = toISO(end);
  }
});

// boot
initCities();
