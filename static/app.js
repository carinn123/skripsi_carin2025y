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
  sel.length=1; const frag=document.createDocumentFragment();
  for(const r of records){const opt=document.createElement('option');opt.value=r.label;opt.textContent=r.label;opt.dataset.label=r.label;opt.dataset.slug=r.slug||"";opt.dataset.entity=r.entity||"";frag.appendChild(opt)}
  sel.appendChild(frag);
}
function wireSearch(){
  const search=document.getElementById('search-kab'); const sel=document.getElementById('kabupaten'); if(!search||!sel) return;
  search.addEventListener('input',()=>{const q=search.value.toLowerCase(); for(const opt of sel.options){ if(!opt.value) continue; const label=(opt.dataset.label||opt.textContent).toLowerCase(); opt.hidden=!label.includes(q); }});
}
function getSelectedCityLabel(){const sel=document.getElementById('kabupaten'); const opt=sel.options[sel.selectedIndex]; return opt?.dataset?.label||sel.value;}

// ===== Cities & Islands =====
async function initCities(){
  try{
    let res=await fetch('/api/cities_full'); if(!res.ok) throw new Error("fallback");
    const data=await res.json();
    fillSelect('kabupaten', data.map(d=>({label:d.label, slug:d.slug, entity:d.entity})));
  }catch(_){
    try{ const r2=await fetch('/api/cities'); const arr=await r2.json(); fillSelect('kabupaten', arr.map(l=>({label:l}))); }
    catch(e){ console.error("Gagal memuat daftar kota/kab:", e); }
  }
  wireSearch();
}
(async function initIslands(){
  try{
    const res=await fetch('/api/islands'); const islands=await res.json();
    const sel=document.getElementById('pulau');
    islands.forEach(n=>{const opt=document.createElement('option'); opt.value=n; opt.textContent=n; sel.appendChild(opt);});
  }catch(e){ console.error(e); }
})();

// Enable minggu (global + beranda)
const bulanSel=document.getElementById('bulan');
const mingguSel=document.getElementById('minggu');
bulanSel?.addEventListener('change',()=>{ if(!bulanSel.value){mingguSel.value="";mingguSel.disabled=true;} else {mingguSel.disabled=false;} });
const b_bulanSel=document.getElementById('b_bulan');
const b_mingguSel=document.getElementById('b_minggu');
b_bulanSel?.addEventListener('change',()=>{ if(!b_bulanSel.value){b_mingguSel.value="";b_mingguSel.disabled=true;} else {b_mingguSel.disabled=false;} });

// ===== MAP (Beranda) =====
let __PROV_GJ=null;
async function getProvGeoJSON(){ if(__PROV_GJ) return __PROV_GJ; const r=await fetch('/static/indonesia_provinces.geojson?v=1',{cache:'no-store'}); if(!r.ok) throw new Error('GeoJSON provinsi tidak ditemukan'); __PROV_GJ=await r.json(); return __PROV_GJ; }
let map,geoLayer;
function ensureMap(){ if(map) return map; map=L.map('map',{scrollWheelZoom:true}).setView([-2.5,118],5); L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:10, attribution:'&copy; OpenStreetMap'}).addTo(map); return map; }

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
      style: f=>{ const raw=f.properties.Propinsi||f.properties.PROVINSI||f.properties.provinsi||f.properties.name||f.properties.NAMOBJ||""; const rec=vmap[normProv(raw)];
        const fill=rec? (rec.cat==='low'?'#2ecc71':rec.cat==='mid'?'#f1c40f':'#e74c3c') : '#bdc3c7';
        return {color:'#fff',weight:1,fillColor:fill,fillOpacity:.85};},
      onEachFeature:(feature,layer)=>{ const raw=feature.properties.Propinsi||feature.properties.PROVINSI||feature.properties.provinsi||feature.properties.name||feature.properties.NAMOBJ||"—";
        const rec=vmap[normProv(raw)]; const val=rec? Math.round(rec.val):null; const cat=rec? rec.cat:'no-data';
        layer.bindPopup(`<b>${raw}</b><br/>${val?('Rp '+rupiah(val)):'—'}<br/>Kategori: ${cat}`); }
    }).addTo(m);
    try{ m.fitBounds(geoLayer.getBounds(), {padding:[20,20]}); }catch(e){}
    document.getElementById('statPulau').textContent=island||'Semua Pulau';
    let scope=`Tahun ${tahun}`; if(bulan) scope=`Bulan ${monthsID[+bulan]} ${tahun}`; if(bulan&&minggu) scope=`Minggu ke-${minggu}, ${monthsID[+bulan]} ${tahun}`;
    document.getElementById('statScope').textContent=scope;
    document.getElementById('statLast').textContent=js.last_actual||'-';
  }catch(e){ console.error(e); alert('Gagal menampilkan peta: '+e.message); }
  finally{ loading.style.display='none'; }
}
window.showMapBeranda = showMapBeranda;

// ===== TINJAUAN TREN =====
let trendChart=null;
function renderTrendChart(series,granularity,title){
  const placeholder=document.getElementById('trendPlaceholder');
  const ctx=document.getElementById('trendChart').getContext('2d');
  const labels=series.map(s=>s.label);
  const values=series.map(s=>s.value);
  if(!series.length){ placeholder.style.display='block'; if(trendChart){trendChart.destroy();trendChart=null;} return; }
  placeholder.style.display='none';
  const datasetLabel= granularity==='yearly'?'Rata-rata per Bulan': (granularity==='monthly'?'Rata-rata per Minggu':'Harian (Minggu terpilih)');
  const chartData={labels, datasets:[{label:datasetLabel, data: values, tension:.25, fill:true, pointRadius:3}]};
  const options={responsive:true, maintainAspectRatio:false, plugins:{title:{display:true,text:title}, tooltip:{callbacks:{label:(c)=>` ${datasetLabel}: Rp ${rupiah(c.parsed.y)}`}}}, scales:{y:{ticks:{callback:v=>'Rp '+rupiah(v)}}, x:{ticks:{autoSkip:true,maxRotation:0}}}};
  if(trendChart){ trendChart.data=chartData; trendChart.options=options; trendChart.update(); } else { trendChart=new Chart(ctx,{type:'line',data:chartData,options}); }
}
async function loadData(){
  const cityLabel=getSelectedCityLabel();
  const tahun=document.getElementById('tahun').value;
  const bulan=document.getElementById('bulan').value;
  const minggu=document.getElementById('minggu').value;
  const loading=document.getElementById('loadingSpinner');
  if(!cityLabel||!tahun){ alert('Silakan pilih kabupaten/kota dan tahun dahulu!'); return; }
  loading.style.display='inline-block';
  try{
    const url=`/api/trend?city=${encodeURIComponent(cityLabel)}&year=${encodeURIComponent(tahun)}&month=${encodeURIComponent(bulan)}&week=${encodeURIComponent(minggu)}`;
    const res=await fetch(url); const trend=await res.json(); if(!res.ok) throw new Error(trend.error||'Server error');
    const s=trend.stats||{}; const fmt=n=>(n==null||Number.isNaN(n))?"-":rupiah(Math.round(n));
    document.getElementById('hargaMin').textContent=fmt(s.min);
    document.getElementById('hargaMax').textContent=fmt(s.max);
    document.getElementById('hargaRata').textContent=fmt(s.mean);
    document.getElementById('volatilitas').textContent=(s.vol_pct==null?"-":s.vol_pct.toFixed(2));
    const scope= bulan&&minggu ? `Minggu ke-${minggu}, ${monthsID[+bulan]} ${tahun}` : (bulan ? `${monthsID[+bulan]} ${tahun}` : `Tahun ${tahun}`);
    const title=`Tren Harga • ${cityLabel.toUpperCase()} • ${scope}`;
    renderTrendChart(trend.series||[], trend.granularity, title);
    document.getElementById('analisisTren').textContent=`Analisis ${scope}: n=${s.n||0}, rata-rata Rp ${fmt(s.mean)}, rentang Rp ${fmt(s.min)}–Rp ${fmt(s.max)}, volatilitas ${(s.vol_pct==null?'-':s.vol_pct.toFixed(2))}%.`;
    document.querySelector('.nav-link[data-section="tren"]')?.click();
  }catch(e){ console.error(e); alert('Gagal memuat tren: '+e.message); }
  finally{ loading.style.display='none'; }
}
window.loadData = loadData;

// ===== PREDIKSI =====
let predChart=null;
function ensurePredChart(){
  const ctx=document.getElementById('predChart').getContext('2d');
  if(!predChart){
    predChart=new Chart(ctx,{
      type:'line',
      data:{labels:[],datasets:[]},
      options:{
        responsive:true, maintainAspectRatio:false,
        interaction:{mode:'index',intersect:false},
        plugins:{
          title:{display:true,text:'Aktual vs Forecast (Gradient Boosting)'},
          tooltip:{mode:'index',intersect:false,callbacks:{
            label:(c)=>` ${c.dataset.label}: Rp ${new Intl.NumberFormat('id-ID').format(Math.round(c.parsed.y))}`
          }},
          legend:{position:'top'}
        },
        scales:{
          y:{ticks:{callback:v=>'Rp '+new Intl.NumberFormat('id-ID').format(v)}},
          x:{ticks:{autoSkip:true,maxRotation:0}}
        }
      }
    });
  }
  return predChart;
}
function fullDateRange(start,end){ const s=new Date(start), e=new Date(end); const out=[]; if(isNaN(s)||isNaN(e)||s>e) return out; for(let d=new Date(s); d<=e; d.setDate(d.getDate()+1)){ out.push(new Date(d).toISOString().slice(0,10)); } return out; }
async function loadPrediksi(){
  const cityLabel=getSelectedCityLabel();
  const start=document.getElementById('startDate').value;
  const end=document.getElementById('endDate').value;
  const placeholder=document.getElementById('predPlaceholder');
  if(!cityLabel){ alert('Pilih kabupaten/kota dulu.'); return; }
  if(!start||!end){ alert('Tanggal mulai & akhir wajib diisi.'); return; }
  try{
    const url=`/api/predict_range?city=${encodeURIComponent(cityLabel)}&start=${start}&end=${end}&naive_fallback=1`;
    const res=await fetch(url); const js=await res.json(); if(!res.ok) throw new Error(js.error||'Server error');
    const labels=fullDateRange(start,end);
    const aMap=Object.fromEntries((js.actual||[]).map(p=>[p.date,p.value]));
    const pMap=Object.fromEntries((js.predicted||[]).map(p=>[p.date,p.value]));
    const actualY=labels.map(d=>(d in aMap)?aMap[d]:null);
    const predictY=labels.map(d=>(d in pMap)?pMap[d]:null);
    const chart=ensurePredChart(); placeholder.style.display='none';
    chart.data.labels=labels;
    chart.data.datasets=[
      {label:'Aktual', data:actualY, borderWidth:2, spanGaps:true, pointRadius:2},
      {label:'Forecast', data:predictY, borderWidth:2, borderDash:[6,4], spanGaps:true, pointRadius:2}
    ];
    chart.update();
    document.querySelector('.nav-link[data-section="prediksi"]')?.click();
  }catch(e){ console.error(e); alert('Gagal memuat prediksi: '+e.message); }
}
window.loadPrediksi = loadPrediksi;

// boot
initCities();
