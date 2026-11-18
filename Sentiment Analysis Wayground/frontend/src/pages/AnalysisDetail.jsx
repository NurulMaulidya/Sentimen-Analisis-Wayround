import React, { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import api from '../services/api'
import InferenceModelSelect from '../components/InferenceModelSelect'
import SessionNav from '../components/SessionNav'
import SentimentCharts from '../components/SentimentCharts'
import { ErrorNotice, Loading, TableWrap, EvaluationView } from '../components/AnalysisViews'
import { apiError, versionLabel, statusLabel } from '../utils/analysisUi'

const analysisStages = ['Deduplikasi', 'Preprocessing', 'TF-IDF', 'Labeling SVM', 'Aspek', 'Evaluasi']

export default function AnalysisDetail() {
  const { id } = useParams(), navigate = useNavigate()
  const [model, setModel] = useState('v4')
  const [analysis,setAnalysis] = useState(null), [error,setError] = useState(null), [busy,setBusy] = useState(''), [runs,setRuns] = useState([]), [snapshot,setSnapshot] = useState(null)
  const reload = async () => {const [a,r] = await Promise.all([api.getAnalysis(id),api.getRuns(id)]);setAnalysis(a.analysis);setRuns(r.runs)}
  useEffect(()=>{let active=true;setAnalysis(null);setModel('v4');setSnapshot(null);setError(null);Promise.all([api.getAnalysis(id),api.getRuns(id)]).then(([a,r])=>{if(active){setAnalysis(a.analysis);setRuns(r.runs)}}).catch(e=>{if(active)setError(apiError(e))});return()=>{active=false}},[id])
  const [failedStage, setFailedStage] = useState('')
  const run = async () => {
    setError(null)
    setFailedStage('')
    let stage = ''
    const start = name => { stage = name; setBusy(name) }
    try {
      start('Deduplikasi');await api.deduplicate(id)
      start('Preprocessing');await api.preprocess(id)
      start('TF-IDF');await api.transformTfidf(id, model)
      const a = (await api.getAnalysis(id)).analysis
      if (!a.flow.feature_ready) {setFailedStage(stage);await reload();setError('Tidak ada ulasan dengan fitur dikenal. Periksa detail preprocessing dan TF-IDF.');return}
      start('Labeling SVM');await api.predict(id, model)
      start('Aspek');await api.detectAspects(id)
      const predicted = (await api.getAnalysis(id)).analysis
      if (predicted.flow.evaluable > 0) {start('Evaluasi');await api.evaluate(id)}
      await reload()
    } catch(e) {setFailedStage(stage);setError(apiError(e));await reload().catch(()=>{})} finally {setBusy('')}
  }
  const remove = async () => {if(!window.confirm('Hapus sesi beserta hasil dan snapshot sebelumnya?'))return;try{await api.deleteAnalysis(id);navigate('/history')}catch(e){setError(apiError(e))}}
  const viewRun = async run => {setError(null);try{const r=await api.getRun(id,run);setSnapshot(r.snapshot)}catch(e){setError(apiError(e))}}
  const currentStage = analysisStages.indexOf(busy || failedStage)
  const current = String(analysis?.id) === id ? analysis : null
  if (!current) return error ? <ErrorNotice>{error}</ErrorNotice> : <Loading/>
  const f=current.flow
  return <div className="space-y-5"><SessionNav id={id} analysis={current}/><header className="flex justify-between gap-3"><div><h1 className="text-2xl font-bold">{current.name}</h1><p className="text-sm text-gray-500">{current.filename} · {statusLabel(current.status)}</p></div><button className="text-red-700 border rounded px-3 disabled:opacity-40" disabled={!!busy} onClick={remove}>Hapus</button></header>{error && <ErrorNotice>{error}</ErrorNotice>}<section className="bg-white border border-slate-200 rounded-xl p-5 flex flex-wrap gap-4 justify-between items-center"><div><p className="font-semibold">Satu langkah untuk melihat gambaran ulasan Anda.</p><p className="text-sm text-gray-600">Hasil sebelumnya tetap tersimpan saat Anda menjalankan analisis ulang.</p><InferenceModelSelect value={model} onChange={setModel} disabled={!!busy} /></div><button className="px-4 py-2 rounded bg-primary-600 text-white disabled:opacity-40" disabled={!!busy} onClick={run}>{busy ? `Memproses ${busy}…` : current.predictions_ready ? 'Analisis ulang' : 'Jalankan analisis lengkap'}</button></section>
    {currentStage >= 0 && <ol className="grid gap-2 sm:grid-cols-3" aria-label="Tahapan analisis" aria-live="polite">
      {analysisStages.filter(name => name !== 'Evaluasi' || current.has_actual_labels).map((name, index) => {
        const status = index < currentStage ? 'Selesai' : index > currentStage ? 'Menunggu' : failedStage ? 'Terhenti' : 'Berjalan'
        return <li key={name} aria-current={index === currentStage ? 'step' : undefined}
          className="rounded-lg border border-primary-100 bg-white p-3 text-sm">
          <span className="font-medium">{name}</span>
          <span className="block text-xs text-slate-600">{status}</span>
        </li>
      })}
    </ol>}
    <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">{[['Ulasan dianalisis',f.predicted],['Positif',f.distribution.positif],['Negatif',f.distribution.negatif],['Netral',f.distribution.netral]].map(([k,v])=><div key={k} className="stat-tile"><p className="text-xs text-slate-500">{k}</p><p className="text-3xl font-semibold mt-3">{v}</p></div>)}</section><details className="surface-panel"><summary className="cursor-pointer font-semibold text-sm">Rincian jumlah data dan proses</summary><div className="mt-4"><div className="grid grid-cols-2 lg:grid-cols-4 gap-3">{[['Data upload',f.uploaded],['Ulasan valid',f.valid_source],['Teks awal kosong',f.empty_source],['Duplikat',f.duplicates],['Setelah deduplikasi',f.after_dedup],['Preprocessing selesai',f.preprocessed],['Teks akhir kosong',f.empty_processed],['Vektor nol',f.zero_vectors],['Diprediksi',f.predicted],['Memuat aspek',f.with_aspects],['Tanpa aspek',f.without_aspects],[current.mode === 'research_demo' ? 'Pasangan label pakar dan SVM' : 'Pasangan evaluasi',f.evaluable]].map(([k,v])=><div key={k} className="bg-white border rounded-lg p-3"><p className="text-2xl font-bold">{v}</p><p className="text-xs text-gray-500">{k}</p></div>)}</div></div></details>{!f.source_complete && <p className="text-amber-800 text-sm">Riwayat lama tidak menyimpan seluruh baris sumber. Jumlah teks awal kosong tidak dapat direkonstruksi sepenuhnya.</p>}{current.predictions_ready && <section className="bg-white border rounded-xl p-4"><SentimentCharts distribution={{'Hasil sesi':f.distribution}} caption="Distribusi Labeling SVM"/></section>}<div className="flex flex-wrap gap-3"><Link className="action-primary" to={`/analysis/${id}/results/predictions`}>Lihat hasil sentimen →</Link><Link className="action-quiet" to={`/analysis/${id}/results/aspects`}>Lihat analisis aspek →</Link>{current.has_actual_labels && <Link className="action-quiet" to={`/analysis/${id}/results/evaluation`}>Evaluasi file ini →</Link>}</div><details className="surface-panel"><summary className="cursor-pointer font-semibold text-sm">Telusuri preprocessing dan model</summary><div className="mt-4 space-y-4"><div className="flex flex-wrap gap-4 text-sm">{[['data/original','Data awal'],['pipeline/preprocessing/cleaning','Tujuh tahap preprocessing'],['tfidf','Bobot TF-IDF']].map(([path,name])=><Link key={path} className="text-primary-700 underline" to={`/analysis/${id}/${path}`}>{name}</Link>)}</div><section className="bg-white border rounded-xl p-4"><p>Versi model hasil tersimpan: <strong>{versionLabel(current)}</strong></p>{current.preprocessing_signature && <p className="mt-1 break-all text-xs text-gray-500">Fingerprint preprocessing {current.preprocessing_signature}</p>}{current.predicted_at && <p className="text-sm text-gray-500 mt-1">{new Date(current.predicted_at).toLocaleString('id-ID',{timeZone:'Asia/Jakarta'})} WIB</p>}<p className="text-sm text-gray-500 mt-2">Ringkasan sesi ini mencakup seluruh data. Data upload tidak mengubah model penelitian.</p></section></div></details><section className="bg-white border rounded-xl p-4 space-y-3"><h2 className="font-semibold">Unduh hasil sesi</h2><div className="flex flex-wrap gap-3 text-sm">{['validation','preprocessing',...(current.predictions_ready ? ['predictions','aspects'] : []),...(current.evaluation_ready ? ['evaluation'] : [])].map(k=><a key={k} className="text-primary-700 underline" href={`/api/v1/analysis/${id}/export/${k}`}>{({validation:'Validasi',preprocessing:'Preprocessing',predictions:'Prediksi',aspects:'Aspek',evaluation:'Evaluasi'})[k]}</a>)}</div></section>{runs.length > 0 && <section className="bg-white border rounded-xl p-4 space-y-3"><h2 className="font-semibold">Snapshot hasil sebelum analisis ulang</h2><p className="text-xs text-gray-500">Snapshot hanya dibaca dari penyimpanan, tidak dihitung ulang dengan model aktif.</p>{runs.map(r=><div className="flex flex-wrap gap-3 text-sm" key={r.id}><span>#{r.id} · {new Date(r.created_at).toLocaleString('id-ID')} · model {r.analysis.model_version}</span><button onClick={()=>viewRun(r.id)} className="text-primary-700 underline">Lihat snapshot</button></div>)}{snapshot && <div className="space-y-3 border-t pt-3"><p>Model snapshot: {snapshot.analysis.model_version}</p><TableWrap><table className="w-full text-sm text-left"><thead><tr><th className="p-3">Ulasan tersimpan</th><th>Prediksi</th><th>Aspek</th></tr></thead><tbody>{snapshot.reviews.map(r=><tr className="border-t" key={r.id}><td className="p-3">{r.original_content}</td><td>{r.predicted_label || '—'}</td><td>{r.detected_aspects?.join(', ') || '—'}</td></tr>)}</tbody></table></TableWrap>{snapshot.evaluation && <EvaluationView evaluation={snapshot.evaluation}/>}<button onClick={()=>setSnapshot(null)} className="text-primary-700 underline">Tutup snapshot</button></div>}</section>}</div>
}
