import React, { useEffect, useState } from 'react'
import { Link, useLocation, useParams, useSearchParams } from 'react-router-dom'
import SessionNav from '../components/SessionNav'
import ProcessDetails from '../components/ProcessDetails'
import apiService from '../services/api'
import { AspectView, ErrorNotice, EvaluationView, Loading, Pagination, PredictionView } from '../components/AnalysisViews'
import { apiError, CLASSES, LABELS, predicted, resultTab, versionLabel } from '../utils/analysisUi'

const noActualLabels = 'Evaluasi model belum tersedia karena dataset tidak memiliki label aktual. Silakan unggah dataset berlabel untuk menghitung metrik dan confusion matrix.'

export default function AnalysisResults() {
  const { id } = useParams()
  const { pathname } = useLocation()
  const tab = resultTab(pathname)
  const [analysis, setAnalysis] = useState(null)
  const [data, setData] = useState(null)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [params, setParams] = useSearchParams()
  const page = Math.max(1, Number.parseInt(params.get('page'), 10) || 1)
  const aspect = tab === 'aspects' ? params.get('aspect') || '' : ''
  const sentiment = params.get('sentiment') || ''
  const search = params.get('search') || ''
  const [draft, setDraft] = useState(search)
  const [refresh, setRefresh] = useState(0)
  useEffect(() => { setDraft(search); setSelected(null) }, [id, tab, search])
  const changeFilter = (key, nextValue) => {
    const next = new URLSearchParams(params)
    if (nextValue) next.set(key, nextValue)
    else next.delete(key)
    next.delete('page')
    if (key === 'aspect' && nextValue === '__none__') next.delete('sentiment')
    setParams(next)
  }
  useEffect(() => {
    let active = true
    setBusy(true); setError(null)
    const fetchData = tab === 'aspects' ? apiService.getAspectsData(id, 1, 20)
      : tab === 'evaluation' ? apiService.getEvaluation(id) : apiService.getPredictions(id, page, 20, sentiment || null, search)
    Promise.all([apiService.getAnalysis(id), fetchData, tab === 'aspects' ? apiService.getAspectReviews(id, page, aspect, sentiment, search)
      : tab === 'evaluation' ? apiService.getPredictions(id, page, 20, sentiment || null, search) : Promise.resolve(null)]).then(([a, result, reviewRows]) => {
      if (tab === 'evaluation') result = { ...reviewRows, evaluation: result.evaluation }
      if (tab === 'aspects') result = { ...result, review_rows: reviewRows.data, flow: reviewRows.flow, pagination: reviewRows.pagination }
      if (active) { setAnalysis(a.analysis); setData({ id, tab, value: result }) }
    }).catch(err => { if (active) { setError(apiError(err)); setData(null) } }).finally(() => { if (active) setBusy(false) })
    return () => { active = false }
  }, [id, tab, page, aspect, sentiment, search, refresh])
  const value = data?.id === id && data?.tab === tab ? data.value : null
  const current = String(analysis?.id) === id ? analysis : null
  const runMissing = async () => {
    setBusy(true); setError(null)
    try { if (tab === 'aspects') await apiService.detectAspects(id); else if (tab === 'predictions') await apiService.predict(id); else await apiService.evaluate(id); setRefresh(x => x + 1) }
    catch (err) { setError(apiError(err)); setBusy(false) }
  }
  return <div className="space-y-4"><SessionNav id={id} analysis={current} />
    <Link className="text-primary-700 text-sm" to={`/analysis/${id}`}>← Kembali ke analisis</Link>
    <div><h1 className="text-xl font-bold">Hasil analisis{current ? `: ${current.name}` : ''}</h1><p className="text-sm text-gray-500">Versi model: {current ? versionLabel(current) : '…'}</p></div>
    {error && <ErrorNotice>{error}</ErrorNotice>}
    <nav className="flex border-b gap-1" aria-label="Hasil analisis">{[['predictions', 'Labeling SVM'], ['aspects', 'Aspek'], ['evaluation', 'Evaluasi']].filter(([key]) => key !== 'evaluation' || current?.has_actual_labels).map(([key, label]) => <Link aria-current={tab === key ? 'page' : undefined} className={`px-4 py-3 text-sm ${tab === key ? 'border-b-2 border-primary-600 text-primary-700 font-semibold' : 'text-gray-600'}`} key={key} to={`/analysis/${id}/results/${key}`}>{label}</Link>)}</nav>
    {busy ? <Loading /> : current && value && <section className="bg-white border rounded-xl p-4 space-y-4">
      {tab === 'evaluation' && <>
        <DatasetEvaluationSummary analysis={current} />
        {!current.has_actual_labels ? <p>{noActualLabels}</p>
          : value.evaluation ? <EvaluationView evaluation={value.evaluation} matrixExportUrl={`/api/v1/analysis/${id}/export/confusion-matrix`} />
          : current.predictions_ready && current.flow.evaluable === 0 ? <p>Tidak ada ulasan dengan label aktual dan prediksi yang keduanya valid.</p>
          : current.flow.evaluable > 0 ? <button className="action-primary" onClick={runMissing}>Jalankan evaluasi</button>
          : <p>Jalankan prediksi SVM untuk menghitung metrik file ini.</p>}
      </>}
      {tab === 'predictions' && current.predictions_ready && !current.has_actual_labels && <p className="text-sm text-slate-600">{noActualLabels}</p>}
      {!predicted(current) && tab === 'evaluation' ? null
        : !predicted(current) ? <div className="space-y-3"><p>Belum ada label prediksi. Jalankan TF-IDF dan prediksi SVM; ulasan tanpa fitur dikenal dikecualikan.</p>{tab === 'predictions' && current.flow?.tfidf_ready ? <button onClick={runMissing} className="bg-primary-600 text-white rounded px-4 py-2">Jalankan Labeling SVM</button> : <Link className="text-primary-700 underline" to={`/analysis/${id}/tfidf`}>Buka TF-IDF</Link>}</div>

        : tab === 'aspects' && !value.ready ? <div><p>Analisis aspek belum dijalankan.</p>{current.model_version && <button className="mt-3 px-4 py-2 bg-primary-600 text-white rounded" onClick={runMissing}>Jalankan analisis aspek</button>}</div>
        : <>
          <div className="flex flex-wrap gap-3 items-center">
            {tab === 'aspects' && <label className="text-sm">Filter aspek <select className="border rounded p-2" value={aspect} onChange={e => changeFilter('aspect', e.target.value)}><option value="">Semua</option>{Object.keys(value.distribution || {}).map(key => <option key={key}>{key}</option>)}<option value="__none__">Tanpa aspek</option></select></label>}
            <label className="text-sm">{tab === 'aspects' ? 'Sentimen aspek' : 'Filter sentimen'} <select className="border rounded p-2" disabled={aspect === '__none__'} value={sentiment} onChange={e => changeFilter('sentiment', e.target.value)}><option value="">Semua</option>{CLASSES.map(key => <option value={key} key={key}>{LABELS[key]}</option>)}</select></label>
            <form className="flex flex-wrap gap-2" onSubmit={e => { e.preventDefault(); changeFilter('search', draft.trim()) }}><input aria-label="Cari ulasan" className="border rounded p-2 text-sm" value={draft} onChange={e => setDraft(e.target.value)} placeholder="Cari ulasan…" /><button className="border rounded px-3 text-sm">Cari</button></form>
          </div>
          {tab === 'aspects' && <p className="text-sm text-slate-600">{value.pagination?.total ?? 0} ulasan sesuai filter. Grafik dan ringkasan mencakup seluruh sesi.</p>}
          {tab === 'aspects' ? <AspectView data={value} onReview={setSelected} /> : <PredictionView data={value} onReview={setSelected} />}
          <Pagination pagination={value.pagination} onPage={nextPage => setParams({ ...Object.fromEntries(params), page: String(nextPage) })} disabled={busy} />
        </>}
      {current.model_identity_status === 'unknown' && <p className="text-sm text-amber-800">Versi prediksi lama tidak tercatat. Analisis ulang dari halaman analisis untuk memakai model aktif.</p>}
    </section>}
    {current?.predictions_ready && <div className="flex flex-wrap gap-3 text-sm"><a className="text-primary-700 underline" href={`/api/v1/analysis/${id}/export/${tab === 'aspects' ? 'aspects' : 'predictions'}`}>{tab === 'aspects' ? 'Unduh analisis aspek CSV' : 'Unduh prediksi CSV'}</a>{tab === 'predictions' && <Link className="text-primary-700 underline" to={`/analysis/${id}/results/aspects`}>Lanjut ke Analisis Aspek →</Link>}</div>}
    {selected && <ProcessDetails id={id} reviewId={selected} onClose={()=>setSelected(null)}/>}
  </div>
}


function DatasetEvaluationSummary({ analysis }) {
  const flow = analysis.flow
  const counts = [
    ['Baris diunggah', flow.uploaded], ['Preprocessing selesai', flow.preprocessed],
    ['Prediksi SVM', flow.predicted], ['Pasangan evaluasi', flow.evaluable],
    ['Teks awal kosong', flow.empty_source], ['Duplikat', flow.duplicates],
    ['Kosong setelah preprocessing', flow.empty_processed], ['Tanpa fitur TF-IDF', flow.zero_vectors],
    ['Label kosong di file', flow.missing_actual_labels],
    ['Prediksi tanpa label', flow.predicted_without_label],
    ['Cocok development', flow.canonical_matches?.development],
    ['Cocok holdout', flow.canonical_matches?.holdout],
  ]
  return <section className="space-y-3">
    <h2 className="font-semibold">Dataset evaluasi</h2>
    <p className="text-sm">{analysis.filename} · {analysis.evaluation_scope === 'legacy_unknown' ? 'Asal data belum tercatat' : analysis.mode === 'research_demo' ? 'Demonstrasi Data Penelitian' : 'Pengujian Data Baru'}</p>
    <dl className="grid grid-cols-2 gap-2 md:grid-cols-4">
      {counts.map(([label, count]) => <div key={label} className="rounded bg-slate-50 p-2">
        <dt className="text-xs text-slate-600">{label}</dt><dd className="font-semibold">{count ?? '—'}</dd>
      </div>)}
    </dl>
    <table className="text-sm text-left">
      <thead><tr><th className="p-2">Kelas</th><th className="p-2">Label aktual di file</th><th className="p-2">Prediksi SVM</th></tr></thead>
      <tbody>{CLASSES.map(key => <tr key={key}><th scope="row" className="p-2">{LABELS[key]}</th>
        <td className="p-2">{flow.actual_distribution?.[key] ?? '—'}</td>
        <td className="p-2">{flow.distribution?.[key] ?? '—'}</td></tr>)}</tbody>
    </table>
  </section>
}
