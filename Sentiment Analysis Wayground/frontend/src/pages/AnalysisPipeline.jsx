import React, { useEffect, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import SessionNav from '../components/SessionNav'
import ProcessDetails from '../components/ProcessDetails'
import apiService from '../services/api'
import { ErrorNotice, Loading, Pagination, StageTable } from '../components/AnalysisViews'
import { apiError, STAGES } from '../utils/analysisUi'

export default function AnalysisPipeline() {
  const { id, stage: requested } = useParams()
  const { pathname } = useLocation()
  const compare = pathname.endsWith('/compare')
  const stage = requested || 'cleaning'
  const valid = STAGES.some(([key]) => key === stage)
  const [analysis, setAnalysis] = useState(null)
  const [data, setData] = useState(null)
  const [selected, setSelected] = useState(null)
  const [page, setPage] = useState(1)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState(null)
  const [refresh, setRefresh] = useState(0)
  useEffect(() => { setPage(1); setData(null) }, [id, stage, compare])
  useEffect(() => {
    let active = true
    setBusy(true); setError(null)
    if (!valid && !compare) { setError('Tahap preprocessing tidak dikenal.'); setBusy(false); return }
    Promise.all([apiService.getAnalysis(id), apiService.getPreprocessingStage(id, compare ? 'all' : stage, page, 20)]).then(([a, result]) => {
      if (active) { setAnalysis(a.analysis); setData({ id, stage, compare, value: result }) }
    }).catch(err => { if (active) { setError(apiError(err)); setData(null) } }).finally(() => { if (active) setBusy(false) })
    return () => { active = false }
  }, [id, stage, compare, page, refresh, valid])
  const current = String(analysis?.id) === id ? analysis : null
  const value = data?.id === id && data?.stage === stage && data?.compare === compare ? data.value : null
  const run = async () => {
    setBusy(true); setError(null)
    try { if (!current.dedup_ready) await apiService.deduplicate(id); await apiService.preprocess(id); setRefresh(x => x + 1) }
    catch (err) { setError(apiError(err)); setBusy(false) }
  }
  return <div className="space-y-4"><SessionNav id={id} analysis={current} />
    <Link to={`/analysis/${id}`} className="text-sm text-primary-700">← Kembali ke analisis</Link>
    <h1 className="text-xl font-bold">{compare ? 'Perbandingan preprocessing' : 'Tahap preprocessing'}</h1>
    <p className="text-sm text-gray-500">{current?.name} · Tujuh tahap dan hasil akhir</p>
    <nav className="flex flex-wrap gap-2" aria-label="Tahap preprocessing">{STAGES.map(([key, label]) => <Link className={`border rounded px-3 py-2 text-sm ${!compare && stage === key ? 'bg-primary-600 text-white' : 'bg-white'}`} aria-current={!compare && stage === key ? 'page' : undefined} to={`/analysis/${id}/pipeline/preprocessing/${key}`} key={key}>{label}</Link>)}<Link className={`border rounded px-3 py-2 text-sm ${compare ? 'bg-primary-600 text-white' : 'bg-white'}`} to={`/analysis/${id}/pipeline/compare`}>Bandingkan semua</Link></nav>
    {error && <ErrorNotice>{error}</ErrorNotice>}
    {busy ? <Loading /> : current && !current.preprocessing_ready ? <div className="bg-white p-4 border rounded"><p>{current.flow?.after_dedup === 0 ? 'Tidak ada ulasan valid untuk diproses. Periksa data awal atau unggah file lain.' : 'Preprocessing belum dijalankan.'}</p><button disabled={current.flow?.after_dedup === 0} className="mt-3 px-4 py-2 bg-primary-600 text-white rounded disabled:opacity-40" onClick={run}>Jalankan preprocessing</button></div> : value && valid && <section className="bg-white p-4 border rounded-xl"><StageTable rows={value.data || []} stage={stage} compare={compare} stages={STAGES} onReview={setSelected} /><Pagination pagination={value.pagination} onPage={setPage} disabled={busy} /></section>}
    {current?.preprocessing_ready && <Link className="inline-block px-4 py-2 bg-primary-600 text-white rounded" to={`/analysis/${id}/tfidf`}>Lanjut ke TF-IDF →</Link>}
    {selected && <ProcessDetails id={id} reviewId={selected} onClose={()=>setSelected(null)}/>}
  </div>
}
