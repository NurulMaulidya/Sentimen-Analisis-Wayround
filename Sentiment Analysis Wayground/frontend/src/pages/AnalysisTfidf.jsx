import React, { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import api from '../services/api'
import InferenceModelSelect from '../components/InferenceModelSelect'
import SessionNav from '../components/SessionNav'
import ProcessDetails from '../components/ProcessDetails'
import { ErrorNotice, Loading, Pagination, TableWrap } from '../components/AnalysisViews'
import { apiError } from '../utils/analysisUi'

export default function AnalysisTfidf() {
  const { id } = useParams()
  const [model, setModel] = useState(null)
  const [page, setPage] = useState(1), [data, setData] = useState(null), [busy, setBusy] = useState(true)
  const [error, setError] = useState(null), [selected, setSelected] = useState(null), [refresh, setRefresh] = useState(0)
  useEffect(() => { setPage(1); setSelected(null); setModel(null) }, [id])
  useEffect(() => {
    let active = true
    setBusy(true); setError(null); setData(null)
    Promise.all([api.getAnalysis(id), api.getTfidf(id, page)]).then(([a, t]) => { if (active) setData({ id, a: a.analysis, t }) }).catch(e => { if (active) setError(apiError(e)) }).finally(() => { if (active) setBusy(false) })
    return () => { active = false }
  }, [id, page, refresh])
  const current = data?.id === id ? data : null
  const inspectedModel = current?.t.flow.tfidf_model_version?.startsWith('kandidat-karakter-') ? 'karakter' : 'v4'
  const modelToUse = model ?? inspectedModel
  const config = current?.t.data[0]?.configuration
  const run = async () => {
    setBusy(true); setError(null)
    try { await api.transformTfidf(id, modelToUse); setRefresh(x => x + 1) } catch (e) { setError(apiError(e)); setBusy(false) }
  }
  return <div className="space-y-5"><SessionNav id={id} analysis={current?.a} /><h1 className="text-xl font-bold">Pembobotan TF-IDF</h1><p className="text-sm text-gray-600">Transformasi memakai vocabulary dan IDF model terlatih. Data upload tidak melatih ulang vectorizer.</p>{error && <ErrorNotice>{error}</ErrorNotice>}{busy ? <Loading /> : current && <><InferenceModelSelect value={modelToUse} onChange={setModel} disabled={current.a.predictions_ready} /><div className="flex flex-wrap items-center gap-3">{current.t.flow.tfidf_ready && <a className="text-primary-700 underline text-sm" href={`/api/v1/analysis/${id}/export/tfidf`}>Unduh bobot TF-IDF</a>}<button disabled={!current.a.preprocessing_ready} onClick={run} className="bg-primary-600 text-white rounded px-4 py-2 disabled:opacity-40">Transformasi TF-IDF</button>{current.t.flow.tfidf_ready && <Link className="text-primary-700 underline" to={`/analysis/${id}/results/predictions`}>Lanjut ke Labeling SVM →</Link>}</div>{!current.a.preprocessing_ready && <p>Jalankan preprocessing terlebih dahulu.</p>}<div className="grid grid-cols-2 md:grid-cols-4 gap-3">{[['Fitur siap', current.t.flow.feature_ready], ['Vektor nol', current.t.flow.zero_vectors], ['Teks akhir kosong', current.t.flow.empty_processed], ['Versi vectorizer', current.t.flow.tfidf_model_version || 'Belum ditransformasi']].map(([k, v]) => <div key={k} className="bg-white border rounded-lg p-3"><p className="text-xs text-gray-500">{k}</p><p className="font-semibold break-all">{v}</p></div>)}</div><p className="text-sm text-amber-800">Teks akhir kosong dan vektor tanpa fitur dikenal dikecualikan dari prediksi; keduanya tidak otomatis diberi label netral.</p>{current.t.data.length > 0 && <><p className="text-sm">Vocabulary aktual: {current.t.data[0].vocabulary_size} {config?.char ? ' · TF-IDF kata (1–3) dan karakter (3–5)' : config ? ` · batas fitur ${config.max_features} · n-gram ${config.ngram_range.join('–')}` : ''}</p><TableWrap><table className="text-sm text-left w-full"><thead><tr>{['ID', 'Ulasan', 'Teks final', 'Fitur nonzero', 'Status', 'Detail'].map(x => <th className="p-3" key={x}>{x}</th>)}</tr></thead><tbody>{current.t.data.map(r => <tr className="border-t align-top" key={r.id}><td className="p-3">{r.id}</td><td className="p-3 min-w-48">{r.original_content}</td><td className="p-3">{r.final_processed_text || '—'}</td><td className="p-3">{r.nonzero}</td><td className="p-3">{{ready:'Siap',zero_vector:'Vektor nol',empty_processed:'Teks akhir kosong'}[r.status]}</td><td className="p-3"><button className="text-primary-700 underline whitespace-nowrap" onClick={() => setSelected(r.id)}>Lihat Bobot</button></td></tr>)}</tbody></table></TableWrap><Pagination pagination={current.t.pagination} onPage={setPage} /></>}</>}{selected && <ProcessDetails id={id} reviewId={selected} onClose={() => setSelected(null)} />}</div>
}
