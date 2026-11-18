import React, { useEffect, useState } from 'react'
import api from '../services/api'
import { STAGES, apiError, stageValue } from '../utils/analysisUi'
import { ErrorNotice, Loading, TableWrap } from './AnalysisViews'

export function FeatureTable({ data }) {
  const [all, setAll] = useState(false)
  if (!data) return <p className="text-sm text-gray-500">TF-IDF belum ditransformasikan untuk hasil ini.</p>
  return <section className="space-y-3"><p>{data.nonzero} fitur nonzero · vocabulary {data.vocabulary_size ?? '—'} · versi {data.model_version || 'hasil tersimpan'}</p><p className="text-sm text-gray-600">Bobot berasal dari transformasi vectorizer terlatih. Bobot tertinggi bukan probabilitas atau penjelasan tunggal keputusan SVM.</p><TableWrap><table className="w-full text-sm text-left"><thead><tr>{['Fitur', 'IDF model', 'TF-IDF'].map(x => <th className="p-2" key={x}>{x}</th>)}</tr></thead><tbody>{(all ? data.features : data.features?.slice(0, 20))?.map(f => <tr key={f.feature} className="border-t"><td className="p-2">{f.feature}</td><td className="p-2">{f.idf.toFixed(6)}</td><td className="p-2">{f.weight.toFixed(6)}</td></tr>)}</tbody></table></TableWrap>{data.features?.length > 20 && <button className="text-primary-700 underline" onClick={() => setAll(!all)}>{all ? 'Tampilkan 20 fitur' : 'Tampilkan seluruh fitur'}</button>}</section>
}

export default function ProcessDetails({ id, reviewId, prediction, onClose }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    let active = true
    setData(null); setError(null)
    if (prediction) setData({ review: { original: prediction.original, ...prediction.preprocessing_steps }, tfidf: { ...prediction.tfidf, model_version: prediction.model_version } })
    else api.getFeatures(id, reviewId).then(r => { if (active) setData(r) }).catch(e => { if (active) setError(apiError(e)) })
    const close = e => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', close)
    const previous = document.body.style.overflow; document.body.style.overflow = 'hidden'
    return () => { active = false; document.removeEventListener('keydown', close); document.body.style.overflow = previous }
  }, [id, reviewId, prediction, onClose])
  return <div className="fixed inset-0 z-50 bg-black/40 flex justify-end" onClick={onClose}><section role="dialog" aria-modal="true" aria-label="Detail proses ulasan" className="bg-white w-full max-w-2xl h-full overflow-y-auto p-5 space-y-4 shadow-xl" onClick={e => e.stopPropagation()}><div className="flex justify-between items-center"><h2 className="font-bold text-lg">Detail proses ulasan</h2><button autoFocus className="border rounded px-3 py-2" onClick={onClose}>Tutup</button></div>{error ? <ErrorNotice>{error}</ErrorNotice> : !data ? <Loading /> : <><p className="p-3 bg-slate-50 rounded whitespace-pre-wrap">{data.review.original}</p>{STAGES.map(([key, name], i) => <div key={key} className="border rounded p-3"><h3 className="font-semibold text-sm">{i < 7 ? `${i + 1}. ` : ''}{name}</h3><p className="text-sm break-words whitespace-pre-wrap mt-1">{data.review.final == null ? 'Belum diproses' : stageValue(data.review[key])}</p></div>)}<h3 className="font-bold">Pembobotan TF-IDF</h3><FeatureTable data={data.tfidf} /></>}</section></div>
}
