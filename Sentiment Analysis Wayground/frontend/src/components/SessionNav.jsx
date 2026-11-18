import React, { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import api from '../services/api'

export default function SessionNav({ id, analysis: supplied }) {
  const { pathname } = useLocation()
  const [loaded, setLoaded] = useState(null)
  useEffect(() => {
    let active = true
    if (!supplied) api.getAnalysis(id).then(r => { if (active) setLoaded(r.analysis) }).catch(() => {})
    return () => { active = false }
  }, [id, supplied])
  const a = supplied || (String(loaded?.id) === String(id) ? loaded : null)
  const steps = [
    ['data/original', 'Validasi & Data Awal', true, a?.dedup_ready],
    ['pipeline/preprocessing/cleaning', 'Preprocessing', true, a?.preprocessing_ready],
    ['tfidf', 'TF-IDF', a?.preprocessing_ready, a?.flow?.tfidf_ready],
    ['results/predictions', 'Labeling SVM', a?.flow?.tfidf_ready || a?.predictions_ready, a?.predictions_ready],
    ['results/aspects', 'Analisis Aspek', a?.predictions_ready, a?.aspects_ready],
    ['', 'Ringkasan', true, a?.status === 'completed'],
    ...(a?.has_actual_labels ? [['results/evaluation', 'Evaluasi Upload', a?.predictions_ready, a?.evaluation_ready]] : [])
  ]
  return <div className="space-y-2 workflow-strip"><p className="text-xs text-gray-500">Sesi #{id} · {a?.name || 'Analisis data uji'}{a?.evaluation_scope === 'new_data' ? ' · Pengujian Data Baru' : ''}</p>{a?.mode === 'research_demo' && <p className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900">Demonstrasi Data Penelitian · Analisis deskriptif pada data penelitian, termasuk sampel development dan holdout jika ada.</p>}<nav aria-label="Alur analisis" className="flex gap-2">{steps.map(([path, name, enabled, done], i) => {
    const url = `/analysis/${id}${path ? '/' + path : ''}`
    const current = path.startsWith('pipeline') ? pathname.includes('/pipeline') : pathname === url
    return <Link key={name} to={url} aria-current={current ? 'step' : undefined} aria-disabled={!enabled} onClick={e => { if (!enabled) e.preventDefault() }} className={`text-sm rounded-lg border px-3 py-2 ${current ? 'bg-primary-600 text-white' : enabled ? 'bg-white text-slate-700' : 'bg-gray-100 text-gray-400'}`}>{done ? '✓' : i + 1} {name}</Link>
  })}</nav></div>
}
