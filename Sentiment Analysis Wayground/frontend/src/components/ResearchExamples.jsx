import React, { useEffect, useState } from 'react'
import api from '../services/api'
import { apiError, STAGES, stageValue } from '../utils/analysisUi'
import { Badge, ErrorNotice, Loading } from './AnalysisViews'
import { FeatureTable } from './ProcessDetails'

export default function ResearchExamples() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    let active = true
    api.getResearchExamples().then(result => {
      if (active) setData(result)
    }).catch(err => {
      if (active) setError(apiError(err))
    })
    return () => { active = false }
  }, [])

  if (error) return <ErrorNotice>{error}</ErrorNotice>
  if (!data) return <Loading />

  return <section className="surface-panel space-y-4">
    <h2 className="text-lg font-semibold">Dari ulasan ke prediksi</h2>
    <p className="text-sm text-slate-600">Contoh holdout diagnostik · model {data.model_version}. Label rujukan berasal dari dataset penelitian dengan pelabelan leksikon dan validasi pakar sebagian. Prediksi diambil dari hasil evaluasi tersimpan.</p>
    {data.examples.length === 0 && <p className="text-sm">Belum ada contoh dengan teks preprocessing yang cocok dengan hasil tersimpan.</p>}
    {data.examples.map(example => <details key={example.id} className="rounded-xl border border-primary-100 p-4">
      <summary className="cursor-pointer font-medium">Lihat proses ulasan #{example.id}</summary>
      <div className="mt-4 space-y-4">
        <p className="whitespace-pre-wrap break-words rounded bg-primary-50 p-3">{example.original}</p>
        <div className="flex flex-wrap gap-3 text-sm">
          <span>Label rujukan <Badge value={example.reference} /></span>
          <span>Prediksi tersimpan <Badge value={example.prediction} /></span>
        </div>
        <dl className="grid gap-3 sm:grid-cols-2">{STAGES.map(([key, title]) => <div key={key} className="rounded border p-3">
          <dt className="font-semibold text-sm">{title}</dt>
          <dd className="mt-1 break-words text-sm">{stageValue(example.steps[key])}</dd>
        </div>)}</dl>
        <h3 className="font-semibold">Pembobotan TF-IDF</h3>
        <FeatureTable data={example.tfidf} />
      </div>
    </details>)}
  </section>
}
