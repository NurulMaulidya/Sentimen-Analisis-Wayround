import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Upload } from 'lucide-react'
import api from '../services/api'
import { ErrorNotice, Loading } from '../components/AnalysisViews'
import SentimentCharts, { colors } from '../components/SentimentCharts'
import { apiError, CLASSES, LABELS, versionLabel } from '../utils/analysisUi'

const number = new Intl.NumberFormat('id-ID')
const percent = new Intl.NumberFormat('id-ID', { maximumFractionDigits: 1, style: 'percent' })

function SentimentDonut({ counts, title }) {
  const total = CLASSES.reduce((sum, label) => sum + counts[label], 0)
  let start = 0
  const segments = CLASSES.map(label => {
    const end = start + (counts[label] / (total || 1)) * 100
    const segment = colors[label] + ' ' + start + '% ' + end + '%'
    start = end
    return segment
  })

  return <figure className="surface-panel space-y-4" aria-label={title}>
    <figcaption className="font-semibold">{title}</figcaption>
    {total ? <div className="flex flex-wrap items-center gap-5">
      <div className="w-36 h-36 shrink-0 rounded-full grid place-items-center"
        style={{ background: 'conic-gradient(' + segments.join(', ') + ')' }} aria-hidden="true">
        <div className="w-24 h-24 bg-white rounded-full grid place-content-center text-center">
          <strong className="text-xl">{number.format(total)}</strong>
          <span className="text-xs text-slate-500">ulasan</span>
        </div>
      </div>
      <ul className="flex-1 min-w-36 space-y-2 text-sm">
        {CLASSES.map(label => <li className="flex items-center justify-between gap-2" key={label}>
          <span className="flex items-center gap-2"><i className="w-3 h-3 rounded" style={{ background: colors[label] }} />
            {LABELS[label]}</span>
          <strong>{number.format(counts[label])} <span className="font-normal text-slate-500">({percent.format(counts[label] / total)})</span></strong>
        </li>)}
      </ul>
    </div> : <p className="text-sm text-slate-500">Belum ada ulasan berlabel pada sumber ini.</p>}
  </figure>
}

function AspectCounts({ counts }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1])
  const maximum = Math.max(1, ...entries.map(([, count]) => count))
  return <figure className="surface-panel space-y-4" aria-label="Distribusi aspek">
    <figcaption className="font-semibold">Distribusi aspek</figcaption>
    <ul className="space-y-3">
      {entries.map(([aspect, count]) => <li key={aspect}>
        <div className="flex justify-between gap-3 mb-1 text-sm"><span>{aspect}</span><strong>{number.format(count)}</strong></div>
        <div className="h-2.5 bg-slate-100 rounded overflow-hidden" aria-hidden="true">
          <div className="h-full rounded bg-primary-600" style={{ width: 100 * count / maximum + '%' }} />
        </div>
      </li>)}
    </ul>
    <p className="text-xs text-slate-500">Jumlah ulasan yang menyebut tiap aspek. Satu ulasan dapat dihitung pada beberapa aspek.</p>
  </figure>
}

function StatCards({ items }) {
  return <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
    {items.map(([label, value]) => <div className="stat-tile" key={label}>
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className="text-2xl font-semibold mt-2 text-slate-900">{number.format(value)}</p>
    </div>)}
  </div>
}

function AspectInsights({ counts, sentiments, unit, resultLink }) {
  const mostMentioned = Object.entries(counts || {}).filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1])[0]
  const mostNegative = Object.entries(sentiments || {}).map(([aspect, values]) => ({
    aspect, negative: values.negatif || 0, total: CLASSES.reduce((sum, label) => sum + (values[label] || 0), 0),
  })).filter(row => row.negative > 0).sort((a, b) => b.negative / b.total - a.negative / a.total || b.total - a.total)[0]
  if (!mostMentioned && !mostNegative) return null
  return <section className="surface-panel space-y-3" aria-label="Temuan utama">
    <h2 className="font-semibold">Temuan utama</h2>
    <div className="grid gap-3 sm:grid-cols-2">
      {mostMentioned && <div className="rounded-xl bg-primary-50 p-4">
        <p className="text-sm text-primary-800">Aspek paling sering disebut</p>
        <p className="mt-1 font-semibold">{mostMentioned[0]}</p>
        <p className="text-sm">{number.format(mostMentioned[1])} ulasan</p>
      </div>}
      {mostNegative && <div className="rounded-xl bg-primary-50 p-4">
        <p className="text-sm text-primary-800">Proporsi prediksi negatif tertinggi</p>
        <p className="mt-1 font-semibold">{mostNegative.aspect} · {percent.format(mostNegative.negative / mostNegative.total)}</p>
        <p className="text-sm">{number.format(mostNegative.negative)} dari {number.format(mostNegative.total)} {unit}</p>
        {resultLink && <Link className="mt-2 inline-block text-sm font-medium text-primary-800 underline"
          to={resultLink(mostNegative.aspect, 'negatif')}>Baca ulasan terkait</Link>}
      </div>}
    </div>
  </section>
}

export default function Dashboard() {
  const [research, setResearch] = useState(null)
  const [activity, setActivity] = useState(null)
  const [errors, setErrors] = useState({})
  const [source, setSource] = useState('research')
  const [asPercentage, setAsPercentage] = useState(false)

  useEffect(() => {
    let active = true
    api.getModelStats().then(response => { if (active) setResearch(response.stats) })
      .catch(error => { if (active) setErrors(previous => ({ ...previous, research: apiError(error) })) })
    api.getStats().then(response => { if (active) setActivity(response.stats) })
      .catch(error => { if (active) setErrors(previous => ({ ...previous, upload: apiError(error) })) })
    return () => { active = false }
  }, [])

  const isResearch = source === 'research'
  const data = isResearch ? research : activity
  const latest = activity?.latest_analysis
  const aspectUnit = isResearch ? 'kalimat aspek' : 'pasangan ulasan–aspek dengan sentimen'
  const resultLink = !isResearch && latest
    ? (aspect, sentiment) => `/analysis/${latest.id}/results/aspects?${new URLSearchParams({ aspect, sentiment })}`
    : undefined
  const labels = isResearch && research
    ? { negatif: research.negative, netral: research.neutral, positif: research.positive }
    : activity?.distribution
  const cards = isResearch && research
    ? [['Dataset berlabel', research.labeled_count], ['Development', research.train_samples],
    ['Holdout diagnostik', research.test_samples], ['Data sumber', research.raw_samples]]
    : [['Sesi analisis', activity?.total_analyses], ['Ulasan diprediksi', latest?.prediction_count],
    ['Analisis selesai', activity?.completed_analyses], ['Prediksi tunggal', activity?.single_predictions]]

  return <div className="space-y-5 dashboard-page">
    <header className="dashboard-hero rounded-2xl p-5 pb-6 sm:p-7 sm:pb-8 flex flex-wrap items-end justify-between gap-4">
      <div>
        <span className="eyebrow">WAYGROUND · SENTIMENT ANALYSIS</span>
        <h1 className="mt-2 text-2xl sm:text-3xl font-semibold tracking-tight">Lakukan Uji Coba Analisis Ulasan</h1>
        <p className="mt-2 text-sm">Ringkasan sentimen dan aspek dari penelitian Wayground atau unggahan terakhir.</p>
      </div>
      <div className="flex flex-wrap gap-2">
        <Link to="/new-analysis" className="action-primary"><Upload size={17} />Analisis Data Baru</Link>
        <Link to="/predict-new" className="action-quiet">Coba satu ulasan <ArrowRight size={15} /></Link>
      </div>
    </header>

    <section className="space-y-4" aria-label="Ringkasan dan grafik">
      <div className="flex flex-wrap gap-2" role="group" aria-label="Sumber grafik">
        {[['research', 'Dataset penelitian'], ['upload', 'Unggahan terakhir']].map(([key, title]) =>
          <button key={key} type="button" aria-pressed={source === key}
            className={'rounded-lg border px-3 py-2 text-sm font-medium ' + (source === key ? 'border-primary-800 bg-primary-800 text-white' : 'border-primary-100 bg-white text-primary-800 hover:bg-primary-50')}
            onClick={() => setSource(key)}>{title}</button>)}
      </div>
      {errors[source] ? <ErrorNotice>{errors[source]}</ErrorNotice> : !data ? <Loading /> : !isResearch && !latest ?
        <div className="surface-panel"><h2 className="font-semibold">Belum ada analisis unggahan yang selesai</h2>
          <p className="mt-2 text-sm text-slate-500">Selesaikan analisis file untuk melihat grafik dari sesi tersebut.</p>
          <Link to="/new-analysis" className="inline-block mt-3 text-primary-700 underline text-sm">Mulai analisis</Link>
        </div> : <>
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-slate-600">
            <p>{isResearch
              ? 'Dataset penelitian · model ' + research.model_version + ' · label gabungan leksikon dan pakar'
              : latest.name + ' · model ' + versionLabel(latest)}</p>
            <Link to={isResearch ? '/research' : '/analysis/' + latest.id} className="text-primary-700 font-medium">
              {isResearch ? 'Buka evaluasi penelitian →' : 'Buka hasil unggahan →'}
            </Link>
          </div>
          <StatCards items={cards} />
          <div className="grid lg:grid-cols-2 gap-4">
            <SentimentDonut counts={labels} title={isResearch ? 'Distribusi label dataset' : 'Distribusi prediksi sentimen'} />
            {data.aspect_counts ? <AspectCounts counts={data.aspect_counts} />
              : <div className="surface-panel"><h2 className="font-semibold">Distribusi aspek</h2>
                <p className="mt-2 text-sm text-slate-500">Ringkasan aspek belum tersedia untuk versi penelitian ini.</p></div>}
          </div>
          <AspectInsights counts={data.aspect_counts} sentiments={data.aspect_sentiment} unit={aspectUnit} resultLink={resultLink} />
          {data.aspect_sentiment && Object.values(data.aspect_sentiment).some(counts => CLASSES.some(label => counts[label] > 0)) &&
            <div className="surface-panel space-y-3">
              <div className="flex justify-between items-center flex-wrap gap-2">
                <p className="text-sm text-slate-500">{isResearch ? 'Prediksi pada kalimat yang terdeteksi untuk tiap aspek.' : 'Prediksi pada pasangan ulasan dan aspek.'}</p>
                <button type="button" className="action-quiet" aria-pressed={asPercentage}
                  onClick={() => setAsPercentage(value => !value)}>{asPercentage ? 'Tampilkan jumlah' : 'Tampilkan persentase'}</button>
              </div>
              <SentimentCharts distribution={data.aspect_sentiment} caption="Sentimen per aspek"
                percentage={asPercentage} unit={aspectUnit} resultLink={resultLink} />
            </div>}
        </>}
    </section>
  </div>
}
