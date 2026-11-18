import React from 'react'
import { Link } from 'react-router-dom'
import { CLASSES, LABELS } from '../utils/analysisUi'

export const colors = { negatif: '#b91c1c', netral: '#1d4ed8', positif: '#15803d' }
export default function SentimentCharts({ distribution, caption = 'Distribusi sentimen', percentage = false, unit = 'ulasan', resultLink }) {
  const entries = Object.entries(distribution || {})
  const max = Math.max(1, ...entries.map(([, c]) => CLASSES.reduce((n, k) => n + (c[k] || 0), 0)))
  return <figure className="space-y-4" aria-label={caption}><figcaption className="font-semibold">{caption}</figcaption><div className="flex gap-4 flex-wrap text-xs">{CLASSES.map(k => <span key={k} className="flex gap-1 items-center"><i className="w-3 h-3 inline-block rounded" style={{ background: colors[k] }} />{LABELS[k]}</span>)}</div>{entries.map(([name, counts]) => {
    const total = CLASSES.reduce((n, k) => n + (counts[k] || 0), 0)
    return <div key={name} className="space-y-1"><div className="flex justify-between gap-2 text-sm"><span>{name}</span><span>n = {total}</span></div><div className="flex bg-gray-100 rounded h-7" aria-hidden={resultLink ? undefined : true}>{CLASSES.map(k => {
      const count = counts[k] || 0
      const width = 100 * count / (percentage ? total || 1 : max)
      const title = `${name} · ${LABELS[k]}: ${count}`
      const style = { width: `${width}%`, background: colors[k] }
      const className = 'flex items-center justify-center overflow-hidden text-white text-xs'
      const display = width >= 10 ? (percentage ? `${width.toFixed(0)}%` : count) : ''
      return resultLink && count
        ? <Link key={k} to={resultLink(name, k)} aria-label={title} title={title} className={className} style={style}>{display}</Link>
        : <div key={k} title={title} className={className} style={style}>{display}</div>
    })}</div><p className="text-xs text-gray-500">{CLASSES.map(k => `${LABELS[k]} ${counts[k] || 0}${percentage ? ` (${(100 * (counts[k] || 0) / (total || 1)).toFixed(1)}%)` : ''}`).join(' · ')}</p></div>
  })}<p className="text-xs text-gray-500">{percentage ? `Persentase dihitung terhadap jumlah ${unit} pada masing-masing kategori.` : `Panjang batang membandingkan jumlah ${unit}.`}</p></figure>
}
