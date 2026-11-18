import React, { useEffect, useRef, useState } from 'react'
import { LABELS, percent } from '../utils/analysisUi'

export default function ConfusionMatrixChart({ matrix, labels, csvUrl, descriptive }) {
  const canvas = useRef(null)
  const [normalized, setNormalized] = useState(false)
  useEffect(() => {
    const chart = canvas.current
    const ctx = chart?.getContext('2d')
    if (!ctx || !matrix?.length) return

    const left = 130
    const top = 98
    const width = 115
    const height = 68
    const max = Math.max(1, ...matrix.flat())
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, chart.width, chart.height)
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillStyle = '#172033'
    ctx.font = 'bold 19px sans-serif'
    ctx.fillText('Confusion matrix', chart.width / 2, 23)
    if (descriptive) {
      ctx.font = '13px sans-serif'
      ctx.fillText('Analisis deskriptif pada data penelitian', chart.width / 2, 46)
    }
    ctx.font = 'bold 14px sans-serif'
    ctx.fillText('Prediksi SVM', left + 1.5 * width, 65)
    ctx.fillText('Label aktual', 72, 83)

    labels.forEach((label, i) => {
      ctx.font = '14px sans-serif'
      ctx.fillStyle = '#172033'
      ctx.fillText(LABELS[label], left + (i + 0.5) * width, 83)
      ctx.fillText(LABELS[label], 72, top + (i + 0.5) * height)
      const total = matrix[i].reduce((sum, count) => sum + count, 0)
      matrix[i].forEach((count, j) => {
        const share = total ? count / total : 0
        const intensity = normalized ? share : count / max
        const x = left + j * width
        const y = top + i * height
        ctx.fillStyle = `rgba(104, 46, 135, ${0.06 + intensity * 0.87})`
        ctx.fillRect(x + 1, y + 1, width - 2, height - 2)
        ctx.fillStyle = intensity > 0.65 ? '#ffffff' : '#172033'
        ctx.font = i === j ? 'bold 15px sans-serif' : '15px sans-serif'
        ctx.fillText(normalized ? `${count} · ${total ? percent(share) : '—'}` : String(count),
          x + width / 2, y + height / 2)
      })
    })
  }, [matrix, labels, normalized, descriptive])

  const download = () => {
    const link = document.createElement('a')
    link.href = canvas.current.toDataURL('image/png')
    link.download = `confusion-matrix-${csvUrl.split('/')[4]}.png`
    link.click()
  }

  return <figure className="space-y-2">
    <canvas ref={canvas} width={500} height={315} className="max-w-full border rounded"
      role="img" aria-label="Heatmap confusion matrix. Nilai sel juga tersedia dalam tabel." />
    <figcaption className="flex flex-wrap gap-3 text-sm">
      <label><input type="checkbox" checked={normalized} onChange={event => setNormalized(event.target.checked)} /> Persentase per baris</label>
      <button className="text-primary-700 underline" onClick={download}>Unduh grafik PNG</button>
      <a className="text-primary-700 underline" href={csvUrl}>Unduh matriks CSV</a>
    </figcaption>
  </figure>
}
