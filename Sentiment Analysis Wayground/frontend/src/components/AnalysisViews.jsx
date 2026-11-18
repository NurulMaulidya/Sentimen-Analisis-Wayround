import React from 'react'
import ConfusionMatrixChart from './ConfusionMatrixChart'
import SentimentCharts from './SentimentCharts'
import { CLASSES, LABELS, percent, stageValue } from '../utils/analysisUi'

export const ErrorNotice = ({ children }) => <div role="alert" className="p-3 rounded-lg bg-red-50 text-red-800 border border-red-200">{children}</div>
export const Loading = () => <p role="status" className="p-6 text-gray-500">Memuat data…</p>
export const Badge = ({ value }) => <span className={`inline-block px-2 py-1 rounded text-xs ${value === 'positif' ? 'bg-green-100 text-green-800' : value === 'negatif' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'}`}>{LABELS[value] || '—'}</span>
export const TableWrap = ({ children }) => <div className="overflow-x-auto border border-gray-200 rounded-lg">{children}</div>
export const Pagination = ({ pagination, onPage, disabled }) => !pagination ? null : <div className="flex flex-wrap items-center justify-between gap-2 text-sm mt-3">
  <span>{pagination.total} baris · Halaman {pagination.total_pages ? pagination.page : 0} dari {pagination.total_pages}</span>
  <div className="flex gap-2"><button className="px-3 py-1 border rounded disabled:opacity-40" disabled={disabled || pagination.page <= 1} onClick={() => onPage(pagination.page - 1)}>Sebelumnya</button><button className="px-3 py-1 border rounded disabled:opacity-40" disabled={disabled || pagination.page >= pagination.total_pages} onClick={() => onPage(pagination.page + 1)}>Berikutnya</button></div>
</div>

export function PredictionView({ data, onReview }) {
  return <div className="space-y-4">
    <p className="font-semibold">Total diprediksi: {Object.values(data.distribution || {}).reduce((n,v)=>n+v,0)} ulasan</p>
    <div className="grid grid-cols-3 gap-3">{CLASSES.map(label => <div className="p-3 bg-gray-50 rounded-lg text-center" key={label}><p className="text-2xl font-bold">{data.distribution?.[label] ?? '—'}</p><p>{LABELS[label]}</p></div>)}</div>
    <SentimentCharts distribution={{ 'Seluruh hasil sesi': data.distribution || {} }} caption="Distribusi Labeling SVM" />
    {data.excluded_zero_vectors > 0 && <p className="text-sm text-amber-800">{data.excluded_zero_vectors} ulasan tidak diprediksi karena tidak memiliki fitur TF-IDF yang dikenali.</p>}
    {data.excluded_empty_processed > 0 && <p className="text-sm text-amber-800">{data.excluded_empty_processed} ulasan tidak diprediksi karena kosong setelah preprocessing.</p>}
    <TableWrap><table className="w-full text-sm text-left"><thead className="bg-gray-50"><tr>{['ID', 'Ulasan', 'Teks hasil preprocessing', 'Lexicon', 'Prediksi SVM', data.mode === 'research_demo' ? 'Label pakar' : 'Label aktual', 'Kecocokan', 'Asal data', 'Detail'].map(x => <th className="p-3" key={x}>{x}</th>)}</tr></thead><tbody>
      {data.data?.map(row => <tr className="border-t align-top" key={row.id}><td className="p-3">{row.id}</td><td className="p-3 min-w-56 whitespace-pre-wrap break-words">{row.original_content}</td><td className="p-3 min-w-48 break-words">{row.final_processed_text}</td><td className="p-3"><Badge value={row.lexicon_label} /></td><td className="p-3"><Badge value={row.predicted_label} /></td><td className="p-3">{row.actual_label ? <Badge value={row.actual_label} /> : '—'}</td><td className="p-3">{row.is_correct == null ? '—' : row.is_correct ? 'Cocok' : 'Tidak cocok'}</td><td className="p-3">{row.data_origin || '—'}</td><td className="p-3">{onReview && <button className="text-primary-700 underline whitespace-nowrap" onClick={()=>onReview(row.id)}>Lihat Proses</button>}</td></tr>)}
      {!data.data?.length && <tr><td className="p-4 text-gray-500" colSpan={9}>Tidak ada ulasan yang cocok dengan filter.</td></tr>}
    </tbody></table></TableWrap>
  </div>
}

export function AspectView({ data, onReview }) {
  return <div className="space-y-4">
    <p className="text-sm text-gray-600">Sentimen tiap aspek dihitung dari kalimat aspek itu sendiri memakai model SVM yang sama, bukan disalin dari sentimen ulasan. Satu ulasan dapat memuat beberapa aspek.</p>
    <p>{data.summary?.total_reviews_with_aspects ?? 0} ulasan unik · {data.summary?.total_aspect_mentions ?? 0} penyebutan aspek</p>
    {data.flow && <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">{[['Diprediksi',data.flow.predicted],['Dengan aspek',data.flow.with_aspects],['Tanpa aspek',data.flow.without_aspects],['Pasangan ulasan–aspek',data.summary?.total_aspect_mentions]].map(([k,v])=><div className="bg-slate-50 rounded p-3" key={k}><p className="text-xl font-bold">{v ?? '—'}</p><p className="text-xs text-gray-600">{k}</p></div>)}</div>}
    <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">{Object.entries(data.distribution || {}).map(([name,c])=><div className="border rounded-lg p-3" key={name}><h3 className="text-sm font-semibold">{name}</h3><p className="text-2xl font-bold mt-1">{c.total}</p><p className="text-xs text-gray-500">ulasan unik pada aspek ini</p><p className="text-xs mt-2">Negatif {c.negatif} · Netral {c.netral} · Positif {c.positif}</p></div>)}</div>
    <SentimentCharts distribution={data.distribution} caption="Sentimen per aspek" />
    {data.review_rows && <TableWrap><table className="w-full text-sm text-left"><caption className="p-3 font-semibold text-left">Hasil aspek setiap ulasan</caption><thead><tr>{['ID','Ulasan','Sentimen ulasan','Aspek terdeteksi','Kalimat terkait'].map(x=><th key={x} className="p-3">{x}</th>)}</tr></thead><tbody>{data.review_rows.map(r=><tr className="border-t align-top" key={r.id}><td className="p-3">{r.id}</td><td className="p-3 min-w-48">{r.original_content}{onReview && <button onClick={()=>onReview(r.id)} className="block text-primary-700 underline mt-2">Lihat Proses</button>}</td><td className="p-3"><Badge value={r.predicted_label}/></td><td className="p-3">{r.detected_aspects?.length ? r.detected_aspects.map(a=><span key={a} className="block text-xs rounded bg-purple-50 p-1 mb-1">{a}</span>) : 'Tidak terdeteksi'}</td><td className="p-3 min-w-48">{r.aspect_details?.map(a=><details key={a.id}><summary className="cursor-pointer text-primary-700">{a.aspect_name} · {LABELS[a.sentiment] || 'Belum diprediksi'}</summary><p className="whitespace-pre-wrap text-xs">{a.sentences.join(' / ')}</p></details>)}</td></tr>)}</tbody></table></TableWrap>}
    <TableWrap><table className="w-full text-sm text-left"><thead className="bg-gray-50"><tr>{['Aspek', 'Negatif', 'Netral', 'Positif', 'Total penyebutan'].map(x => <th className="p-3" key={x}>{x}</th>)}</tr></thead><tbody>{Object.entries(data.distribution || {}).map(([aspect, counts]) => <tr className="border-t" key={aspect}><th className="p-3">{aspect}</th>{[...CLASSES, 'total'].map(label => <td className="p-3" key={label}>{counts[label] ?? '—'}</td>)}</tr>)}</tbody></table></TableWrap>
    {!data.review_rows && <>
    <TableWrap><table className="w-full text-sm text-left"><thead className="bg-gray-50"><tr><th className="p-3">Aspek</th><th className="p-3">Kalimat yang memuat aspek</th><th className="p-3">Sentimen aspek</th></tr></thead><tbody>{data.data?.map(row => <tr className="border-t align-top" key={row.id}><td className="p-3">{row.aspect_name}</td><td className="p-3 whitespace-pre-wrap">{(row.sentences || []).join(' / ')}</td><td className="p-3"><Badge value={row.sentiment} /></td></tr>)}{!data.data?.length && <tr><td className="p-4 text-gray-500" colSpan={3}>Tidak ada penyebutan aspek pada filter ini.</td></tr>}</tbody></table></TableWrap>
    </>}
  </div>
}

export function EvaluationView({ evaluation, scope = 'upload', matrixExportUrl }) {
  const labels = evaluation.confusion_matrix_labels || CLASSES
  const report = evaluation.classification_report || {}
  const descriptive = evaluation.scope === 'research_descriptive'
  const missingClasses = labels.filter(label => (report[label] || report[LABELS[label]])?.support === 0)
  return <div className="space-y-4">
    <p className="text-sm text-gray-600">{scope === 'research' ? 'Evaluasi penelitian tersimpan: diagnostic holdout dengan paparan historis.' : descriptive ? 'Analisis deskriptif pada data penelitian. Sebagian ulasan dapat berasal dari development atau holdout. Hasil ini bukan evaluasi independen.' : evaluation.scope === 'legacy_unknown' ? 'Riwayat evaluasi lama. Asal dataset belum tercatat.' : 'Evaluasi data baru berdasarkan label aktual pada file yang diunggah.'}</p>
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">{[['accuracy', 'Accuracy'], ['precision_macro', 'Macro precision'], ['recall_macro', 'Macro recall'], ['f1_macro', 'Macro F1']].map(([key, label]) => <div className="p-3 bg-gray-50 rounded-lg" key={key}><p className="text-xs text-gray-600">{label}</p><p className="text-xl font-bold">{percent(evaluation.metrics?.[key])}</p></div>)}</div>
    <p>Dievaluasi: {evaluation.correct_count + evaluation.incorrect_count} ulasan · Cocok: {evaluation.correct_count} · Tidak cocok: {evaluation.incorrect_count}</p>
    {missingClasses.length > 0 && <p className="rounded bg-amber-50 p-3 text-sm text-amber-900">Kelas aktual tanpa sampel evaluasi: {missingClasses.map(label => LABELS[label] || label).join(', ')}. Macro-average mencakup ketiga kelas.</p>}
    {matrixExportUrl && <ConfusionMatrixChart matrix={evaluation.confusion_matrix} labels={labels}
      csvUrl={matrixExportUrl} descriptive={descriptive} />}
    <TableWrap><table className="w-full text-sm text-center"><caption className="p-3 text-left font-semibold">Confusion matrix — baris aktual, kolom prediksi</caption><thead className="bg-gray-50"><tr><th className="p-3">Aktual / Prediksi</th>{labels.map(label => <th className="p-3" scope="col" key={label}>{LABELS[label] || label}</th>)}</tr></thead><tbody>{labels.map((label, i) => <tr className="border-t" key={label}><th className="p-3" scope="row">{LABELS[label] || label}</th>{labels.map((other, j) => {
      const row = evaluation.confusion_matrix?.[i] || []
      const count = row[j]
      const total = row.reduce((sum, value) => sum + value, 0)
      const share = count == null || !total ? null : count / total
      return <td className={`p-3 ${i === j ? 'font-semibold' : ''}`} key={other}
        style={{ backgroundColor: `rgba(128, 56, 116, ${(share ?? 0) * 0.24})` }}
        title={share == null ? undefined : `${percent(share)} dari kelas aktual`}>{count ?? '—'}</td>
    })}</tr>)}</tbody></table></TableWrap>
    <TableWrap><table className="w-full text-sm text-left"><caption className="p-3 text-left font-semibold">Classification report</caption><thead className="bg-gray-50"><tr>{['Kelas', 'Precision', 'Recall', 'F1', 'Support'].map(x => <th className="p-3" key={x}>{x}</th>)}</tr></thead><tbody>{labels.map(label => { const row = report[label] || report[LABELS[label]]; return <tr className="border-t" key={label}><th className="p-3">{LABELS[label] || label}</th><td className="p-3">{percent(row?.precision)}</td><td className="p-3">{percent(row?.recall)}</td><td className="p-3">{percent(row?.['f1-score'])}</td><td className="p-3">{row?.support ?? '—'}</td></tr> })}</tbody></table></TableWrap>
    {!matrixExportUrl && evaluation.actual_vs_predicted?.some(r=>!r.correct) && <TableWrap><table className="w-full text-sm text-left"><caption className="p-3 text-left font-semibold">Ulasan salah klasifikasi</caption><thead><tr><th className="p-3">Ulasan</th><th>Aktual</th><th>Prediksi</th></tr></thead><tbody>{evaluation.actual_vs_predicted.filter(r=>!r.correct).map((r,i)=><tr key={r.id || i} className="border-t"><td className="p-3">{r.original_content || r.id || 'Riwayat lama tanpa teks'}</td><td className="p-3"><Badge value={r.actual}/></td><td className="p-3"><Badge value={r.predicted}/></td></tr>)}</tbody></table></TableWrap>}

  </div>
}

export function StageTable({ rows, stage, compare, stages, onReview }) {
  return <TableWrap><table className="w-full text-sm text-left"><thead className="bg-gray-50"><tr><th className="p-3">Ulasan asli</th>{(compare ? stages : [stages.find(([key]) => key === stage)]).filter(Boolean).map(([key, label]) => <th className="p-3 min-w-48" key={key}>{label}</th>)}</tr></thead><tbody>{rows.map(row => <tr className="border-t align-top" key={row.id}><td className="p-3 min-w-64 whitespace-pre-wrap">{row.original_content}{onReview && <button onClick={()=>onReview(row.id)} className="block text-primary-700 underline mt-2">Lihat Semua Tahapan</button>}</td>{(compare ? stages : [[stage]]).map(([key]) => <td className="p-3 min-w-48 break-words" key={key}>{stageValue(row[key])}</td>)}</tr>)}{!rows.length && <tr><td className="p-4 text-gray-500" colSpan={compare ? stages.length + 1 : 2}>Tidak ada data.</td></tr>}</tbody></table></TableWrap>
}
