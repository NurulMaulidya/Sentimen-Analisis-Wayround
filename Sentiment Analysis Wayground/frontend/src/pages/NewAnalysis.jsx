import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'
import { ErrorNotice, Loading, TableWrap } from '../components/AnalysisViews'
import { apiError } from '../utils/analysisUi'

export default function NewAnalysis() {
  const navigate = useNavigate()
  const [mode, setMode] = useState('new_data')
  const labelName = mode === 'research_demo' ? 'Label pakar' : 'Label aktual'
  const [file, setFile] = useState(null), [name, setName] = useState(''), [preview, setPreview] = useState(null)
  const [content, setContent] = useState(''), [label, setLabel] = useState('__none__'), [busy, setBusy] = useState(false), [error, setError] = useState(null)
  const [uploadIssues, setUploadIssues] = useState(null)
  const issues = uploadIssues || preview?.label_issues?.[label]
  const inspect = async event => {
    const selected = event.target.files?.[0]
    setPreview(null); setError(null); setFile(null); setUploadIssues(null)
    if (!selected) return
    if (!/\.(csv|tsv|txt)$/i.test(selected.name) || selected.size > 5 * 1024 * 1024) { setError('Gunakan CSV/TSV/TXT maksimal 5 MB.'); return }
    setBusy(true)
    try {
      const result = await api.inspectUpload(selected)
      setFile(selected); setPreview(result)
      const reviewColumn = result.content_column || result.columns[0]
      setContent(reviewColumn)
      const candidates = result.label_candidates.filter(col => col !== reviewColumn)
      setLabel(candidates.length > 1 ? '' : candidates[0] || '__none__')
    } catch (e) { setError(apiError(e)) } finally { setBusy(false) }
  }
  const upload = async () => {
    setBusy(true); setError(null); setUploadIssues(null)
    try { const result = await api.uploadAnalysis(file, name.trim() || file.name, content, label, mode); navigate(`/analysis/${result.analysis.id}/data/original`) }
    catch (e) { setError(apiError(e)); setUploadIssues(e.response?.data?.label_issues) } finally { setBusy(false) }
  }
  return <div className="max-w-4xl mx-auto space-y-5"><h1 className="text-2xl font-bold">Analisis Data Wayground</h1><p className="text-sm text-gray-600">Upload → Validasi & Data Awal → 7 Preprocessing → Lexicon → TF-IDF → Labeling SVM → Analisis Aspek → Ringkasan</p><section className="bg-white border rounded-xl p-5 space-y-4"><fieldset className="space-y-3">
    <legend className="font-semibold mb-2">Mode analisis</legend>
    <label className="block text-sm">
      <input type="radio" name="mode" value="new_data" checked={mode === 'new_data'} onChange={e => setMode(e.target.value)} disabled={busy} />
      <span className="ml-2 font-medium">Pengujian Data Baru</span>
      <span className="block ml-5 text-gray-600">Untuk ulasan di luar development dan holdout canonical. Metrik memakai label aktual di file ini.</span>
    </label>
    <label className="block text-sm">
      <input type="radio" name="mode" value="research_demo" checked={mode === 'research_demo'} onChange={e => setMode(e.target.value)} disabled={busy} />
      <span className="ml-2 font-medium">Demonstrasi Data Penelitian</span>
      <span className="block ml-5 text-gray-600">Menerima form validasi pakar dan kedua split canonical. Metrik merupakan analisis deskriptif pada data penelitian.</span>
    </label>
  </fieldset><label className="block text-sm font-medium">Nama analisis<input className="block border rounded p-2 w-full mt-1" value={name} onChange={e => setName(e.target.value)} disabled={busy} placeholder="Contoh: Uji ulasan Wayground Oktober" /></label><label className="block border-2 border-dashed rounded-lg p-6 text-center cursor-pointer"><span className="block font-semibold">Pilih file dataset</span><span className="block text-sm text-gray-500 mt-1">CSV, TSV, TXT (maks. 5MB)</span><input aria-label="File dataset" className="block w-full mt-4 text-sm" type="file" accept=".csv,.tsv,.txt" onChange={inspect} disabled={busy} /></label>{file && <p className="text-sm">{file.name} · {preview?.total} baris</p>}{error && <ErrorNotice>{error}</ErrorNotice>}{busy && <Loading />}{preview && <><div className="grid sm:grid-cols-2 gap-4"><label className="text-sm">Kolom ulasan<select aria-label="Kolom ulasan" disabled={busy} value={content} onChange={e => setContent(e.target.value)} className="block border rounded p-2 w-full mt-1">{preview.columns.map(c => <option key={c}>{c}</option>)}</select></label><label className="text-sm">{labelName} (opsional)<select aria-label={`Kolom ${labelName.toLowerCase()}`} disabled={busy} value={label} onChange={e => { setLabel(e.target.value); setUploadIssues(null); setError(null) }} className="block border rounded p-2 w-full mt-1"><option value="" disabled>Pilih kolom label</option><option value="__none__">Tanpa {labelName.toLowerCase()}</option>{preview.columns.filter(c => c !== content).map(c => <option key={c}>{c}</option>)}</select></label></div>
    {!label && <p className="text-sm text-amber-800">Beberapa kolom label terdeteksi. Pilih kolom yang berisi label pembanding. Untuk form pakar, pilih label final pakar jika tersedia.</p>}
    {issues && [['empty', 'Label kosong'], ['invalid', 'Label tidak dikenal atau ambigu']].map(([key, title]) => issues[key]?.length > 0 &&
      <details key={key} className="text-sm text-amber-800">
        <summary>{title} · {issues[key].length} baris</summary>
        <p className="break-words">Nomor baris data {issues[key].join(', ')}</p>
      </details>
    )}
    <p className="text-sm text-gray-500">Pratinjau 5 baris. Seluruh data tersedia setelah upload, termasuk baris yang tidak diproses. Label kosong tetap dapat diprediksi; label tidak dikenal harus diperbaiki.</p><TableWrap><table className="w-full text-sm text-left"><thead><tr>{preview.columns.map(c => <th className="p-2" key={c}>{c}</th>)}</tr></thead><tbody>{preview.preview.map((r, i) => <tr key={i} className="border-t">{preview.columns.map(c => <td className="p-2 whitespace-pre-wrap" key={c}>{r[c] || '—'}</td>)}</tr>)}</tbody></table></TableWrap><button className="bg-primary-600 text-white rounded px-4 py-2 disabled:opacity-40" disabled={busy || !content || !label || label === content || preview.label_issues?.[label]?.invalid.length > 0} onClick={upload}>Upload & Lanjutkan</button></>}</section><section className="text-sm bg-primary-50 border border-primary-100 rounded-lg p-4"><p className="font-semibold">Format contoh</p><pre className="mt-2 whitespace-pre-wrap">{'content,label\nAplikasi error tidak bisa login,negatif\nSangat membantu belajar,positif'}</pre><p className="mt-2">Label yang didukung adalah negatif, netral, dan positif. Kedua mode menggunakan TF-IDF dan SVM yang sudah dilatih secara offline.</p></section></div>
}
