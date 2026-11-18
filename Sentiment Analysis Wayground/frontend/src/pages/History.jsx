import React, { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import api from '../services/api'
import { Badge, ErrorNotice, Loading, Pagination, TableWrap } from '../components/AnalysisViews'
import { apiError, versionLabel, statusLabel } from '../utils/analysisUi'

export default function History() {
  const [params,setParams] = useSearchParams(), tab=params.get('tab') || 'batch'
  const [data,setData]=useState(null), [page,setPage]=useState(1), [error,setError]=useState(null), [busy,setBusy]=useState(true), [refresh,setRefresh]=useState(0)
  const [draft, setDraft] = useState('')
  const [search, setSearch] = useState('')
  const [order, setOrder] = useState('newest')
  const query = search.toLocaleLowerCase('id-ID')
  const singleSearch = tab === 'single' ? search : ''
  const singleOrder = tab === 'single' ? order : 'newest'
  const batchRows = (data?.analyses || []).filter(row =>
    (tab === 'batch' || (row.mode === tab && row.evaluation_scope !== 'legacy_unknown')) && [row.name, row.filename].some(value => value?.toLocaleLowerCase('id-ID').includes(query))
  ).sort((a, b) => {
    const byDate = (a.created_at || '').localeCompare(b.created_at || '') || a.id - b.id
    return order === 'oldest' ? byDate : -byDate
  })
  const rows = tab === 'single' ? data?.data || [] : batchRows

  const switchTab = next => {
    setPage(1)
    setDraft('')
    setSearch('')
    setParams({ tab: next })
  }
  useEffect(() => {
    let active = true
    setBusy(true)
    setError(null)
    setData(null)
    const request = tab === 'single' ? api.getSingleHistory(page, singleSearch, singleOrder) : api.getAnalyses()
    request.then(result => {
      if (active) setData(result)
    }).catch(err => {
      if (active) setError(apiError(err))
    }).finally(() => {
      if (active) setBusy(false)
    })
    return () => { active = false }
  }, [tab, page, refresh, singleSearch, singleOrder])
  const remove=async id=>{if(!window.confirm('Hapus hasil tersimpan ini?'))return;try{if(tab==='single')await api.deleteSingle(id);else await api.deleteAnalysis(id);setRefresh(v=>v+1)}catch(e){setError(apiError(e))}}
  return <div className="space-y-5"><header className="flex flex-wrap justify-between gap-3"><h1 className="text-2xl font-bold">Riwayat Analisis</h1><Link to="/new-analysis" className="bg-primary-600 text-white rounded px-4 py-2 text-sm">Analisis baru</Link></header><nav className="flex flex-wrap gap-2" aria-label="Jenis riwayat"><Link className="action-quiet" to="/research">Evaluasi Akhir Holdout</Link>{[['batch','Upload Dataset'],['new_data','Evaluasi Data Baru'],['research_demo','Analisis Data Pakar / Penelitian'],['single','Prediksi Tunggal']].map(([k,n])=><button key={k} className={`border rounded px-4 py-2 ${tab===k?'bg-primary-600 text-white':'bg-white'}`} aria-pressed={tab===k} onClick={()=>switchTab(k)}>{n}</button>)}</nav>
    <form className="flex flex-wrap items-end gap-3" onSubmit={event => {
      event.preventDefault()
      setPage(1)
      setSearch(draft.trim())
    }}>
      <label className="text-sm">Cari riwayat
        <input className="input-field mt-1 block" value={draft} onChange={event => setDraft(event.target.value)}
          placeholder={tab === 'single' ? 'Teks ulasan' : 'Nama analisis atau file'} />
      </label>
      <button className="action-quiet">Cari</button>
      <label className="text-sm">Urutkan
        <select className="input-field mt-1 block" value={order} onChange={event => { setPage(1); setOrder(event.target.value) }}>
          <option value="newest">Terbaru</option>
          <option value="oldest">Terlama</option>
        </select>
      </label>
    </form>
    {error && <ErrorNotice>{error}</ErrorNotice>}{busy?<Loading/>:data && <section className="bg-white border rounded-xl p-4"><TableWrap><table className="w-full text-sm text-left"><thead><tr>{(tab==='single'?['ID / Waktu','Ulasan','Sentimen / aspek','Versi model','Aksi']:['Analisis / File','Status','Upload / Prediksi / Evaluasi','Negatif / Netral / Positif','Versi model','Aksi']).map(k=><th className="p-3" key={k}>{k}</th>)}</tr></thead><tbody>{tab==='single'?rows.map(r=><tr key={r.id} className="border-t align-top"><td className="p-3">#{r.id}<p className="text-xs text-gray-500">{new Date(r.created_at).toLocaleString('id-ID')}</p></td><td className="p-3 min-w-48">{r.original}</td><td className="p-3"><Badge value={r.predicted_label}/><p className="text-xs mt-2">{r.detected_aspects.join(', ') || 'Tanpa aspek'}</p></td><td className="p-3">{r.model_version}</td><td className="p-3 space-y-2"><Link to={`/single-history/${r.id}`} className="block text-primary-700 underline">Lihat Detail</Link><button onClick={()=>remove(r.id)} className="text-red-700">Hapus</button></td></tr>):rows.map(a=><tr className="border-t align-top" key={a.id}><td className="p-3"><p className="font-semibold">{a.name}</p><p className="text-xs text-gray-500">#{a.id} · {a.filename}</p>
    <p className="text-xs text-gray-500">{a.created_at ? new Date(a.created_at + 'Z').toLocaleString('id-ID', { timeZone: 'Asia/Jakarta' }) : '—'}</p>
    <p className="text-xs text-gray-500">{a.evaluation_scope === 'legacy_unknown' ? 'Asal data belum tercatat' : a.mode === 'research_demo' ? 'Analisis deskriptif penelitian' : 'Pengujian data baru'}</p></td><td className="p-3">{statusLabel(a.status)}</td><td className="p-3">{a.original_count} / {a.prediction_count} / {a.evaluated_count ?? 0}</td><td className="p-3">{a.distribution ? ['negatif','netral','positif'].map(k=>a.distribution[k]).join(' / ') : '—'}</td><td className="p-3">{versionLabel(a)}</td><td className="p-3"><Link to={`/analysis/${a.id}`} className="block text-primary-700 underline">Buka</Link>{a.evaluation_ready && <Link to={`/analysis/${a.id}/results/evaluation`} className="block text-primary-700 underline">Lihat metrik</Link>}<button className="text-red-700 mt-2" onClick={()=>remove(a.id)}>Hapus</button></td></tr>)}{rows.length === 0 && <tr><td colSpan={tab === 'single' ? 5 : 6} className="p-4 text-slate-500">Tidak ada riwayat yang cocok.</td></tr>}</tbody></table></TableWrap>{tab==='single' && <Pagination pagination={data.pagination} onPage={setPage}/>}<p className="text-xs text-gray-500 mt-3">Detail berasal dari hasil tersimpan beserta versi modelnya. Membuka riwayat tidak menjalankan prediksi ulang.</p></section>}</div>
}
