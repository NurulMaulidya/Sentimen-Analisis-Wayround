import React from 'react'
import { Menu, ArrowUpRight } from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'

export default function Header({ onMenuClick }) {
  const { pathname } = useLocation()
  const title = pathname === '/' ? 'Ringkasan' : pathname.startsWith('/analysis/') ? 'Ruang analisis' : pathname.startsWith('/research') ? 'Hasil penelitian' : pathname.startsWith('/history') ? 'Riwayat' : 'Analisis ulasan'
  return <header className="app-header"><div className="flex items-center gap-3"><button aria-label="Buka atau tutup navigasi" onClick={onMenuClick} className="lg:hidden p-2 rounded-lg hover:bg-slate-100"><Menu size={19}/></button><span className="text-sm text-slate-500">Wayground <span className="mx-2 text-slate-300">/</span><span className="font-medium text-slate-800">{title}</span></span></div><Link className="hidden sm:flex items-center gap-2 text-xs font-medium text-slate-500 hover:text-primary-700" to="/research">Tentang model <ArrowUpRight size={14}/></Link></header>
}
