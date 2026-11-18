import React from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  History,
  MessageSquare,
  BarChart3,
} from 'lucide-react'

const menuItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/new-analysis', icon: Upload, label: 'Analisis Data Uji' },
  { path: '/predict-new', icon: MessageSquare, label: 'Prediksi Satu Ulasan' },
  { path: '/history', icon: History, label: 'Riwayat Analisis' },
  { path: '/research', icon: BarChart3, label: 'Penelitian & Evaluasi' },
]

const Sidebar = ({ isOpen, onToggle }) => {
  const location = useLocation()

  const isActive = (path) => {
    if (path === '/') return location.pathname === '/'
    if (path === '/new-analysis') return location.pathname === path || location.pathname.startsWith('/analysis/')
    if (path === '/predict-new') return location.pathname.startsWith('/single-history/') || location.pathname === path
    return location.pathname.startsWith(path)
  }

  return (
    <aside
      className={`fixed left-0 top-0 h-screen app-sidebar text-white z-40 transition-all duration-200 ${
        isOpen ? 'w-56' : 'w-14'
      }`}
    >
      {/* Logo */}
      <div className="h-20 flex items-center px-3 border-b border-white/10">
        <Link to="/" aria-label="Wayground — Dashboard" className="flex items-center gap-2">
          <img src="/wayground-logo.png" alt="" width="32" height="32"
            className="w-8 h-8 rounded-lg flex-shrink-0 ring-1 ring-white/20" />
          {isOpen && (
            <div><span className="text-sm font-bold tracking-wide">Wayground</span><p className="text-[10px] text-slate-400 tracking-widest mt-1">SENTIMENT WORKSPACE</p></div>
          )}
        </Link>
      </div>

      {/* Menu */}
      <nav className="p-2 pt-6 h-[calc(100vh-5rem)] overflow-y-auto">
        <ul className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon
            const active = isActive(item.path)

            return (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  title={item.label}
                  aria-label={item.label}
                  className={`flex items-center gap-2 px-2 py-3 rounded-xl transition-colors ${
                    active
                      ? 'sidebar-active text-white'
                      : 'text-primary-100 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  {isOpen && (
                    <span className="text-sm font-medium truncate">{item.label}</span>
                  )}
                  {!isOpen && (
                    <div className="absolute left-full ml-2 px-2 py-1 bg-slate-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none hidden">
                      {item.label}
                    </div>
                  )}
                </NavLink>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Toggle */}
      <button
        onClick={onToggle}
        aria-label={isOpen ? 'Ringkas navigasi' : 'Perluas navigasi'}
        aria-expanded={isOpen}
        className="absolute -right-3 top-24 w-6 h-6 bg-white border border-slate-200 rounded-full flex items-center justify-center shadow hover:bg-slate-50"
      >
        <svg className={`w-3 h-3 text-slate-600 transition-transform ${isOpen ? '' : 'rotate-180'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>
    </aside>
  )
}

export default Sidebar
