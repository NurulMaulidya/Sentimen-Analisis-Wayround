import React, { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'

// Pages
import Dashboard from './pages/Dashboard'
import NewAnalysis from './pages/NewAnalysis'
import AnalysisDetail from './pages/AnalysisDetail'
import AnalysisPipeline from './pages/AnalysisPipeline'
import AnalysisResults from './pages/AnalysisResults'
import AnalysisData from './pages/AnalysisData'
import AnalysisTfidf from './pages/AnalysisTfidf'
import Research from './pages/Research'
import History from './pages/History'
import PredictNew from './pages/PredictNew'

// Components
import Sidebar from './components/Sidebar'
import Header from './components/Header'

const AppContent = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => window.innerWidth >= 1024)
  useEffect(() => {
    const media = window.matchMedia('(min-width: 1024px)')
    const update = () => setIsSidebarOpen(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  return (
    <div className="flex h-screen app-shell">
      <Sidebar isOpen={isSidebarOpen} onToggle={() => setIsSidebarOpen(!isSidebarOpen)} />
      <div className={`flex-1 flex flex-col min-w-0 ${isSidebarOpen ? 'ml-14 lg:ml-56' : 'ml-14'} transition-all duration-200`}>
        <Header onMenuClick={() => setIsSidebarOpen(!isSidebarOpen)} />
        <main className="flex-1 overflow-auto app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/new-analysis" element={<NewAnalysis />} />
            <Route path="/predict-new" element={<PredictNew />} />
            <Route path="/single-history/:predictionId" element={<PredictNew />} />
            <Route path="/research" element={<Research />} />
            <Route path="/history" element={<History />} />
            <Route path="/analysis/:id" element={<AnalysisDetail />} />
            <Route path="/analysis/:id/data/:kind" element={<AnalysisData />} />
            <Route path="/analysis/:id/tfidf" element={<AnalysisTfidf />} />
            <Route path="/analysis/:id/pipeline" element={<AnalysisPipeline />} />
            <Route path="/analysis/:id/pipeline/preprocessing" element={<AnalysisPipeline />} />
            <Route path="/analysis/:id/pipeline/preprocessing/:stage" element={<AnalysisPipeline />} />
            <Route path="/analysis/:id/pipeline/compare" element={<AnalysisPipeline />} />
            <Route path="/analysis/:id/results" element={<AnalysisResults />} />
            <Route path="/analysis/:id/results/predictions" element={<AnalysisResults />} />
            <Route path="/analysis/:id/results/aspects" element={<AnalysisResults />} />
            <Route path="/analysis/:id/results/evaluation" element={<AnalysisResults />} />
            <Route path="/upload" element={<Navigate to="/new-analysis" replace />} />
            <Route path="/preprocessing" element={<Navigate to="/" replace />} />
            <Route path="/pelabelan" element={<Navigate to="/" replace />} />
            <Route path="/evaluasi" element={<Navigate to="/research" replace />} />
            <Route path="/analisis-aspek" element={<Navigate to="/" replace />} />
            <Route path="/prediksi" element={<Navigate to="/predict-new" replace />} />
            <Route path="/riwayat-prediksi" element={<Navigate to="/history" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

const App = () => (
  <BrowserRouter>
    <AppContent />
  </BrowserRouter>
)

export default App
