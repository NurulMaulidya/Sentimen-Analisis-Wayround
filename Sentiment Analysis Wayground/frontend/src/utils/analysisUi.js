export const CLASSES = ['negatif', 'netral', 'positif']
export const LABELS = { negatif: 'Negatif', netral: 'Netral', positif: 'Positif' }
export const STAGES = [
  ['cleaning', 'Cleaning'], ['case_folding', 'Case folding'], ['tokenizing', 'Tokenisasi'],
  ['normalization', 'Normalisasi'], ['negation', 'Penanganan negasi'],
  ['stopwords', 'Stopword removal'], ['stemming', 'Stemming'], ['final', 'Hasil akhir']
]
export const predicted = a => Boolean(a?.predictions_ready ?? ['predicted', 'evaluated', 'aspects_completed', 'completed'].includes(a?.status))
export const resultTab = path => ['predictions', 'aspects', 'evaluation'].find(tab => path.endsWith(`/results/${tab}`)) || 'predictions'
export const percent = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(2)}%`
export const showValue = value => Array.isArray(value) ? value.join(' · ') || '—' : value == null || value === '' ? '—' : String(value)
export const stageValue = value => value == null ? 'Belum diproses' : Array.isArray(value) ? value.join(' · ') || '[] (kosong)' : value === '' ? '(kosong)' : String(value)
export const apiError = (error, fallback = 'Permintaan gagal') => error?.response?.data?.error || error?.message || fallback
export const versionLabel = analysis => analysis?.model_version || (predicted(analysis) ? 'Tidak diketahui (riwayat lama)' : 'Belum diprediksi')
export const statusLabel = status => ({ uploaded: 'Baru', deduplicated: 'Deduplikasi selesai', preprocessed: 'Preprocessing selesai', predicted: 'Diprediksi', evaluated: 'Evaluasi selesai', aspects_completed: 'Aspek selesai', completed: 'Selesai' }[status] || status)
