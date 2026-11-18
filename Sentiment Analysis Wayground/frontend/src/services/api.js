import axios from 'axios'

const API_BASE = '/api/v1'

const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Helper to unwrap axios response data
const unwrap = (response) => response.data

// API service functions for the redesigned website
export const apiService = {
  // ==========================================
  // HEALTH & STATS
  // ==========================================

  healthCheck: () => apiClient.get('/health').then(unwrap),
  getStats: () => apiClient.get('/stats').then(unwrap),
  getModelStats: () => apiClient.get('/model/stats').then(unwrap),
  getStages: () => apiClient.get('/stages').then(unwrap),

  // ==========================================
  // ANALYSIS MANAGEMENT
  // ==========================================

  // Get all analyses
  getAnalyses: () => apiClient.get('/analysis').then(unwrap),

  // Get single analysis
  getAnalysis: (id) => apiClient.get(`/analysis/${id}`).then(unwrap),

  // Delete analysis
  deleteAnalysis: (id) => apiClient.delete(`/analysis/${id}`).then(unwrap),

  // Upload new analysis
  inspectUpload: (file) => {
    const form = new FormData(); form.append('file', file)
    return apiClient.post('/analysis/inspect', form, { headers: { 'Content-Type': 'multipart/form-data' } }).then(unwrap)
  },
  uploadAnalysis: (file, name = null, contentColumn = '', labelColumn = '', mode = 'new_data') => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('mode', mode)
    if (name) formData.append('name', name)
    if (contentColumn) formData.append('content_column', contentColumn)
    if (labelColumn) formData.append('label_column', labelColumn)
    return apiClient.post('/analysis/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    }).then(unwrap)
  },

  // ==========================================
  // PIPELINE OPERATIONS
  // ==========================================

  // Deduplicate
  deduplicate: (analysisId) =>
    apiClient.post(`/analysis/${analysisId}/deduplicate`).then(unwrap),

  // Preprocess
  preprocess: (analysisId) =>
    apiClient.post(`/analysis/${analysisId}/preprocess`).then(unwrap),

  // Detect aspects
  detectAspects: (analysisId) =>
    apiClient.post(`/analysis/${analysisId}/aspects`).then(unwrap),

  // Predict
  predict: (analysisId, model) =>
    apiClient.post(`/analysis/${analysisId}/predict`, { model }).then(unwrap),

  // Evaluate
  evaluate: (analysisId) =>
    apiClient.post(`/analysis/${analysisId}/evaluate`).then(unwrap),

  // ==========================================
  // DATA RETRIEVAL
  // ==========================================

  // Original dataset
  getOriginalData: (analysisId, page = 1, limit = 50, search = '', status = '') => {
    const params = { page, limit }
    if (search) params.search = search
    if (status) params.status = status
    return apiClient.get(`/analysis/${analysisId}/original`, { params }).then(unwrap)
  },

  // Deduplicated dataset
  getDeduplicatedData: (analysisId, page = 1, limit = 50, showDuplicates = false) => {
    const params = { page, limit, show_duplicates: showDuplicates }
    return apiClient.get(`/analysis/${analysisId}/deduplicated`, { params }).then(unwrap)
  },

  // Preprocessing stage data
  getPreprocessingStage: (analysisId, stage, page = 1, limit = 50, search = '') => {
    const params = { page, limit }
    if (search) params.search = search
    return apiClient.get(`/analysis/${analysisId}/preprocessing/${stage}`, { params }).then(unwrap)
  },

  // Compare preprocessing for single review
  getReviewComparison: (analysisId, reviewId) =>
    apiClient.get(`/analysis/${analysisId}/compare/${reviewId}`).then(unwrap),

  // Aspects data
  getAspectsData: (analysisId, page = 1, limit = 50, aspect = null) => {
    const params = { page, limit }
    if (aspect) params.aspect = aspect
    return apiClient.get(`/analysis/${analysisId}/aspects`, { params }).then(unwrap)
  },

  // Predictions data
  getPredictions: (analysisId, page = 1, limit = 50, sentiment = null, search = '') => {
    const params = { page, limit }
    if (sentiment) params.sentiment = sentiment
    if (search) params.search = search
    return apiClient.get(`/analysis/${analysisId}/predictions`, { params }).then(unwrap)
  },

  // Evaluation data
  getEvaluation: (analysisId) =>
    apiClient.get(`/analysis/${analysisId}/evaluation`).then(unwrap),

  // ==========================================
  // SINGLE REVIEW PREDICTION
  // ==========================================

  predictSingle: (text, model) =>
    apiClient.post('/predict-single', { text, model }).then(unwrap),
  getTfidf: (id, page = 1) => apiClient.get(`/analysis/${id}/tfidf`, { params: { page, limit: 20 } }).then(unwrap),
  transformTfidf: (id, model) => apiClient.post(`/analysis/${id}/tfidf`, { model }).then(unwrap),
  getFeatures: (id, review) => apiClient.get(`/analysis/${id}/features/${review}`).then(unwrap),
  getAspectReviews: (id, page = 1, aspect = '', sentiment = '', search = '') => apiClient.get(`/analysis/${id}/aspect-reviews`, { params: { page, limit: 20, aspect, sentiment, search } }).then(unwrap),
  getSingleHistory: (page = 1, search = '', order = 'newest') => apiClient.get('/single-history', { params: { page, limit: 20, search, order } }).then(unwrap),
  getSingle: id => apiClient.get(`/single-history/${id}`).then(unwrap),
  deleteSingle: id => apiClient.delete(`/single-history/${id}`).then(unwrap),
  getResearch: () => apiClient.get('/research').then(unwrap),
  getResearchExamples: () => apiClient.get('/research/examples').then(unwrap),
  getRuns: id => apiClient.get(`/analysis/${id}/runs`).then(unwrap),
  getRun: (id, run) => apiClient.get(`/analysis/${id}/runs/${run}`).then(unwrap),

  // ==========================================
  // PIPELINE STAGES INFO
  // ==========================================

  getPreprocessingStages: () => apiClient.get('/stages').then(unwrap),
}

export default apiService
