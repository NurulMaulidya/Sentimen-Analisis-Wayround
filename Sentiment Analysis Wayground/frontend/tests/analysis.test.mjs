import test from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { buildSync } from 'esbuild'
import { resultTab, predicted, percent, versionLabel, showValue, stageValue } from '../src/utils/analysisUi.js'

const require = createRequire(import.meta.url)
const React = require('react')
const { renderToStaticMarkup } = require('react-dom/server')
const compiled = buildSync({ entryPoints: [fileURLToPath(new URL('../src/components/AnalysisViews.jsx', import.meta.url))], bundle: true, write: false, platform: 'node', format: 'cjs', external: ['react'] })
const module = { exports: {} }
new Function('require', 'module', 'exports', compiled.outputFiles[0].text)(require, module, module.exports)
const views = module.exports

test('URLs select their own result tabs', () => {
  assert.equal(resultTab('/analysis/1/results/aspects'), 'aspects')
  assert.equal(resultTab('/analysis/2/results/evaluation'), 'evaluation')
  assert.equal(resultTab('/analysis/3/results'), 'predictions')
})
test('intermediate completed stages retain available predictions', () => {
  for (const status of ['predicted', 'aspects_completed', 'evaluated', 'completed']) assert.equal(predicted({ status }), true)
  assert.equal(predicted({ status: 'uploaded', predictions_ready: false }), false)
})
test('legacy versions stay unknown instead of adopting current deployment', () => {
  assert.match(versionLabel({ status: 'completed' }), /Tidak diketahui/)
  assert.equal(versionLabel({ model_version: 'v1' }), 'v1')
})
test('missing values are distinct from real zero metrics', () => {
  assert.equal(percent(0), '0.00%')
  assert.equal(percent(null), '—')
  assert.equal(percent(NaN), '—')
  assert.equal(showValue(['tidak_bagus', 'error']), 'tidak_bagus · error')
})
test('evaluation renders actual matrix counts and axis labels', () => {
  const html = renderToStaticMarkup(React.createElement(views.EvaluationView, { evaluation: {
    metrics: { accuracy: 0.5, f1_macro: 0 }, correct_count: 2, incorrect_count: 2,
    confusion_matrix_labels: ['negatif', 'netral', 'positif'], confusion_matrix: [[1, 1, 0], [0, 0, 1], [0, 0, 1]],
    classification_report: { negatif: { precision: 1, recall: 0.5, 'f1-score': 2 / 3, support: 2 } }
  } }))
  assert.match(html, /baris aktual, kolom prediksi/)
  assert.match(html, /50.00%/)
  assert.match(html, /0.00%/)
  assert.match(html, /Support/)
  assert.doesNotMatch(html, /NaN|undefined/)
})
test('aspect view distinguishes unique reviews from multiple mentions', () => {
  const html = renderToStaticMarkup(React.createElement(views.AspectView, { data: {
    summary: { total_reviews_with_aspects: 2, total_aspect_mentions: 3 }, data: [], distribution: {}
  } }))
  assert.match(html, /2 ulasan unik · 3 penyebutan aspek/)
  assert.match(html, /dihitung dari kalimat aspek itu sendiri/)
  assert.doesNotMatch(html, /mengikuti sentimen ulasan utuh/)
})

test('research evaluation describes its source rather than uploaded labels', () => {
  const html = renderToStaticMarkup(React.createElement(views.EvaluationView, { scope: 'research', evaluation: {
    metrics: {}, correct_count: 0, incorrect_count: 0, confusion_matrix: []
  } }))
  assert.match(html, /Evaluasi penelitian tersimpan/)
  assert.doesNotMatch(html, /dibandingkan dengan label pada file ini/)
})

test('aspect per-review table retains unmatched reviews and global counts', () => {
  const html = renderToStaticMarkup(React.createElement(views.AspectView, { data: {
    summary: { total_reviews_with_aspects: 1, total_aspect_mentions: 2 },
    flow: { predicted: 2, with_aspects: 1, without_aspects: 1 },
    distribution: { 'Gamifikasi': { negatif: 0, netral: 0, positif: 1, total: 1 } },
    review_rows: [{ id: 4, original_content: 'teks tanpa aspek', predicted_label: 'positif', detected_aspects: [], aspect_details: [] }]
  } }))
  assert.match(html, /Hasil aspek setiap ulasan/)
  assert.match(html, /Tidak terdeteksi/)
  assert.match(html, /Sentimen per aspek/)
  assert.match(html, /n = 1/)
  assert.doesNotMatch(html, /NaN|undefined/)
})


test('preprocessing distinguishes pending steps from genuinely empty outputs', () => {
  assert.equal(stageValue(null), 'Belum diproses')
  assert.equal(stageValue([]), '[] (kosong)')
  assert.equal(stageValue(''), '(kosong)')
})
