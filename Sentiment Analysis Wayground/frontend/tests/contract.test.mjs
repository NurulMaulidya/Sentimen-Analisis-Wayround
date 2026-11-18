import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative, extname } from 'node:path'
import { fileURLToPath } from 'node:url'

const SRC = fileURLToPath(new URL('../src', import.meta.url))

const walk = (dir) => readdirSync(dir).flatMap((entry) => {
  const full = join(dir, entry)
  return statSync(full).isDirectory() ? walk(full) : [full]
})

const sourceFiles = walk(SRC).filter((file) => ['.js', '.jsx'].includes(extname(file)))
const read = (file) => readFileSync(file, 'utf8')

test('every API method called by the UI exists in the API service', () => {
  const apiSource = read(join(SRC, 'services', 'api.js'))
  const exported = new Set(
    // Properti objek layanan: `nama: (...) =>` maupun `nama: arg =>`.
    [...apiSource.matchAll(/^\s{2}([A-Za-z_$][\w$]*):\s*(?:\(|async\b|[A-Za-z_$][\w$]*\s*=>)/gm)].map((match) => match[1])
  )
  assert.ok(exported.size > 20, 'API service harus mengekspor banyak metode')

  const missing = []
  for (const file of sourceFiles) {
    if (file.endsWith(join('services', 'api.js'))) continue
    const source = read(file)
    // Panggilan pada objek api: `api.foo(` / `apiService.foo(`.
    for (const match of source.matchAll(/\bapi(?:Service)?\.([A-Za-z_$][\w$]*)\s*\(/g)) {
      if (!exported.has(match[1])) {
        missing.push(`${relative(SRC, file)} -> ${match[1]}`)
      }
    }
  }
  assert.deepEqual(missing, [], `Metode API tidak ada di services/api.js: ${missing.join(', ')}`)
})

test('the UI does not reference a removed context or a training workflow', () => {
  const offenders = []
  for (const file of sourceFiles) {
    const source = read(file)
    if (/AnalysisContext|AnalysisProvider|useAnalysis/.test(source)) {
      offenders.push(`${relative(SRC, file)}: AnalysisContext`)
    }
    // Website tidak boleh punya jalur pelatihan model.
    if (/\bapi(?:Service)?\.(train|trainModel|fit)\s*\(/.test(source)) {
      offenders.push(`${relative(SRC, file)}: pemanggilan training`)
    }
  }
  assert.deepEqual(offenders, [], `Teks terlarang di UI: ${offenders.join(', ')}`)
})

test('no source file carries mojibake from a wrong encoding', () => {
  const offenders = []
  for (const file of sourceFiles) {
    const source = read(file)
    if (/â€|Ã©|Ã¡|Â\w|�/.test(source)) {
      offenders.push(relative(SRC, file))
    }
  }
  assert.deepEqual(offenders, [], `Teks salah encoding: ${offenders.join(', ')}`)
})

test('aspect sentiment is described as coming from the aspect sentences', () => {
  const offenders = []
  for (const file of sourceFiles) {
    const source = read(file)
    if (/mengikuti sentimen ulasan utuh|sentimen aspek mengikuti sentimen ulasan/i.test(source)) {
      offenders.push(relative(SRC, file))
    }
  }
  assert.deepEqual(offenders, [], `Keterangan aspek bertentangan: ${offenders.join(', ')}`)
})
