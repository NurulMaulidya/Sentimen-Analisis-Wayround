import React from 'react'

export default function InferenceModelSelect({ value, onChange, disabled }) {
  return (
    <label className="block text-sm font-medium">
      Model analisis
      <select className="block mt-1 border rounded p-2" value={value} onChange={e => onChange(e.target.value)} disabled={disabled}>
        <option value="v4">v4 · model penelitian</option>
        <option value="karakter">Kandidat · TF-IDF kata dan karakter</option>
      </select>
      {value === 'karakter' && <span className="block mt-1 text-xs text-amber-800">
        Eksperimental. Aspek memakai pemisahan klausa. Evaluasi independen menunggu anotasi.
      </span>}
    </label>
  )
}
