import { useEffect, useMemo, useState } from 'react'
import {
  BarChart3,
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  CloudRain,
  Droplets,
  FlaskConical,
  Leaf,
  Loader2,
  Search,
  ShieldCheck,
  Sprout,
  Thermometer,
  Upload,
  X,
  Zap,
} from 'lucide-react'

const DEFAULTS = {
  N: 90,
  P: 42,
  K: 43,
  temperature: 20.8,
  humidity: 82,
  ph: 6.5,
  rainfall: 202,
}

const FIELD_CONFIG = [
  { key: 'N', label: 'Nitrogen', symbol: 'N', unit: 'mg/kg', icon: FlaskConical },
  { key: 'P', label: 'Phosphorus', symbol: 'P', unit: 'mg/kg', icon: Sprout },
  { key: 'K', label: 'Potassium', symbol: 'K', unit: 'mg/kg', icon: Leaf },
  { key: 'temperature', label: 'Temperature', symbol: 'T', unit: '°C', icon: Thermometer },
  { key: 'humidity', label: 'Humidity', symbol: 'H', unit: '%', icon: Droplets },
  { key: 'ph', label: 'Soil pH', symbol: 'pH', unit: '0–14', icon: FlaskConical },
  { key: 'rainfall', label: 'Rainfall', symbol: 'R', unit: 'mm', icon: CloudRain },
]

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`
}

function titleCase(value) {
  return String(value || '')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (m) => m.toUpperCase())
}

function formatNumber(value, decimals = 4) {
  const number = Number(value)

  if (!Number.isFinite(number)) {
    return '—'
  }

  return Number(
    number.toFixed(decimals)
  ).toString()
}

function App() {
  const [values, setValues] = useState(DEFAULTS)
  const [crops, setCrops] = useState([])
  const [cropCatalog, setCropCatalog] = useState([])
  const [catalogCategory, setCatalogCategory] = useState('All')
  const [catalogSearch, setCatalogSearch] = useState('')
  const [targetCrop, setTargetCrop] = useState('')
  const [metrics, setMetrics] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [soilImage, setSoilImage] = useState(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    Promise.all([
      fetch('/api/crops').then((r) => r.json()),
      fetch('/api/metrics').then((r) => r.json()),
      fetch('/api/crop-catalog').then((r) => r.json()),
    ])
      .then(([cropData, metricData, catalogData]) => {
        setCrops(cropData.crops || [])
        setMetrics(metricData)
        setCropCatalog(catalogData.crops || [])
      })
      .catch(() => {
        // The form remains usable; the API call later reports the real issue.
      })
  }, [])

  const inputCount = useMemo(
    () => Object.values(values).filter((v) => v !== '' && v !== null).length,
    [values]
  )

  const setField = (key, value) => {
    setValues((current) => ({ ...current, [key]: value }))
  }

  const useDemo = () => {
    setValues(DEFAULTS)
    setTargetCrop('')
    setError('')
    setResult(null)
  }

  const submit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...values, target_crop: targetCrop || null }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Analysis failed')
      setResult(data)
    } catch (err) {
      setError(err.message || 'Unable to complete analysis')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const uploadSoil = async (file) => {
    if (!file) return
    setSoilImage(file)
    setLoading(true)
    setError('')
    try {
      const form = new FormData()
      Object.entries(values).forEach(([key, value]) => form.append(key, value))
      if (targetCrop) form.append('target_crop', targetCrop)
      form.append('soil_image', file)
      const response = await fetch('/api/analyze-with-image', {
        method: 'POST',
        body: form,
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Image analysis failed')
      setResult(data)
    } catch (err) {
      setError(err.message || 'Unable to analyze the image')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Leaf size={22} /></div>
          <div>
            <div className="brand-name">AgriSense <span>AI</span></div>
            <div className="brand-subtitle">Crop & soil intelligence platform</div>
          </div>
        </div>
        <div className="system-chip">
          <span className="online-dot" /> Model services online
        </div>
      </header>

      <main>
        <section className="hero section-pad">
          <div className="hero-copy">
            <div className="eyebrow">SMART AGRICULTURE / DECISION SUPPORT</div>
            <h1>Turn soil data into a <em>better crop decision.</em></h1>
            <p>
              Combine soil nutrients and climate conditions to predict the best-fit crop,
              evaluate a crop you specifically care about, identify limiting factors,
              and get practical corrective actions.
            </p>
            <div className="hero-badges">
              <span><ShieldCheck size={16} /> Explainable results</span>
              <span><Zap size={16} /> Fast inference</span>
              <span><Leaf size={16} /> {cropCatalog.length || 182} reference crops</span>
            </div>
          </div>
          <div className="hero-visual">
            <div className="visual-orbit orbit-a" />
            <div className="visual-orbit orbit-b" />
            <div className="visual-core"><Sprout size={66} /></div>
            <div className="float-card fc-a"><span>99.09%</span><small>test accuracy</small></div>
            <div className="float-card fc-b"><span>99.50%</span><small>5-fold CV</small></div>
          </div>
        </section>

        <section className="section-pad workspace">
          <aside className="input-panel panel">
            <div className="panel-heading">
              <div>
                <div className="panel-kicker">INPUTS</div>
                <h2>Farm conditions</h2>
              </div>
              <button className="ghost-btn" type="button" onClick={useDemo}>Demo values</button>
            </div>

            <div className="progress-line"><span style={{ width: `${(inputCount / 7) * 100}%` }} /></div>

            <form onSubmit={submit}>
              <div className="field-grid">
                {FIELD_CONFIG.map(({ key, label, unit, icon: Icon }) => (
                  <label className="field" key={key}>
                    <span className="field-label"><span>{label}</span><small>{unit}</small></span>
                    <div className="input-wrap">
                      <Icon size={16} />
                      <input
                        type="number"
                        step="0.01"
                        required
                        value={values[key]}
                        onChange={(e) => setField(key, e.target.value)}
                      />
                    </div>
                  </label>
                ))}
              </div>

              <label className="select-field">
                <span>Specific crop to evaluate <small>optional</small></span>
                <select value={targetCrop} onChange={(e) => setTargetCrop(e.target.value)}>
                  <option value="">Let AI choose the best crop</option>
                  {cropCatalog.map((item, index) => (
                    <option key={`${item.crop}-${index}`} value={item.crop}>
                      {item.crop}
                    </option>
                  ))}
                </select>
              </label>

              <button className="primary-btn" type="submit" disabled={loading}>
                {loading ? <><Loader2 size={18} className="spin" /> Analyzing...</> : <>Analyze conditions <ArrowRight size={18} /></>}
              </button>
            </form>

            <div className="panel-note">
              <ShieldCheck size={15} />
              Use laboratory soil-test values where available. AI output is decision support, not a substitute for agronomic validation.
            </div>
          </aside>

          <section className="results-panel">
            {!result && !loading && !error && (
              <div className="empty-result panel">
                <div className="empty-illustration"><BarChart3 size={38} /></div>
                <div className="panel-kicker">RESULTS</div>
                <h2>Your agricultural assessment will appear here.</h2>
                <p>Run an analysis to see the recommended crop, confidence, supporting crops, limiting factors, remedies, and explanation.</p>
              </div>
            )}

            {loading && (
              <div className="empty-result panel">
                <div className="loading-circle"><Loader2 size={34} className="spin" /></div>
                <div className="panel-kicker">RUNNING AI</div>
                <h2>Analyzing soil and climate signals...</h2>
                <p>Combining crop probabilities, feature profiles, suitability checks, and explanation.</p>
              </div>
            )}

            {error && (
              <div className="error-banner"><AlertTriangle size={20} /><div><strong>Analysis failed</strong><p>{error}</p></div><button onClick={() => setError('')}><X size={18} /></button></div>
            )}

            {result && !loading && <AnalysisResult result={result} />}
          </section>
        </section>

        <section className="section-pad lower-grid">
          <SoilUpload
            file={soilImage}
            dragging={dragging}
            setDragging={setDragging}
            onFile={uploadSoil}
            available={result?.soil_model_available}
            analysis={result?.soil_image_analysis}
          />

          <ModelSummary metrics={metrics} />
        </section>

        <section className="section-pad catalog-section">
          <CropCatalog
            crops={cropCatalog}
            supportedCrops={crops}
            category={catalogCategory}
            setCategory={setCatalogCategory}
            search={catalogSearch}
            setSearch={setCatalogSearch}
          />
        </section>
      </main>

      <footer className="footer section-pad">
        <div><Leaf size={18} /> AgriSense AI</div>
        <span>Built around a Flask API + React decision-support frontend.</span>
      </footer>
    </div>
  )
}

function CropCatalog({ crops, supportedCrops, category, setCategory, search, setSearch }) {
  const categories = ['All', ...Array.from(new Set(crops.map((item) => item.major_category))).sort()]
  const supported = new Set(supportedCrops.map((c) => String(c).toLowerCase()))
  const filtered = crops.filter((item) => {
    const matchesCategory = category === 'All' || item.major_category === category
    const q = search.trim().toLowerCase()
    const matchesSearch = !q || [item.crop, item.crop_group, item.major_category].some((v) => String(v).toLowerCase().includes(q))
    return matchesCategory && matchesSearch
  })

  return (
    <section className="panel crop-catalog-panel">
      <div className="card-title">
        <div><div className="panel-kicker">INDIAN CROP REFERENCE</div><h3>Crop types from the attached reference</h3></div>
        <span>{crops.length} entries</span>
      </div>
      <p className="card-copy">The reference catalog is added alongside the current ML model. All reference crops are enabled. The original trained ML classes use learned model evidence; other reference crops use transparent profile-based suitability scoring until crop-specific numerical training data is added.</p>
      <div className="catalog-controls">
        <input className="catalog-search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search crops, groups or categories..." />
        <select value={category} onChange={(e) => setCategory(e.target.value)}>
          {categories.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
      </div>
      <div className="catalog-grid">
        {filtered.map((item, index) => {
          const normalized = String(item.crop).toLowerCase()
          const ai = supported.has(normalized) || (normalized.includes('coffee') && supported.has('coffee'))
          return (
            <article className="catalog-item" key={`${item.crop}-${index}`}>
              <div className="catalog-item-top"><span className="catalog-crop">{item.crop}</span>{ai && <span className="catalog-badge">AI supported</span>}</div>
              <div className="catalog-meta">{item.major_category} · {item.crop_group}</div>
              <div className="catalog-system">{item.system_notes}</div>
              <div className="catalog-varieties">
                {typeof item.varieties === 'string'
                  ? item.varieties
                      .split(';')
                      .map((v) => v.trim())
                      .filter(Boolean)
                      .slice(0, 5)
                      .join(' · ')
                  : Array.isArray(item.varieties)
                    ? item.varieties
                        .slice(0, 5)
                        .join(' · ')
                    : ''}
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}

function AnalysisResult({ result }) {
  const suitability = result.agriculture_suitability
  const selected = result.selected_crop
  const explanations = (result.explanation || []).filter((x) => x.feature !== 'unavailable')

  return (
    <div className="results-stack">
      <div className="result-hero panel">
        <div>
          <div className="panel-kicker">AI RECOMMENDATION</div>
          <div className="recommendation-row">
            <div className="crop-emblem"><Sprout size={28} /></div>
            <div>
              <div className="recommendation-name">{titleCase(result.prediction.crop)}</div>
              <div className="recommendation-sub">Best-fit crop for the supplied conditions</div>
            </div>
          </div>
        </div>
        <div className="confidence-ring"><strong>{percent(result.prediction.probability)}</strong><span>confidence</span></div>
      </div>

      <div className={`suitability-card ${suitability.suitable ? 'good' : 'bad'}`}>
        {suitability.suitable ? <CheckCircle2 size={22} /> : <AlertTriangle size={22} />}
        <div><span className="panel-kicker">AGRICULTURAL SUITABILITY</span><strong>{suitability.status}</strong><p>{suitability.message}</p></div>
      </div>

      <div className="two-col">
        <section className="panel result-card">
          <div className="card-title"><div><div className="panel-kicker">RANKING</div><h3>Top supported crops</h3></div><span>{result.top_crops.length} candidates</span></div>
          <div className="crop-bars">
            {result.top_crops.map((item) => (
              <div className="crop-bar-row" key={item.crop}>
                <div><span>{titleCase(item.crop)}</span><b>{percent(item.probability)}</b></div>
                <div className="bar-track"><span style={{ width: `${Math.max(2, item.probability * 100)}%` }} /></div>
              </div>
            ))}
          </div>
        </section>

        <section className="panel result-card">
          <div className="card-title"><div><div className="panel-kicker">TARGET CHECK</div><h3>{titleCase(selected.crop)}</h3></div><span className={selected.suitable ? 'mini-good' : 'mini-bad'}>{selected.status}</span></div>
          <p className="card-copy">{selected.message}</p>
          <div className="detail-grid">
            {selected.details.slice(0, 7).map((item) => (
              <div className="detail-item" key={item.feature}>
                <div><span>{item.name}</span><b className={item.status}>{item.status}</b></div>
                <small>
                  Current: {formatNumber(item.value)}
                  {' · Target: '}
                  {formatNumber(item.crop_q1)} 
                  {'–'} 
                  {formatNumber(item.crop_q3)}
                </small>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="two-col">
        <section className="panel result-card">
          <div className="card-title"><div><div className="panel-kicker">LIMITING FACTORS</div><h3>What needs attention</h3></div></div>
          {result.reasons?.length ? (
            <div className="reason-list">{result.reasons.map((reason) => <div className="reason" key={reason.feature}><AlertTriangle size={16} /><div><strong>{reason.name}</strong><p>{reason.reason}</p></div></div>)}</div>
          ) : <div className="all-good"><CheckCircle2 size={18} /> No major parameter mismatch detected.</div>}
        </section>

        <section className="panel result-card">
          <div className="card-title"><div><div className="panel-kicker">REMEDIES</div><h3>Recommended actions</h3></div></div>
          {result.remedies?.length ? (
            <div className="remedy-list">{result.remedies.map((remedy, index) => <div className="remedy" key={`${remedy.feature}-${index}`}><span>{String(index + 1).padStart(2, '0')}</span><div><strong>{titleCase(remedy.feature)} · {remedy.type}</strong><p>{remedy.remedy}</p></div></div>)}</div>
          ) : <div className="all-good"><CheckCircle2 size={18} /> No corrective action needed from the current diagnostic layer.</div>}
        </section>
      </div>

      <section className="panel result-card xai-card">
        <div className="card-title"><div><div className="panel-kicker">EXPLAINABLE AI</div><h3>Feature contribution signals</h3></div><span>SHAP-based</span></div>
        {explanations.length ? (
          <div className="xai-grid">{explanations.slice(0, 7).map((item) => { const max = Math.max(...explanations.map((x) => Number(x.importance || 0)), 1e-9); return <div className="xai-row" key={item.feature}><div><span>{titleCase(item.feature)}</span><b>{Number(item.impact || 0).toFixed(4)}</b></div><div className="bar-track"><span style={{ width: `${Math.min(100, Math.max(2, (Number(item.importance || 0) / max) * 100))}%` }} /></div></div> })}</div>
        ) : <div className="xai-unavailable">Explanation data was unavailable for this run.</div>}
      </section>
    </div>
  )
}

function SoilUpload({ file, dragging, setDragging, onFile, available, analysis }) {
  return (
    <section className="panel soil-panel">
      <div className="card-title"><div><div className="panel-kicker">SOIL IMAGE MODE</div><h3>Visual soil classification</h3></div><span>{available ? 'model ready' : 'model pending'}</span></div>
      <p className="card-copy">Upload a field image to run the MobileNetV2 soil classifier when its trained model is available.</p>
      <label
        className={`dropzone ${dragging ? 'dragging' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); onFile(e.dataTransfer.files?.[0]) }}
      >
        <input type="file" accept="image/*" hidden onChange={(e) => onFile(e.target.files?.[0])} />
        <div className="upload-icon"><Upload size={24} /></div>
        <strong>{file ? file.name : 'Drop a soil image here'}</strong>
        <span>PNG, JPG, JPEG or WEBP · max 8 MB</span>
        <button type="button">Choose image</button>
      </label>
      {analysis?.available === false && <div className="soil-status"><AlertTriangle size={17} /> {analysis.message}</div>}
      {analysis?.soil_type && <div className="soil-result"><div className="soil-type">{analysis.soil_type}</div><div><span>Confidence</span><strong>{percent(analysis.probability)}</strong></div></div>}
    </section>
  )
}

function ModelSummary({ metrics }) {
  const test = metrics?.test_accuracy ? percent(metrics.test_accuracy) : '—'
  const cv = metrics?.cv_accuracy_mean ? percent(metrics.cv_accuracy_mean) : '—'
  return (
    <section className="panel model-panel">
      <div className="card-title"><div><div className="panel-kicker">MODEL HEALTH</div><h3>Current crop model</h3></div><Activity size={19} /></div>
      <div className="stats-block">
        <div><strong>{metrics?.samples?.toLocaleString() || '—'}</strong><span>training records</span></div>
        <div><strong>{metrics?.n_classes || '—'}</strong><span>crop classes</span></div>
        <div><strong>{test}</strong><span>test accuracy</span></div>
        <div><strong>{cv}</strong><span>5-fold CV</span></div>
      </div>
      <div className="model-foot"><span><CheckCircle2 size={15} /> Ensemble active</span><span>RF + LR + GB</span></div>
    </section>
  )
}

export default App
