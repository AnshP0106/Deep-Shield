
import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  AudioLines,
  Bot,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  CircleStop,
  Eye,
  FileAudio,
  FileImage,
  FileVideo,
  Fingerprint,
  Image as ImageIcon,
  LoaderCircle,
  ScanFace,
  Shield,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X,
} from 'lucide-react'
import './App.css'

const API_URL = (
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

// ---------------------------------------------------------
// HELPERS
// ---------------------------------------------------------

function isImageFile(file) {
  return Boolean(file?.type?.startsWith('image/'))
}

function getMediaIcon(file) {
  const type = file?.type || ''

  if (type.startsWith('image/')) return FileImage
  if (type.startsWith('audio/')) return FileAudio
  if (type.startsWith('video/')) return FileVideo

  return UploadCloud
}

function formatConfidence(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null
  }

  return `${(value <= 1 ? value * 100 : value).toFixed(1)}%`
}

function evidenceUrl(path) {
  if (typeof path !== 'string' || !path.trim()) return null

  try {
    const url = new URL(path, `${API_URL}/`)

    // Evidence must come from the configured API origin.
    const apiOrigin = new URL(API_URL).origin

    if (url.origin !== apiOrigin) return null

    return url.toString()
  } catch {
    return null
  }
}

// ---------------------------------------------------------
// D-ID AI ANALYST
// ---------------------------------------------------------

function DidAgent({ analysis, details = {}, explanation }) {
  const containerRef = useRef(null)

  const [speechStatus, setSpeechStatus] = useState('')
  const [speakingRequest, setSpeakingRequest] = useState(false)

  const clientKey = import.meta.env.VITE_DID_CLIENT_KEY
  const agentId = import.meta.env.VITE_DID_AGENT_ID

  useEffect(() => {
    const container = containerRef.current

    if (!container || !clientKey || !agentId) return undefined

    container.innerHTML = ''

    const script = document.createElement('script')
    script.type = 'module'
    script.src = 'https://agent.d-id.com/v2/index.js'
    script.dataset.mode = 'full'
    script.dataset.targetId = 'did-agent-container'
    script.dataset.clientKey = clientKey
    script.dataset.agentId = agentId
    script.dataset.name = 'did-agent'

    container.appendChild(script)

    return () => {
      container.innerHTML = ''
    }
  }, [clientKey, agentId])

  function buildSpeech() {
    if (!analysis) {
      return (
        'Hello! I am your DeepShield AI analyst. ' +
        'Upload an image, audio recording, or video and run an analysis. ' +
        'I can then explain the returned results and their limitations.'
      )
    }

    const mediaType = String(
      analysis.media_type || 'media'
    ).toLowerCase()

    // Video uses a different response schema.
    const prediction =
      analysis.result ??
      analysis.overall_prediction ??
      'Unknown'

    const confidenceValue =
      analysis.confidence ??
      analysis.overall_fake_probability

    const confidence =
      typeof confidenceValue === 'number'
        ? formatConfidence(confidenceValue)
        : null

    const riskLevel =
      details.risk_level ??
      analysis.risk_level

    const parts = [
      `The ${mediaType} analysis is complete.`,
      `The detector returned the classification: ${prediction}.`,
    ]

    if (confidence) {
      parts.push(`The reported confidence or score is ${confidence}.`)
    }

    if (riskLevel != null) {
      parts.push(`The reported risk level is ${riskLevel}.`)
    }

    if (typeof details.face_count === 'number') {
      parts.push(
        `${details.face_count} faces were detected in the image.`
      )
    }

    if (typeof details.fake_face_count === 'number') {
      parts.push(
        `${details.fake_face_count} faces were classified as fake.`
      )
    }

    if (explanation?.summary) {
      parts.push(String(explanation.summary))
    }

    if (explanation?.why) {
      parts.push(String(explanation.why))
    }

    if (explanation?.simple_explanation) {
      parts.push(String(explanation.simple_explanation))
    }

    if (explanation?.confidence_explanation) {
      parts.push(String(explanation.confidence_explanation))
    }

    if (mediaType === 'image') {
      parts.push(
        'Grad-CAM highlights image regions that influenced the model output. ' +
        'It is not proof that a region was manipulated or that the image is fake.'
      )
    }

    if (mediaType === 'video') {
      parts.push(
        'Video analysis is experimental. ' +
        'The result should be independently verified.'
      )
    }

    parts.push(
      'This is an AI-assisted assessment, not definitive proof of authenticity. ' +
      'Verify important findings independently.'
    )

    // Keep speech reasonably short.
    return parts.join(' ').slice(0, 1800)
  }

  async function explainResult() {
    if (speakingRequest) return

    setSpeechStatus('')

    if (!clientKey || !agentId) {
      setSpeechStatus(
        'Configure the D-ID client key and agent ID in .env.local.'
      )
      return
    }

    const speak = window.DID_AGENTS_API?.functions?.speak

    if (typeof speak !== 'function') {
      setSpeechStatus(
        'D-ID is not ready yet. Wait for the avatar to load and try again.'
      )
      return
    }

    setSpeakingRequest(true)
    setSpeechStatus('Sending the explanation to the avatar…')

    try {
      await speak({
        type: 'text',
        input: buildSpeech(),
      })

      setSpeechStatus('Explanation sent to the avatar.')
    } catch (error) {
      console.error('D-ID speech error:', error)

      setSpeechStatus(
        'The avatar could not speak. Check the browser console and D-ID configuration.'
      )
    } finally {
      setSpeakingRequest(false)
    }
  }

  if (!clientKey || !agentId) {
    return (
      <div className="agent-placeholder">
        <div className="agent-orb">
          <Bot size={34} />
        </div>

        <span className="eyebrow">AVATAR CONFIGURATION</span>
        <h3>Your AI analyst awaits</h3>

        <p>
          Configure your D-ID client key and agent ID
          in the frontend environment file to activate the avatar.
        </p>

        <span className="agent-hint">D-ID · AGENT EMBED</span>
      </div>
    )
  }

  return (
    <div className="did-agent-wrapper">
      <div
        id="did-agent-container"
        ref={containerRef}
        className="did-agent"
      />

      <button
        type="button"
        className="primary-button"
        onClick={explainResult}
        disabled={speakingRequest}
      >
        {speakingRequest ? (
          <LoaderCircle className="spin" size={18} />
        ) : (
          <Bot size={17} />
        )}

        {analysis ? 'EXPLAIN THIS RESULT' : 'INTRODUCE YOURSELF'}
      </button>

      {speechStatus && (
        <p className="micro-copy" role="status" aria-live="polite">
          {speechStatus}
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------
// IMAGE AND GRAD-CAM EVIDENCE
// ---------------------------------------------------------

function EvidenceViewer({ file, previewUrl, faces }) {
  if (!isImageFile(file)) return null

  const evidenceFaces = Array.isArray(faces) ? faces : []

  return (
    <motion.section
      className="evidence-section"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      <div className="evidence-heading">
        <div className="heading-icon cyan">
          <Eye size={19} />
        </div>

        <div>
          <span className="panel-kicker">VISUAL EVIDENCE</span>
          <h3>Image &amp; Grad-CAM</h3>
        </div>
      </div>

      <p className="evidence-disclaimer">
        Grad-CAM shows regions that influenced the model's output.
        Highlighted regions are not proof of manipulation.
      </p>

      <div className="evidence-grid">
        <article className="evidence-card">
          <div className="evidence-card-heading">
            <ImageIcon size={16} />
            <span>ORIGINAL IMAGE</span>
          </div>

          {previewUrl ? (
            <img
              className="evidence-image"
              src={previewUrl}
              alt="Original uploaded image"
            />
          ) : (
            <div className="evidence-unavailable">
              <ImageIcon size={22} />
              <span>Original preview is unavailable.</span>
            </div>
          )}

          <p className="evidence-caption">
            Original file selected for analysis.
          </p>
        </article>

        {evidenceFaces.map((face, index) => {
          const faceId = face?.face_id ?? index + 1
          const heatmapUrl = evidenceUrl(face?.gradcam_url)
          const faceConfidence = formatConfidence(face?.confidence)

          return (
            <article className="evidence-card" key={faceId}>
              <div className="evidence-card-heading">
                <Activity size={16} />
                <span>FACE {faceId} · GRAD-CAM</span>
              </div>

              {heatmapUrl ? (
                <>
                  <a
                    className="evidence-image-link"
                    href={heatmapUrl}
                    target="_blank"
                    rel="noreferrer"
                    title="Open the generated Grad-CAM image"
                  >
                    <img
                      className="evidence-image"
                      src={heatmapUrl}
                      alt={`Grad-CAM overlay for face ${faceId}`}
                      loading="eager"
                      onError={() => {
                        console.error(
                          'Grad-CAM failed to load:',
                          heatmapUrl
                        )
                      }}
                    />
                  </a>

                  <div className="evidence-metadata">
                    <div>
                      <span>Prediction</span>
                      <strong>{face?.prediction ?? 'Unknown'}</strong>
                    </div>

                    <div>
                      <span>Face confidence</span>
                      <strong>{faceConfidence ?? '—'}</strong>
                    </div>
                  </div>

                  <p className="evidence-caption">
                    Face-crop overlay generated by the detector.
                  </p>
                </>
              ) : (
                <div className="evidence-unavailable">
                  <CircleAlert size={22} />
                  <span>
                    No Grad-CAM URL was returned for this face.
                    {face?.gradcam_generated
                      ? ' Check the backend evidence path.'
                      : ' Evidence may not have been generated.'}
                  </span>
                </div>
              )}
            </article>
          )
        })}
      </div>

      {evidenceFaces.length === 0 && (
        <div className="evidence-unavailable">
          <ScanFace size={22} />
          <span>
            No face-level results were returned. The image may contain
            no detectable faces, or evidence generation may have failed.
          </span>
        </div>
      )}
    </motion.section>
  )
}

// ---------------------------------------------------------
// MAIN DASHBOARD
// ---------------------------------------------------------

function App() {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [previewUrl, setPreviewUrl] = useState(null)

  const inputRef = useRef(null)
  const abortControllerRef = useRef(null)
  const requestIdRef = useRef(0)

  useEffect(() => {
    if (!isImageFile(file)) {
      setPreviewUrl(null)
      return undefined
    }

    const objectUrl = URL.createObjectURL(file)
    setPreviewUrl(objectUrl)

    return () => URL.revokeObjectURL(objectUrl)
  }, [file])

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  function cancelCurrentRequest() {
    requestIdRef.current += 1

    abortControllerRef.current?.abort()
    abortControllerRef.current = null

    setLoading(false)
  }

  function chooseFile(nextFile) {
    if (!nextFile) return

    cancelCurrentRequest()

    setFile(nextFile)
    setResult(null)
    setError('')
  }

  function removeFile() {
    cancelCurrentRequest()

    setFile(null)
    setResult(null)
    setError('')

    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  async function analyzeMedia() {
    if (!file || loading) return

    const controller = new AbortController()
    const requestId = ++requestIdRef.current

    abortControllerRef.current = controller

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const body = new FormData()
      body.append('file', file)

      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body,
        signal: controller.signal,
      })

      const data = await response.json().catch(() => ({}))

      if (requestId !== requestIdRef.current) return

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : `Request failed (${response.status})`
        )
      }

      if (!data.analysis) {
        throw new Error(
          'The API response did not contain an analysis result.'
        )
      }

      setResult(data)
    } catch (err) {
      if (requestId !== requestIdRef.current) return

      if (err.name === 'AbortError') {
        setError('The browser request was cancelled.')
      } else {
        setError(
          err.message === 'Failed to fetch'
            ? 'Could not reach FastAPI. Check the backend and CORS configuration.'
            : err.message || 'An unexpected error occurred.'
        )
      }
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false)
        abortControllerRef.current = null
      }
    }
  }

  const analysis = result?.analysis
  const details = analysis?.details ?? {}
  const explanation = result?.explanation
  const confidence = formatConfidence(analysis?.confidence)
  const MediaIcon = getMediaIcon(file)

  const rawConfidence = Number(analysis?.confidence)

  const confidenceScore = Number.isFinite(rawConfidence)
    ? Math.max(
        0,
        Math.min(
          100,
          rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence
        )
      )
    : 0

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="grid-overlay" />

      <header className="topbar">
        <a className="brand" href="#" aria-label="DeepShield home">
          <span className="brand-mark">
            <Shield size={22} />
          </span>

          <span>
            DEEP<span className="brand-accent">SHIELD</span>
          </span>

          <span className="version">BETA</span>
        </a>

        <div className="topbar-right">
          <span className="system-indicator">
            <span /> ANALYSIS CONSOLE
          </span>

          <div className="topbar-icon">
            <Fingerprint size={19} />
          </div>
        </div>
      </header>

      <main className="main-content">
        <motion.section
          className="hero"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65 }}
        >
          <div className="eyebrow hero-eyebrow">
            <span className="eyebrow-line" />
            DIGITAL TRUST INTELLIGENCE
          </div>

          <h1>
            Look beyond <span>what&apos;s real.</span>
          </h1>

          <p>
            Investigate media authenticity with AI-powered analysis.
          </p>

          <div className="hero-tags">
            <span><ScanFace size={14} /> IMAGE FORENSICS</span>
            <span><AudioLines size={14} /> AUDIO ANALYSIS</span>
            <span><FileVideo size={14} /> VIDEO WORKFLOW</span>
          </div>
        </motion.section>

        <section className="dashboard-grid">
          {/* MODULE 01: SCANNER */}
          <motion.section
            className="panel scanner-panel"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12, duration: 0.55 }}
          >
            <div className="panel-heading">
              <div className="heading-icon cyan">
                <ScanFace size={19} />
              </div>

              <div>
                <span className="panel-kicker">MODULE 01</span>
                <h2>Media scanner</h2>
              </div>

              <span className="panel-index">01 / 03</span>
            </div>

            <div
              className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragging(false)
                chooseFile(event.dataTransfer.files?.[0])
              }}
              onClick={() => {
                if (!loading) inputRef.current?.click()
              }}
              role="button"
              tabIndex={0}
              aria-label="Choose media for analysis"
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  if (!loading) inputRef.current?.click()
                }
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*,audio/*,video/*"
                onChange={(event) => {
                  chooseFile(event.target.files?.[0])
                  event.target.value = ''
                }}
                hidden
              />

              <div className="scanner-visual">
                <div className="scanner-ring ring-outer" />
                <div className="scanner-ring ring-inner" />

                <div className="scanner-core">
                  <MediaIcon size={27} />
                </div>

                {!file && <div className="scanner-sweep" />}
              </div>

              <h3>{file ? file.name : 'Drop media for inspection'}</h3>

              <p>
                {file
                  ? 'File selected and ready for analysis'
                  : 'Drag a file here or browse your device'}
              </p>

              <span className="supported-formats">
                IMAGES · AUDIO · VIDEO
              </span>
            </div>

            {file && (
              <motion.div
                className="selected-file"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
              >
                <MediaIcon size={18} />

                <div className="selected-file-info">
                  <strong>{file.name}</strong>
                  <span>
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>

                <button
                  type="button"
                  className="icon-button"
                  aria-label="Remove selected file"
                  onClick={(event) => {
                    event.stopPropagation()
                    removeFile()
                  }}
                >
                  <X size={17} />
                </button>
              </motion.div>
            )}

            {isImageFile(file) && previewUrl && (
              <div className="original-preview">
                <div className="evidence-card-heading">
                  <ImageIcon size={16} />
                  <span>SELECTED IMAGE PREVIEW</span>
                </div>

                <img
                  className="evidence-image"
                  src={previewUrl}
                  alt="Preview of selected image"
                />
              </div>
            )}

            <button
              type="button"
              className="primary-button"
              onClick={analyzeMedia}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <LoaderCircle className="spin" size={18} />
                  ANALYZING MEDIA...
                </>
              ) : (
                <>
                  <Sparkles size={17} />
                  INITIATE ANALYSIS
                  <ChevronRight size={18} />
                </>
              )}
            </button>

            {loading && (
              <button
                type="button"
                className="cancel-button"
                onClick={cancelCurrentRequest}
              >
                <CircleStop size={17} />
                CANCEL ANALYSIS
              </button>
            )}

            <p className="micro-copy">
              Results are estimates, not absolute proof of authenticity.
              Cancelling stops the browser request; server-side processing may continue.
            </p>

            <AnimatePresence>
              {error && (
                <motion.div
                  className="error-banner"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <CircleAlert size={18} />
                  <span>{error}</span>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.section>

          {/* MODULE 02: FORENSIC INTELLIGENCE */}
          <motion.section
            className="panel results-panel"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.22, duration: 0.55 }}
          >
            <div className="panel-heading">
              <div className="heading-icon violet">
                <Activity size={19} />
              </div>

              <div>
                <span className="panel-kicker">MODULE 02</span>
                <h2>Forensic intelligence</h2>
              </div>

              <span className="panel-index">02 / 03</span>
            </div>

            <AnimatePresence mode="wait">
              {loading ? (
                <motion.div
                  className="empty-state"
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="analysis-pulse">
                    <Activity size={30} />
                  </div>

                  <h3>Processing your media</h3>

                  <p>
                    Waiting for the detector and explanation service to respond.
                  </p>

                  <span className="live-label">
                    <span /> REQUEST IN PROGRESS
                  </span>
                </motion.div>
              ) : !analysis ? (
                <motion.div
                  className="empty-state"
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="empty-emblem">
                    <ShieldCheck size={33} />
                  </div>

                  <h3>Awaiting evidence</h3>

                  <p>
                    Upload media to populate this panel with real analysis results.
                  </p>

                  <div className="empty-divider">
                    <span /><span /><span />
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  className="analysis-result"
                  key="result"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                >
                  <div className="result-topline">
                    <span className="result-label">
                      DETECTOR CLASSIFICATION
                    </span>

                    <span className="result-received">
                      <CheckCircle2 size={14} />
                      RESPONSE RECEIVED
                    </span>
                  </div>

                  <h3 className="classification">
                    {analysis.result ??
                      analysis.overall_prediction ??
                      'Analysis complete'}
                  </h3>

                  <div className="confidence-card">
                    <div
                      className="confidence-ring"
                      style={{ '--score': `${confidenceScore}%` }}
                    >
                      <div className="confidence-ring-inner">
                        <strong>
                          {confidence ??
                            formatConfidence(
                              analysis.overall_fake_probability
                            ) ??
                            '—'}
                        </strong>
                        <span>
                          {analysis.confidence != null
                            ? 'CONFIDENCE'
                            : 'FAKE PROBABILITY'}
                        </span>
                      </div>
                    </div>

                    <div className="confidence-copy">
                      <span>MODEL OUTPUT</span>

                      <strong>
                        {confidence
                          ? 'Score returned'
                          : 'No confidence provided'}
                      </strong>

                      <p>
                        A model score is not a guarantee that a file is genuine
                        or manipulated.
                      </p>
                    </div>
                  </div>

                  <div className="detail-list">
                    {details.risk_level != null && (
                      <div>
                        <span>Risk level</span>
                        <strong>{String(details.risk_level)}</strong>
                      </div>
                    )}

                    {analysis.risk_level != null && (
                      <div>
                        <span>Risk level</span>
                        <strong>{String(analysis.risk_level)}</strong>
                      </div>
                    )}

                    {details.face_count != null && (
                      <div>
                        <span>Faces detected</span>
                        <strong>{String(details.face_count)}</strong>
                      </div>
                    )}

                    {details.fake_face_count != null && (
                      <div>
                        <span>Faces classified fake</span>
                        <strong>{String(details.fake_face_count)}</strong>
                      </div>
                    )}

                    {details.real_face_count != null && (
                      <div>
                        <span>Faces classified real</span>
                        <strong>{String(details.real_face_count)}</strong>
                      </div>
                    )}

                    {details.uncertain_face_count != null && (
                      <div>
                        <span>Uncertain faces</span>
                        <strong>{String(details.uncertain_face_count)}</strong>
                      </div>
                    )}

                    {details.ai_probability != null && (
                      <div>
                        <span>AI audio probability</span>
                        <strong>
                          {formatConfidence(details.ai_probability) ??
                            String(details.ai_probability)}
                        </strong>
                      </div>
                    )}

                    {details.human_probability != null && (
                      <div>
                        <span>Human audio probability</span>
                        <strong>
                          {formatConfidence(details.human_probability) ??
                            String(details.human_probability)}
                        </strong>
                      </div>
                    )}

                    {details.risk_reason && (
                      <div className="reason-row">
                        <span>Detector note</span>
                        <strong>{String(details.risk_reason)}</strong>
                      </div>
                    )}
                  </div>

                  {explanation && (
                    <div className="explanation-card">
                      <div className="explanation-title">
                        <Sparkles size={16} />
                        <h4>Gemini explanation</h4>
                      </div>

                      {explanation.summary && <p>{explanation.summary}</p>}
                      {explanation.why && <p>{explanation.why}</p>}

                      {explanation.simple_explanation && (
                        <p>{explanation.simple_explanation}</p>
                      )}

                      {explanation.confidence_explanation && (
                        <p>{explanation.confidence_explanation}</p>
                      )}

                      {explanation.user_advice && (
                        <div className="advice-note">
                          {explanation.user_advice}
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.section>

          {/* MODULE 03: AI ANALYST */}
          <motion.section
            className="panel analyst-panel"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.55 }}
          >
            <div className="panel-heading">
              <div className="heading-icon violet">
                <Bot size={19} />
              </div>

              <div>
                <span className="panel-kicker">MODULE 03</span>
                <h2>AI forensic analyst</h2>
              </div>

              <span className="panel-index">03 / 03</span>
            </div>

            <div className="analyst-intro">
              <span className="analyst-status">
                <span /> CONVERSATIONAL AGENT
              </span>

              <p>
                Your assistant can explain the current media analysis
                and the limitations of the detector.
              </p>
            </div>

            <DidAgent
              analysis={analysis}
              details={details}
              explanation={explanation}
            />

            <div className="analyst-footnote">
              <ShieldCheck size={15} />

              <span>
                The assistant explains supplied findings, not invented forensic evidence.
              </span>
            </div>
          </motion.section>
        </section>

        {analysis && isImageFile(file) && (
          <EvidenceViewer
            file={file}
            previewUrl={previewUrl}
            faces={details.faces}
          />
        )}

        <footer className="footer">
          <div className="footer-brand">
            <Shield size={15} />
            DEEPSHIELD <span>·</span> MEDIA AUTHENTICITY
          </div>

          <span>
            AI-assisted analysis · Verify important findings independently
          </span>
        </footer>
      </main>
    </div>
  )
}

export default App