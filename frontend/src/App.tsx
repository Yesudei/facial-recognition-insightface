import {
  AlertCircle,
  Braces,
  CheckCircle2,
  ChevronDown,
  CircleGauge,
  Database,
  FileImage,
  ImageUp,
  Loader2,
  RefreshCcw,
  ScanFace,
  Sparkles,
  Upload,
  X
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";

type FaceResult = {
  bbox: [number, number, number, number];
  width: number;
  height: number;
  name: string;
  score: number | null;
  is_unknown: boolean;
  embedding_size: number;
};

type AnalyzeResult = {
  filename: string;
  faces_detected: number;
  database_loaded: boolean;
  database_path: string;
  threshold: number;
  faces: FaceResult[];
};

type ImageSize = {
  width: number;
  height: number;
};

const maxUploadBytes = 10 * 1024 * 1024;

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState("");
  const [showJson, setShowJson] = useState(false);
  const [imageSize, setImageSize] = useState<ImageSize>({ width: 0, height: 0 });

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }

    const nextPreviewUrl = URL.createObjectURL(file);
    setPreviewUrl(nextPreviewUrl);
    return () => URL.revokeObjectURL(nextPreviewUrl);
  }, [file]);

  const primaryFace = result?.faces[0] ?? null;
  const confidence = primaryFace?.score == null ? null : Math.round(primaryFace.score * 100);
  const json = useMemo(() => JSON.stringify(result, null, 2), [result]);

  function handleFile(nextFile?: File) {
    if (!nextFile) return;

    if (!nextFile.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }

    if (nextFile.size > maxUploadBytes) {
      setError("Image must be 10 MB or smaller.");
      return;
    }

    setFile(nextFile);
    setResult(null);
    setError("");
    setShowJson(false);
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    handleFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function onDragOver(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(true);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  }

  async function analyzeImage() {
    if (!file) return;

    setIsAnalyzing(true);
    setResult(null);
    setError("");
    setShowJson(false);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `Server returned ${response.status}`);
      }

      const data = (await response.json()) as AnalyzeResult;
      setResult(data);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Analysis failed.";
      setError(message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError("");
    setShowJson(false);
    setImageSize({ width: 0, height: 0 });
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">
            <ScanFace aria-hidden="true" />
          </span>
          <span>FaceID</span>
        </a>
        <div className="topbar-actions">
          <span className="status-pill">
            <Database aria-hidden="true" />
            InsightFace
          </span>
          <span className="status-pill status-pill-strong">
            <Sparkles aria-hidden="true" />
            ArcFace
          </span>
        </div>
      </header>

      <main className="workspace">
        <section className="hero-panel">
          <div className="hero-copy">
            <p className="eyebrow">Facial recognition assignment demo</p>
            <h1>Analyze faces with a cleaner React interface.</h1>
            <p className="hero-text">
              Upload an image, run the local InsightFace backend, and inspect matches,
              confidence scores, bounding boxes, and embeddings.
            </p>
          </div>
          <div className="hero-stats" aria-label="Recognition pipeline">
            <div>
              <span>01</span>
              <strong>Detect</strong>
            </div>
            <div>
              <span>02</span>
              <strong>Embed</strong>
            </div>
            <div>
              <span>03</span>
              <strong>Match</strong>
            </div>
          </div>
        </section>

        <section className="work-grid">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">Input</p>
                <h2>Upload image</h2>
              </div>
              {file ? (
                <button className="icon-button" type="button" onClick={reset} aria-label="Clear image">
                  <X aria-hidden="true" />
                </button>
              ) : null}
            </div>

            {!file ? (
              <label
                className={`drop-zone ${isDragging ? "is-dragging" : ""}`}
                onDragOver={onDragOver}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
              >
                <input type="file" accept="image/*" onChange={onInputChange} />
                <span className="drop-icon">
                  <ImageUp aria-hidden="true" />
                </span>
                <strong>Drop an image here</strong>
                <span>JPG, PNG, WEBP up to 10 MB</span>
                <span className="button-like">
                  <Upload aria-hidden="true" />
                  Browse
                </span>
              </label>
            ) : (
              <div className="preview-stack">
                <div className="preview-frame">
                  <img
                    src={previewUrl}
                    alt="Selected preview"
                    onLoad={(event) => {
                      setImageSize({
                        width: event.currentTarget.naturalWidth,
                        height: event.currentTarget.naturalHeight
                      });
                    }}
                  />
                </div>
                <div className="file-meta">
                  <FileImage aria-hidden="true" />
                  <div>
                    <strong>{file.name}</strong>
                    <span>{formatSize(file.size)}</span>
                  </div>
                </div>
              </div>
            )}

            <div className="panel-footer">
              <span className="subtle">
                {file ? "Ready for backend analysis" : "Waiting for image"}
              </span>
              <button
                className="primary-button"
                type="button"
                disabled={!file || isAnalyzing}
                onClick={analyzeImage}
              >
                {isAnalyzing ? <Loader2 className="spin" aria-hidden="true" /> : <ScanFace aria-hidden="true" />}
                {isAnalyzing ? "Analyzing" : "Analyze"}
              </button>
            </div>
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">Output</p>
                <h2>Recognition result</h2>
              </div>
              <ResultBadge result={result} isAnalyzing={isAnalyzing} />
            </div>

            {isAnalyzing ? (
              <div className="loading-state">
                <Loader2 className="spin" aria-hidden="true" />
                <strong>Running InsightFace</strong>
                <span>Detecting faces and comparing embeddings.</span>
                <div className="progress-track">
                  <span />
                </div>
              </div>
            ) : error ? (
              <div className="message-state error-state">
                <AlertCircle aria-hidden="true" />
                <strong>Analysis error</strong>
                <span>{error}</span>
              </div>
            ) : result && previewUrl ? (
              <div className="results-layout">
                <div className="annotated-image">
                  <img src={previewUrl} alt="Analyzed face result" />
                  {imageSize.width > 0 && imageSize.height > 0 ? (
                    <svg viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="none">
                      {result.faces.map((face, index) => (
                        <g key={`${face.bbox.join("-")}-${index}`}>
                          <rect
                            x={face.bbox[0]}
                            y={face.bbox[1]}
                            width={face.bbox[2] - face.bbox[0]}
                            height={face.bbox[3] - face.bbox[1]}
                            className={face.is_unknown ? "unknown-box" : "known-box"}
                          />
                          <text x={face.bbox[0]} y={Math.max(18, face.bbox[1] - 8)}>
                            {face.is_unknown ? "Unknown" : face.name}
                          </text>
                        </g>
                      ))}
                    </svg>
                  ) : null}
                </div>

                <div className="metrics">
                  <Metric label="Faces detected" value={String(result.faces_detected)} />
                  <Metric
                    label="Best match"
                    value={primaryFace ? primaryFace.name : "No face found"}
                  />
                  <Metric
                    label="Confidence"
                    value={confidence == null ? "No match" : `${confidence}%`}
                    barValue={confidence ?? 0}
                  />
                  <Metric
                    label="Face size"
                    value={primaryFace ? `${Math.round(primaryFace.width)} x ${Math.round(primaryFace.height)} px` : "-"}
                  />
                  <Metric
                    label="Embedding"
                    value={primaryFace ? `${primaryFace.embedding_size} dimensions` : "-"}
                  />
                </div>

                {!result.database_loaded ? (
                  <div className="database-note">
                    <AlertCircle aria-hidden="true" />
                    <span>
                      No enrolled database found at {result.database_path}. Run enrollment to enable named matches.
                    </span>
                  </div>
                ) : null}

                <button className="json-toggle" type="button" onClick={() => setShowJson((open) => !open)}>
                  <Braces aria-hidden="true" />
                  Raw JSON
                  <ChevronDown className={showJson ? "rotate" : ""} aria-hidden="true" />
                </button>
                {showJson ? <pre className="json-output">{json}</pre> : null}
              </div>
            ) : (
              <div className="message-state">
                <CircleGauge aria-hidden="true" />
                <strong>No analysis yet</strong>
                <span>Select an image to start.</span>
              </div>
            )}
          </div>
        </section>

        {result ? (
          <div className="reset-row">
            <button className="secondary-button" type="button" onClick={reset}>
              <RefreshCcw aria-hidden="true" />
              Analyze another image
            </button>
          </div>
        ) : null}
      </main>
    </div>
  );
}

function ResultBadge({
  result,
  isAnalyzing
}: {
  result: AnalyzeResult | null;
  isAnalyzing: boolean;
}) {
  if (isAnalyzing) {
    return <span className="result-badge neutral">Processing</span>;
  }

  if (!result) {
    return <span className="result-badge neutral">Idle</span>;
  }

  return (
    <span className={`result-badge ${result.faces_detected > 0 ? "success" : "warning"}`}>
      {result.faces_detected > 0 ? <CheckCircle2 aria-hidden="true" /> : <AlertCircle aria-hidden="true" />}
      {result.faces_detected} {result.faces_detected === 1 ? "face" : "faces"}
    </span>
  );
}

function Metric({
  label,
  value,
  barValue
}: {
  label: string;
  value: string;
  barValue?: number;
}) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong>{value}</strong>
      {typeof barValue === "number" ? (
        <div className="confidence-track" aria-hidden="true">
          <span style={{ width: `${Math.max(0, Math.min(100, barValue))}%` }} />
        </div>
      ) : null}
    </div>
  );
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
