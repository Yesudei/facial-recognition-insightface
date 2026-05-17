import {
  AlertCircle,
  CheckCircle2,
  CircleGauge,
  FileImage,
  ImageUp,
  Loader2,
  RefreshCcw,
  ScanFace,
  Upload,
  X
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useState } from "react";

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

  function handleFile(nextFile?: File) {
    if (!nextFile) return;

    if (!nextFile.type.startsWith("image/")) {
      setError("Зургийн файл сонгоно уу.");
      return;
    }

    if (nextFile.size > maxUploadBytes) {
      setError("Зураг 10 MB-аас бага байх ёстой.");
      return;
    }

    setFile(nextFile);
    setResult(null);
    setError("");
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

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Зургийг шинжилж чадсангүй.");
      }

      const data = (await response.json()) as AnalyzeResult;
      setResult(data);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Зургийг шинжилж чадсангүй.";
      setError(message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError("");
    setImageSize({ width: 0, height: 0 });
  }

  return (
    <div className="app-shell">
      <main className="workspace">
        <section className="work-grid">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">Оролт</p>
                <h2>Зураг оруулах</h2>
              </div>
              {file ? (
                <button className="icon-button" type="button" onClick={reset} aria-label="Зураг арилгах">
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
                <strong>Зургаа энд оруулна уу</strong>
                <span>JPG, PNG, WEBP - 10 MB хүртэл</span>
                <span className="button-like">
                  <Upload aria-hidden="true" />
                  Сонгох
                </span>
              </label>
            ) : (
              <div className="preview-stack">
                <div className="preview-frame">
                  <img
                    src={previewUrl}
                    alt="Сонгосон зураг"
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
                {file ? "Шинжлэхэд бэлэн" : "Зураг хүлээж байна"}
              </span>
              <button
                className="primary-button"
                type="button"
                disabled={!file || isAnalyzing}
                onClick={analyzeImage}
              >
                {isAnalyzing ? <Loader2 className="spin" aria-hidden="true" /> : <ScanFace aria-hidden="true" />}
                {isAnalyzing ? "Шинжилж байна" : "Шинжлэх"}
              </button>
            </div>
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">Гаралт</p>
                <h2>Үр дүн</h2>
              </div>
              <ResultBadge result={result} isAnalyzing={isAnalyzing} />
            </div>

            {isAnalyzing ? (
              <div className="loading-state">
                <Loader2 className="spin" aria-hidden="true" />
                <strong>Шинжилж байна</strong>
                <span>Царай хайж байна.</span>
                <div className="progress-track">
                  <span />
                </div>
              </div>
            ) : error ? (
              <div className="message-state error-state">
                <AlertCircle aria-hidden="true" />
                <strong>Алдаа гарлаа</strong>
                <span>{error}</span>
              </div>
            ) : result && previewUrl ? (
              <div className="results-layout">
                <div className="annotated-image">
                  <img src={previewUrl} alt="Шинжилсэн зураг" />
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
                            {face.is_unknown ? "Танигдаагүй" : face.name}
                          </text>
                        </g>
                      ))}
                    </svg>
                  ) : null}
                </div>

                <div className="metrics">
                  <Metric label="Олдсон царай" value={String(result.faces_detected)} />
                  <Metric
                    label="Нэр"
                    value={primaryFace ? (primaryFace.is_unknown ? "Танигдаагүй" : primaryFace.name) : "Царай олдсонгүй"}
                  />
                  <Metric
                    label="Итгэлцүүр"
                    value={confidence == null ? "Тохиролгүй" : `${confidence}%`}
                    barValue={confidence ?? 0}
                  />
                  <Metric
                    label="Хэмжээ"
                    value={primaryFace ? `${Math.round(primaryFace.width)} x ${Math.round(primaryFace.height)} px` : "-"}
                  />
                  <Metric
                    label="Вектор"
                    value={primaryFace ? `${primaryFace.embedding_size}` : "-"}
                  />
                </div>

              </div>
            ) : (
              <div className="message-state">
                <CircleGauge aria-hidden="true" />
                <strong>Шинжилгээ хийгдээгүй</strong>
                <span>Эхлээд зураг сонгоно уу.</span>
              </div>
            )}
          </div>
        </section>

        {result ? (
          <div className="reset-row">
            <button className="secondary-button" type="button" onClick={reset}>
              <RefreshCcw aria-hidden="true" />
              Дахин зураг сонгох
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
    return <span className="result-badge neutral">Шинжилж байна</span>;
  }

  if (!result) {
    return <span className="result-badge neutral">Бэлэн</span>;
  }

  return (
    <span className={`result-badge ${result.faces_detected > 0 ? "success" : "warning"}`}>
      {result.faces_detected > 0 ? <CheckCircle2 aria-hidden="true" /> : <AlertCircle aria-hidden="true" />}
      {result.faces_detected} царай
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
