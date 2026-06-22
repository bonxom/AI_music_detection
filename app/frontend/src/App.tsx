import { useCallback, useEffect, useState } from "react";

import {
  analyzeVideo,
  fetchModels,
  selectModel,
  type DetectionMode,
  type DetectionResponse,
  type ModelOption,
} from "./api/client";
import { DetectionModeSelect } from "./components/DetectionModeSelect";
import { MediaPreview } from "./components/MediaPreview";
import { ModelSelect } from "./components/ModelSelect";
import { ResultPanel } from "./components/ResultPanel";
import { StatusMessage } from "./components/StatusMessage";
import { VideoUploadField } from "./components/VideoUploadField";

const SUPPORTED_MEDIA_EXTENSIONS = [
  ".mp4",
  ".mov",
  ".mkv",
  ".avi",
  ".webm",
  ".flv",
  ".m4v",
  ".wmv",
  ".mp3",
  ".wav",
  ".flac",
  ".aac",
  ".ogg",
  ".m4a",
  ".wma",
  ".opus",
];

function isSupportedMediaFile(file: File) {
  if (file.type.startsWith("video/") || file.type.startsWith("audio/")) {
    return true;
  }
  const fileName = file.name.toLowerCase();
  return SUPPORTED_MEDIA_EXTENSIONS.some((extension) => fileName.endsWith(extension));
}

function modeLabel(mode: DetectionMode) {
  return mode === "full_crop" ? "Full crop detection" : "Random 6s crop";
}

export default function App() {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [detectionMode, setDetectionMode] = useState<DetectionMode>("full_crop");
  const [mediaFile, setMediaFile] = useState<File | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loadingModels, setLoadingModels] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Loading available models…");
  const [errorMessage, setErrorMessage] = useState("");
  const [backendFailed, setBackendFailed] = useState(false);

  const loadModels = useCallback(async () => {
    setLoadingModels(true);
    setErrorMessage("");
    setBackendFailed(false);
    setStatusMessage("Loading available models…");

    try {
      const response = await fetchModels();
      setModels(response.models);

      const activeModelName =
        response.active_model?.name ||
        response.models.find((model) => model.active)?.name ||
        response.models[0]?.name ||
        "";
      setSelectedModel(activeModelName);
      setStatusMessage("Ready to analyze.");
    } catch (error) {
      setBackendFailed(true);
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to load available models.",
      );
      setStatusMessage("Backend connection failed.");
    } finally {
      setLoadingModels(false);
    }
  }, []);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const handleVideoChange = (file: File | null) => {
    setErrorMessage("");
    setResult(null);

    if (!file) {
      setMediaFile(null);
      return;
    }

    if (!isSupportedMediaFile(file)) {
      setMediaFile(null);
      setErrorMessage("Please select a supported audio or video file.");
      return;
    }

    setMediaFile(file);
  };

  const handleSubmit = async () => {
    if (!selectedModel) {
      setErrorMessage("Please choose a model before starting inference.");
      return;
    }
    if (!mediaFile) {
      setErrorMessage("Please upload an audio or video file first.");
      return;
    }

    setSubmitting(true);
    setErrorMessage("");
    setResult(null);

    try {
      setStatusMessage("Selecting model…");
      await selectModel(selectedModel);
      setModels((currentModels) =>
        currentModels.map((model) => ({
          ...model,
          active: model.name === selectedModel,
        })),
      );

      setStatusMessage(`Running ${modeLabel(detectionMode).toLowerCase()} inference…`);
      const response = await analyzeVideo(mediaFile, detectionMode);
      setResult(response);
      setStatusMessage("Analysis completed.");
    } catch (error) {
      setStatusMessage("Analysis failed.");
      setErrorMessage(error instanceof Error ? error.message : "Unexpected request error.");
    } finally {
      setSubmitting(false);
    }
  };

  // Determine button disabled reason
  const analyzeDisabled = loadingModels || submitting || models.length === 0 || !mediaFile || backendFailed;
  const getDisabledReason = () => {
    if (backendFailed) return "Backend is not connected";
    if (loadingModels) return "Loading models…";
    if (models.length === 0) return "No models available";
    if (!mediaFile) return "No file selected";
    if (submitting) return "Analysis in progress";
    return undefined;
  };
  const disabledReason = getDisabledReason();

  // Backend status for header pill
  const backendStatus = backendFailed
    ? "disconnected"
    : loadingModels
      ? "connecting"
      : "connected";

  return (
    <div className="min-h-screen px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-page">
        {/* ─── Compact Header ─── */}
        <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-ink sm:text-2xl">
              AI Music Detection
            </h1>
            <p className="mt-0.5 text-sm text-muted">
              Upload audio or video and detect AI-generated content
            </p>
          </div>

          {/* Status pill */}
          <div
            className={`flex items-center gap-2 rounded-pill border px-3 py-1.5 text-xs font-medium ${
              backendStatus === "connected"
                ? "border-emerald-200 bg-success-soft text-success"
                : backendStatus === "connecting"
                  ? "border-amber-200 bg-accent-soft text-amber-700"
                  : "border-red-200 bg-danger-soft text-danger"
            }`}
            role="status"
            aria-label={`Backend status: ${backendStatus}`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                backendStatus === "connected"
                  ? "bg-success"
                  : backendStatus === "connecting"
                    ? "bg-amber-500 animate-subtle-pulse"
                    : "bg-danger"
              }`}
              aria-hidden="true"
            />
            {backendStatus === "connected"
              ? `${models.length} model${models.length !== 1 ? "s" : ""} loaded`
              : backendStatus === "connecting"
                ? "Connecting…"
                : "Backend offline"}
          </div>
        </header>

        {/* ─── Main Grid ─── */}
        <div className="grid items-start gap-6 lg:grid-cols-[7fr_5fr]">
          {/* ─── LEFT: Configure Analysis ─── */}
          <section className="order-2 rounded-card border border-border bg-white p-6 shadow-card lg:order-1">
            <h2 className="text-base font-semibold text-ink">Configure Analysis</h2>

            <div className="mt-5 space-y-5">
              {/* Backend error alert */}
              {backendFailed ? (
                <StatusMessage
                  tone="error"
                  message="Backend connection failed. Make sure the backend server is running, then refresh models."
                  action={{ label: "Retry connection", onClick: () => void loadModels() }}
                />
              ) : null}

              {/* Validation error */}
              {errorMessage && !backendFailed ? (
                <StatusMessage tone="error" message={errorMessage} />
              ) : null}

              {/* Model selector */}
              <ModelSelect
                disabled={loadingModels || submitting || backendFailed}
                models={models}
                selectedModel={selectedModel}
                onChange={setSelectedModel}
              />

              {/* Divider */}
              <hr className="border-border" />

              {/* Detection mode */}
              <DetectionModeSelect
                disabled={loadingModels || submitting || backendFailed}
                value={detectionMode}
                onChange={setDetectionMode}
              />

              {/* Divider */}
              <hr className="border-border" />

              {/* File upload */}
              <VideoUploadField
                disabled={loadingModels || submitting || backendFailed}
                file={mediaFile}
                onChange={handleVideoChange}
              />

              {/* Analyze button */}
              <button
                id="analyze-button"
                className="inline-flex w-full items-center justify-center gap-2 rounded-pill bg-accent px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-accent-hover hover:shadow-md disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
                disabled={analyzeDisabled}
                onClick={() => void handleSubmit()}
                type="button"
                title={disabledReason}
                aria-disabled={analyzeDisabled}
              >
                {submitting ? (
                  <>
                    <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Analyzing…
                  </>
                ) : (
                  "Analyze"
                )}
              </button>

              {/* Status info (non-error) */}
              {!backendFailed && statusMessage ? (
                <p className="text-center text-xs text-muted">{statusMessage}</p>
              ) : null}
            </div>
          </section>

          {/* ─── RIGHT: Results & Preview ─── */}
          <div className="order-1 space-y-5 lg:order-2">
            {/* Result card */}
            <ResultPanel result={result} loading={submitting} />

            {/* Media preview */}
            <MediaPreview file={mediaFile} />

            {/* Session details — light card */}
            <section className="rounded-card border border-border bg-white p-5 shadow-card">
              <p className="section-label">Session Details</p>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div className="rounded-control bg-stone-50 px-3 py-2.5">
                  <p className="text-xs text-muted">Active model</p>
                  <p className="mt-0.5 text-sm font-semibold text-ink">
                    {models.find((model) => model.name === selectedModel)?.display_name ||
                      "Unavailable"}
                  </p>
                </div>
                <div className="rounded-control bg-stone-50 px-3 py-2.5">
                  <p className="text-xs text-muted">Detection mode</p>
                  <p className="mt-0.5 text-sm font-semibold text-ink">{modeLabel(detectionMode)}</p>
                </div>
                <div className="rounded-control bg-stone-50 px-3 py-2.5">
                  <p className="text-xs text-muted">Processing state</p>
                  <p className="mt-0.5 text-sm font-semibold text-ink">
                    {submitting ? "Running inference" : "Idle"}
                  </p>
                </div>
                <div className="rounded-control bg-stone-50 px-3 py-2.5">
                  <p className="text-xs text-muted">Output format</p>
                  <p className="mt-0.5 text-sm font-semibold text-ink">
                    {models.find((model) => model.name === selectedModel)?.num_classes === 1
                      ? "1-logit probability"
                      : "2-class softmax"}
                  </p>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
