import { useEffect, useState } from "react";

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
  return mode === "full_crop" ? "Detect full crop" : "Detect random 6s crop";
}

export default function App() {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [detectionMode, setDetectionMode] = useState<DetectionMode>("full_crop");
  const [mediaFile, setMediaFile] = useState<File | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loadingModels, setLoadingModels] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Loading available probability models...");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await fetchModels();
        setModels(response.models);

        const activeModelName =
          response.active_model?.name ||
          response.models.find((model) => model.active)?.name ||
          response.models[0]?.name ||
          "";
        setSelectedModel(activeModelName);
        setStatusMessage("Ready to analyze a video.");
      } catch (error) {
        setErrorMessage(
          error instanceof Error ? error.message : "Failed to load available models.",
        );
        setStatusMessage("Backend connection failed.");
      } finally {
        setLoadingModels(false);
      }
    };

    void loadModels();
  }, []);

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
      setStatusMessage("Selecting model...");
      await selectModel(selectedModel);
      setModels((currentModels) =>
        currentModels.map((model) => ({
          ...model,
          active: model.name === selectedModel,
        })),
      );

      setStatusMessage(`Running ${modeLabel(detectionMode).toLowerCase()} inference...`);
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

  return (
    <main className="min-h-screen px-4 py-8 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl">
        <section className="overflow-hidden rounded-[40px] border border-white/60 bg-white/45 p-6 shadow-panel backdrop-blur md:p-10">
          <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="space-y-6">
              <div className="space-y-4">
                <p className="text-sm font-semibold uppercase tracking-[0.28em] text-stone-500">
                  AI Music Detection
                </p>
                <h1 className="max-w-xl text-4xl font-semibold leading-tight text-ink md:text-6xl">
                  Audio detection models, now in a user-friendly interface.
                </h1>
                <p className="max-w-2xl text-base text-stone-700 md:text-lg">
                  The backend now uses the probability models from `app/model_prob/` and supports
                  both the notebook full-crop pipeline and the legacy 6-second path.
                </p>
              </div>

              <section className="space-y-5 rounded-[32px] border border-white/70 bg-white/80 p-6">
                <ModelSelect
                  disabled={loadingModels || submitting}
                  models={models}
                  selectedModel={selectedModel}
                  onChange={setSelectedModel}
                />

                <DetectionModeSelect
                  disabled={loadingModels || submitting}
                  value={detectionMode}
                  onChange={setDetectionMode}
                />

                <VideoUploadField
                  disabled={loadingModels || submitting}
                  file={mediaFile}
                  onChange={handleVideoChange}
                />

                <button
                  className="inline-flex w-full items-center justify-center rounded-full bg-stone-950 px-6 py-3 text-sm font-semibold text-white transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:bg-stone-400"
                  disabled={loadingModels || submitting || models.length === 0}
                  onClick={() => void handleSubmit()}
                  type="button"
                >
                  {submitting ? "Analyzing..." : "Analyze Video"}
                </button>
              </section>

              <div className="space-y-3">
                <StatusMessage message={statusMessage} />
                {errorMessage ? <StatusMessage tone="error" message={errorMessage} /> : null}
              </div>
            </div>

            <div className="space-y-6">
              <ResultPanel result={result} />

              <section className="rounded-[32px] border border-white/70 bg-stone-950 p-6 text-stone-100 shadow-panel">
                <p className="text-sm uppercase tracking-[0.2em] text-stone-400">Session</p>
                <div className="mt-4 grid gap-4 sm:grid-cols-2">
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-stone-400">
                      Active model
                    </p>
                    <p className="mt-2 text-lg font-semibold">
                      {models.find((model) => model.name === selectedModel)?.display_name ||
                        "Unavailable"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-stone-400">
                      Detection mode
                    </p>
                    <p className="mt-2 text-lg font-semibold">{modeLabel(detectionMode)}</p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-stone-400">
                      Processing state
                    </p>
                    <p className="mt-2 text-lg font-semibold">
                      {submitting ? "Running inference" : "Idle"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-stone-400">
                      Output format
                    </p>
                    <p className="mt-2 text-lg font-semibold">
                      {models.find((model) => model.name === selectedModel)?.num_classes === 1
                        ? "1-logit probability"
                        : "2-class softmax"}
                    </p>
                  </div>
                </div>
              </section>

              <MediaPreview file={mediaFile} />
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
