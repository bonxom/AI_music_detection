import type { DetectionResponse } from "../api/client";

interface ResultPanelProps {
  result: DetectionResponse | null;
}

function ProbabilityBar({ value }: { value: number }) {
  return (
    <div className="h-3 overflow-hidden rounded-full bg-orange-100">
      <div
        className="h-full rounded-full bg-gradient-to-r from-accent to-orange-400 transition-all duration-500"
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
}

function formatDetectionMode(mode: DetectionResponse["detection_mode"]) {
  return mode === "full_crop" ? "Detect full crop" : "Detect random 6s crop";
}

export function ResultPanel({ result }: ResultPanelProps) {
  if (!result) {
    return (
      <section className="rounded-[32px] border border-white/70 bg-white/70 p-6 shadow-panel backdrop-blur">
        <p className="text-sm uppercase tracking-[0.2em] text-stone-500">Result</p>
        <h2 className="mt-3 text-2xl font-semibold text-ink">Awaiting analysis</h2>
        <p className="mt-3 text-sm text-stone-600">
          Upload a video, choose a model and detection mode, then run inference to see the
          AI-generated probability.
        </p>
      </section>
    );
  }

  const fullCrop = result.full_crop_summary;

  return (
    <section className="rounded-[32px] border border-white/70 bg-white/80 p-6 shadow-panel backdrop-blur">
      <p className="text-sm uppercase tracking-[0.2em] text-stone-500">Result</p>
      <div className="mt-4 flex items-end justify-between gap-4">
        <div>
          <h2 className="text-4xl font-semibold text-ink">
            {result.ai_generated_probability.toFixed(2)}%
          </h2>
          <p className="mt-2 text-sm text-stone-700">
            AI-generated probability: {result.ai_generated_probability.toFixed(2)}%
          </p>
        </div>
        <span className="rounded-full bg-stone-900 px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-white">
          {result.label === "ai" ? "Predicted AI" : "Predicted Human"}
        </span>
      </div>

      <div className="mt-5">
        <ProbabilityBar value={result.ai_generated_probability} />
      </div>

      <dl className="mt-6 grid gap-4 text-sm text-stone-700 sm:grid-cols-2">
        <div className="rounded-2xl bg-orange-50 p-4">
          <dt className="text-xs uppercase tracking-[0.18em] text-stone-500">Selected model</dt>
          <dd className="mt-2 font-medium text-ink">{result.selected_model.display_name}</dd>
        </div>
        <div className="rounded-2xl bg-orange-50 p-4">
          <dt className="text-xs uppercase tracking-[0.18em] text-stone-500">Status</dt>
          <dd className="mt-2 font-medium capitalize text-ink">{result.processing_status}</dd>
        </div>
        <div className="rounded-2xl bg-orange-50 p-4">
          <dt className="text-xs uppercase tracking-[0.18em] text-stone-500">Uploaded file</dt>
          <dd className="mt-2 font-medium text-ink">{result.filename}</dd>
        </div>
        <div className="rounded-2xl bg-orange-50 p-4">
          <dt className="text-xs uppercase tracking-[0.18em] text-stone-500">Detection mode</dt>
          <dd className="mt-2 font-medium text-ink">
            {formatDetectionMode(result.detection_mode)}
          </dd>
        </div>
      </dl>

      {fullCrop ? (
        <div className="mt-6 rounded-[28px] border border-orange-200 bg-white p-5">
          <p className="text-xs uppercase tracking-[0.18em] text-stone-500">Full crop details</p>
          <dl className="mt-4 grid gap-4 text-sm text-stone-700 sm:grid-cols-2">
            <div>
              <dt className="text-xs uppercase tracking-[0.18em] text-stone-500">Chunks scanned</dt>
              <dd className="mt-2 font-medium text-ink">{fullCrop.num_chunks}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-[0.18em] text-stone-500">
                Suspected AI chunks
              </dt>
              <dd className="mt-2 font-medium text-ink">
                {fullCrop.count_chunks_non_ai_lt_50}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-[0.18em] text-stone-500">Best chunk</dt>
              <dd className="mt-2 font-medium text-ink">#{fullCrop.best_chunk_index}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-[0.18em] text-stone-500">Time range</dt>
              <dd className="mt-2 font-medium text-ink">
                {fullCrop.best_start_sec.toFixed(2)}s to {fullCrop.best_end_sec.toFixed(2)}s
              </dd>
            </div>
          </dl>
        </div>
      ) : null}
    </section>
  );
}
