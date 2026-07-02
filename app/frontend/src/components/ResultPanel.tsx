import type { DetectionResponse } from "../api/client";

interface ResultPanelProps {
  result: DetectionResponse | null;
  loading?: boolean;
}

function ProbabilityBar({ value, color }: { value: number; color: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div className="relative h-3 overflow-hidden rounded-full bg-stone-100">
      <div
        className={`h-full rounded-full transition-all duration-700 ease-out ${color}`}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}

function formatDetectionMode(mode: DetectionResponse["detection_mode"]) {
  return mode === "full_crop" ? "Full crop detection" : "Random 6s crop";
}

export function ResultPanel({ result, loading = false }: ResultPanelProps) {
  /* ---------- Loading state ---------- */
  if (loading) {
    return (
      <section
        className="rounded-card border border-border bg-white p-6 shadow-card"
        aria-busy="true"
        aria-live="polite"
      >
        <p className="section-label">Analysis Result</p>
        <div className="mt-6 flex flex-col items-center gap-3 py-6">
          <svg className="h-8 w-8 animate-spin text-accent" fill="none" viewBox="0 0 24 24" aria-hidden="true">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <p className="text-sm font-medium text-ink">Running analysis…</p>
          <p className="text-xs text-muted">This may take a moment depending on file length.</p>
        </div>
      </section>
    );
  }

  /* ---------- Empty state ---------- */
  if (!result) {
    return (
      <section
        className="rounded-card border border-border bg-white p-6 shadow-card"
        aria-live="polite"
      >
        <p className="section-label">Analysis Result</p>
        <div className="mt-6 flex flex-col items-center gap-2 py-6 text-center">
          <svg className="h-10 w-10 text-stone-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
          </svg>
          <h2 className="text-lg font-semibold text-ink">Awaiting analysis</h2>
          <p className="max-w-xs text-sm text-muted">
            Upload a file, choose a model and detection mode, then run analysis to see results.
          </p>
        </div>
      </section>
    );
  }

  /* ---------- Success state ---------- */
  const isAI = result.label === "ai";
  const prob = result.ai_generated_probability;
  const fullCrop = result.full_crop_summary;

  return (
    <section
      className="rounded-card border border-border bg-white p-6 shadow-card"
      aria-live="polite"
    >
      <div className="flex items-center justify-between">
        <p className="section-label">Analysis Result</p>
        <span
          className={`rounded-pill px-3 py-1 text-xs font-semibold ${
            isAI
              ? "bg-danger-soft text-danger"
              : "bg-success-soft text-success"
          }`}
        >
          {isAI ? "Predicted AI" : "Predicted Human"}
        </span>
      </div>

      {/* Main probability */}
      <div className="mt-5">
        <p className="text-4xl font-bold tracking-tight text-ink">
          {prob.toFixed(1)}
          <span className="text-2xl text-muted">%</span>
        </p>
        <p className="mt-1 text-sm text-muted">AI-generated probability</p>
      </div>

      {/* Progress bar */}
      <div className="mt-4">
        <ProbabilityBar
          value={prob}
          color={isAI ? "bg-gradient-to-r from-red-400 to-red-500" : "bg-gradient-to-r from-emerald-400 to-emerald-500"}
        />
        <div className="mt-1 flex justify-between text-xs text-muted">
          <span>Human</span>
          <span>AI</span>
        </div>
      </div>

      {/* Details grid */}
      <dl className="mt-5 grid gap-2 text-sm sm:grid-cols-2">
        <div className="rounded-control bg-stone-50 px-3 py-2.5">
          <dt className="text-xs text-muted">Model</dt>
          <dd className="mt-0.5 font-medium text-ink">{result.selected_model.display_name}</dd>
        </div>
        <div className="rounded-control bg-stone-50 px-3 py-2.5">
          <dt className="text-xs text-muted">Status</dt>
          <dd className="mt-0.5 font-medium capitalize text-ink">{result.processing_status}</dd>
        </div>
        <div className="rounded-control bg-stone-50 px-3 py-2.5">
          <dt className="text-xs text-muted">File</dt>
          <dd className="mt-0.5 truncate font-medium text-ink">{result.filename}</dd>
        </div>
        <div className="rounded-control bg-stone-50 px-3 py-2.5">
          <dt className="text-xs text-muted">Mode</dt>
          <dd className="mt-0.5 font-medium text-ink">
            {formatDetectionMode(result.detection_mode)}
          </dd>
        </div>
      </dl>

      {/* Full crop details */}
      {fullCrop ? (
        <div className="mt-4 rounded-control border border-border bg-stone-50/50 p-4">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted">
            Full crop details
          </p>
          <dl className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
            <div>
              <dt className="text-xs text-muted">Chunks scanned</dt>
              <dd className="mt-0.5 font-medium text-ink">{fullCrop.num_chunks}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted">Suspected AI chunks</dt>
              <dd className="mt-0.5 font-medium text-ink">{fullCrop.count_chunks_non_ai_lt_50}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted">Best chunk</dt>
              <dd className="mt-0.5 font-medium text-ink">#{fullCrop.best_chunk_index}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted">Time range</dt>
              <dd className="mt-0.5 font-medium text-ink">
                {fullCrop.best_start_sec.toFixed(2)}s – {fullCrop.best_end_sec.toFixed(2)}s
              </dd>
            </div>
          </dl>
        </div>
      ) : null}
    </section>
  );
}
