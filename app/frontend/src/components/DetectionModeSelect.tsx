import type { DetectionMode } from "../api/client";

interface DetectionModeSelectProps {
  disabled?: boolean;
  value: DetectionMode;
  onChange: (value: DetectionMode) => void;
}

const OPTIONS: Array<{
  value: DetectionMode;
  title: string;
  description: string;
}> = [
  {
    value: "full_crop",
    title: "Full crop detection",
    description: "Scan the full audio in 6-second chunks and report the most suspicious segment.",
  },
  {
    value: "random_6s_crop",
    title: "Random 6s crop",
    description: "Single random 6-second sample using the legacy inference path.",
  },
];

export function DetectionModeSelect({
  disabled = false,
  value,
  onChange,
}: DetectionModeSelectProps) {
  return (
    <div className="space-y-2">
      <span className="section-label">Detection Mode</span>

      <div className="grid gap-2" role="radiogroup" aria-label="Detection mode">
        {OPTIONS.map((option) => {
          const selected = option.value === value;
          return (
            <button
              key={option.value}
              role="radio"
              aria-checked={selected}
              className={`rounded-control border px-4 py-3 text-left transition ${
                selected
                  ? "border-accent bg-accent-soft shadow-sm"
                  : "border-border bg-white hover:border-stone-300 hover:shadow-sm"
              } disabled:cursor-not-allowed disabled:opacity-50`}
              disabled={disabled}
              onClick={() => onChange(option.value)}
              type="button"
            >
              <div className="flex items-center gap-3">
                <span
                  className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2 transition ${
                    selected ? "border-accent" : "border-stone-300"
                  }`}
                  aria-hidden="true"
                >
                  {selected ? (
                    <span className="h-2 w-2 rounded-full bg-accent" />
                  ) : null}
                </span>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-ink">{option.title}</p>
                  <p className="mt-0.5 text-xs text-muted leading-relaxed">{option.description}</p>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
