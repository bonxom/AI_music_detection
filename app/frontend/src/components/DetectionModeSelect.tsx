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
    title: "Detect full crop",
    description: "Scan the full audio in contiguous 6-second chunks and report the most suspicious segment.",
  },
  {
    value: "random_6s_crop",
    title: "Detect random 6s crop",
    description: "Use the legacy single 6-second inference path from the current backend flow.",
  },
];

export function DetectionModeSelect({
  disabled = false,
  value,
  onChange,
}: DetectionModeSelectProps) {
  return (
    <div className="space-y-3">
      <label className="text-sm font-semibold uppercase tracking-[0.18em] text-stone-600">
        Detection Mode
      </label>

      <div className="grid gap-3">
        {OPTIONS.map((option) => {
          const selected = option.value === value;
          return (
            <button
              key={option.value}
              className={`rounded-[24px] border px-4 py-4 text-left transition ${
                selected
                  ? "border-accent bg-orange-50 shadow-sm"
                  : "border-orange-200 bg-white/85 hover:border-orange-300"
              }`}
              disabled={disabled}
              onClick={() => onChange(option.value)}
              type="button"
            >
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm font-semibold text-ink">{option.title}</p>
                  <p className="mt-1 text-sm text-stone-600">{option.description}</p>
                </div>
                <span
                  className={`h-4 w-4 rounded-full border ${
                    selected ? "border-accent bg-accent" : "border-stone-300 bg-white"
                  }`}
                />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
