import type { ModelOption } from "../api/client";

interface ModelSelectProps {
  disabled?: boolean;
  models: ModelOption[];
  selectedModel: string;
  onChange: (value: string) => void;
}

export function ModelSelect({
  disabled = false,
  models,
  selectedModel,
  onChange,
}: ModelSelectProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-semibold uppercase tracking-[0.18em] text-stone-600">
          Model
        </label>
        <span className="text-xs text-stone-500">{models.length} available</span>
      </div>

      <select
        className="w-full rounded-2xl border border-orange-200 bg-white/90 px-4 py-3 text-sm text-ink shadow-sm outline-none transition focus:border-accent"
        disabled={disabled || models.length === 0}
        value={selectedModel}
        onChange={(event) => onChange(event.target.value)}
      >
        {models.length === 0 ? (
          <option value="">No models found</option>
        ) : (
          models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.display_name} ({model.variant})
            </option>
          ))
        )}
      </select>

      {selectedModel ? (
        <div className="space-y-1 text-xs text-stone-500">
          <p>
            Inference weights:{" "}
            {models.find((model) => model.name === selectedModel)?.relative_path || selectedModel}
          </p>
          <p>
            Output format:{" "}
            {models.find((model) => model.name === selectedModel)?.num_classes === 1
              ? "1-logit probability model"
              : "2-class classifier"}
          </p>
        </div>
      ) : null}
    </div>
  );
}
