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
  const noModels = models.length === 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label
          htmlFor="model-select"
          className="section-label"
        >
          Model
        </label>
        <span className="text-xs text-muted">
          {noModels ? "0 available" : `${models.length} available`}
        </span>
      </div>

      <select
        id="model-select"
        aria-label="Select a detection model"
        className="w-full rounded-control border border-border bg-white px-4 py-3 text-sm text-ink shadow-sm outline-none transition focus:border-accent disabled:cursor-not-allowed disabled:opacity-50"
        disabled={disabled || noModels}
        value={selectedModel}
        onChange={(event) => onChange(event.target.value)}
      >
        {noModels ? (
          <option value="">No models found</option>
        ) : (
          models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.display_name} ({model.variant})
            </option>
          ))
        )}
      </select>

      {noModels && !disabled ? (
        <p className="text-xs text-danger">
          No model files were found in app/model_prob/. Add a model and restart the backend.
        </p>
      ) : null}

      {selectedModel && !noModels ? (
        <div className="space-y-0.5 text-xs text-muted">
          <p>
            Weights:{" "}
            <span className="text-ink/70">
              {models.find((model) => model.name === selectedModel)?.relative_path || selectedModel}
            </span>
          </p>
          <p>
            Output:{" "}
            <span className="text-ink/70">
              {models.find((model) => model.name === selectedModel)?.num_classes === 1
                ? "1-logit probability"
                : "2-class softmax"}
            </span>
          </p>
        </div>
      ) : null}
    </div>
  );
}
