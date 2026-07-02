interface StatusMessageProps {
  tone?: "info" | "error";
  message: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export function StatusMessage({
  tone = "info",
  message,
  action,
}: StatusMessageProps) {
  const isError = tone === "error";

  return (
    <div
      className={`flex items-start gap-3 rounded-control border px-4 py-3 text-sm ${
        isError
          ? "border-red-200 bg-danger-soft text-danger"
          : "border-amber-200 bg-accent-soft text-stone-700"
      }`}
      role={isError ? "alert" : "status"}
    >
      <span className="mt-0.5 shrink-0 text-base" aria-hidden="true">
        {isError ? "⚠" : "ℹ"}
      </span>
      <span className="flex-1">{message}</span>
      {action ? (
        <button
          className={`shrink-0 rounded-pill px-3 py-1 text-xs font-semibold transition ${
            isError
              ? "bg-danger/10 text-danger hover:bg-danger/20"
              : "bg-accent/10 text-accent hover:bg-accent/20"
          }`}
          onClick={action.onClick}
          type="button"
        >
          {action.label}
        </button>
      ) : null}
    </div>
  );
}
