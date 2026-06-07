interface StatusMessageProps {
  tone?: "info" | "error";
  message: string;
}

export function StatusMessage({
  tone = "info",
  message,
}: StatusMessageProps) {
  const classes =
    tone === "error"
      ? "border-red-200 bg-red-50 text-danger"
      : "border-orange-200 bg-orange-50 text-stone-700";

  return (
    <div className={`rounded-2xl border px-4 py-3 text-sm ${classes}`}>{message}</div>
  );
}
