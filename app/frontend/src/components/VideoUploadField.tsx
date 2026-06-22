interface VideoUploadFieldProps {
  disabled?: boolean;
  file: File | null;
  onChange: (file: File | null) => void;
}

export function VideoUploadField({
  disabled = false,
  file,
  onChange,
}: VideoUploadFieldProps) {
  return (
    <div className="space-y-2">
      <span className="section-label">Media File</span>

      <label
        className={`flex cursor-pointer flex-col items-center gap-2 rounded-control border-2 border-dashed px-6 py-5 text-center transition ${
          disabled
            ? "cursor-not-allowed border-stone-200 bg-stone-50 opacity-50"
            : "border-stone-300 bg-white hover:border-accent hover:bg-accent-soft/40"
        }`}
      >
        {/* Upload icon */}
        <svg
          className="h-8 w-8 text-muted"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
          />
        </svg>

        <span className="text-sm font-medium text-ink">
          Choose audio or video to analyze
        </span>
        <span className="text-xs text-muted">
          MP4, MOV, MKV, AVI, WebM, MP3, WAV, FLAC, AAC, OGG, M4A
        </span>

        <input
          className="hidden"
          disabled={disabled}
          type="file"
          aria-label="Upload audio or video file"
          accept="video/*,audio/*,.mp4,.mov,.mkv,.avi,.webm,.flv,.m4v,.wmv,.mp3,.wav,.flac,.aac,.ogg,.m4a,.wma,.opus"
          onChange={(event) => onChange(event.target.files?.[0] || null)}
        />

        <span className="mt-1 inline-flex rounded-pill bg-accent px-4 py-1.5 text-xs font-semibold text-white transition hover:bg-accent-hover">
          Browse files
        </span>
      </label>

      {/* Selected file chip */}
      {file ? (
        <div className="flex items-center gap-2 rounded-control bg-stone-100 px-4 py-2.5 text-sm">
          <svg className="h-4 w-4 shrink-0 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span className="min-w-0 flex-1 truncate font-medium text-ink">
            {file.name}
          </span>
          <span className="shrink-0 text-xs text-muted">
            {(file.size / 1024 / 1024).toFixed(2)} MB
          </span>
          <button
            type="button"
            className="ml-1 shrink-0 rounded-full p-0.5 text-muted transition hover:bg-stone-200 hover:text-ink"
            onClick={(e) => {
              e.preventDefault();
              onChange(null);
            }}
            aria-label="Remove selected file"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ) : (
        <p className="px-1 text-xs text-muted">No file selected</p>
      )}
    </div>
  );
}
