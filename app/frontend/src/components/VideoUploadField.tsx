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
    <div className="space-y-3">
      <label className="text-sm font-semibold uppercase tracking-[0.18em] text-stone-600">
        Media File
      </label>

      <label className="flex cursor-pointer flex-col gap-3 rounded-[28px] border border-dashed border-orange-300 bg-white/75 p-6 text-left shadow-sm transition hover:border-accent">
        <span className="text-lg font-semibold text-ink">Choose audio or video to analyze</span>
        <span className="text-sm text-stone-600">
          Supported inputs include video files such as MP4, MOV, MKV, AVI, and WebM, plus audio
          files such as MP3, WAV, FLAC, AAC, OGG, and M4A.
        </span>
        <input
          className="hidden"
          disabled={disabled}
          type="file"
          accept="video/*,audio/*,.mp4,.mov,.mkv,.avi,.webm,.flv,.m4v,.wmv,.mp3,.wav,.flac,.aac,.ogg,.m4a,.wma,.opus"
          onChange={(event) => onChange(event.target.files?.[0] || null)}
        />
        <span className="inline-flex w-fit rounded-full bg-accent px-4 py-2 text-sm font-semibold text-white">
          Browse media
        </span>
      </label>

      <div className="rounded-2xl bg-stone-900 px-4 py-3 text-sm text-stone-100">
        {file ? `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)` : "No file selected"}
      </div>
    </div>
  );
}
