import { useEffect, useState } from "react";

interface MediaPreviewProps {
  file: File | null;
}

const AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"];
const VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v", ".wmv"];

function hasExtension(file: File, extensions: string[]) {
  const fileName = file.name.toLowerCase();
  return extensions.some((extension) => fileName.endsWith(extension));
}

function getPreviewType(file: File) {
  if (file.type.startsWith("audio/") || hasExtension(file, AUDIO_EXTENSIONS)) {
    return "audio";
  }
  if (file.type.startsWith("video/") || hasExtension(file, VIDEO_EXTENSIONS)) {
    return "video";
  }
  return null;
}

export function MediaPreview({ file }: MediaPreviewProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  if (!file || !previewUrl) {
    return null;
  }

  const previewType = getPreviewType(file);
  if (!previewType) {
    return null;
  }

  return (
    <section className="rounded-card border border-border bg-white p-5 shadow-card">
      <div className="flex items-center justify-between">
        <p className="section-label">Preview</p>
        <p className="truncate text-xs text-muted max-w-[200px]">{file.name}</p>
      </div>

      {previewType === "audio" ? (
        <audio className="mt-4 w-full" controls preload="metadata" src={previewUrl}>
          Your browser does not support audio playback.
        </audio>
      ) : (
        <video
          className="mt-4 w-full max-h-72 rounded-control bg-stone-950 object-contain"
          controls
          preload="metadata"
          src={previewUrl}
        >
          Your browser does not support video playback.
        </video>
      )}
    </section>
  );
}
