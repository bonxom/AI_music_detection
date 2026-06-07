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
    <section className="rounded-[32px] border border-white/70 bg-white/80 p-6 shadow-panel backdrop-blur">
      <p className="text-sm uppercase tracking-[0.2em] text-stone-500">Preview</p>
      <p className="mt-3 text-sm font-medium text-ink">{file.name}</p>

      {previewType === "audio" ? (
        <audio className="mt-4 w-full" controls preload="metadata" src={previewUrl}>
          Your browser does not support audio playback.
        </audio>
      ) : (
        <video
          className="mt-4 max-h-80 w-full rounded-[24px] bg-stone-950"
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
