export type DetectionMode = "random_6s_crop" | "full_crop";

export interface ModelOption {
  name: string;
  display_name: string;
  relative_path: string;
  variant: string;
  variant_source: string;
  num_classes: number;
  probability_mode: string;
  active: boolean;
}

export interface ModelSummary {
  name: string;
  display_name: string;
  relative_path?: string;
  variant: string;
  variant_source?: string;
  num_classes: number;
  probability_mode: string;
}

export interface ModelsResponse {
  active_model: ModelSummary;
  models: ModelOption[];
}

export interface SelectModelResponse {
  message: string;
  selected_model: ModelOption;
}

export interface FullCropSummary {
  num_chunks: number;
  count_chunks_non_ai_lt_50: number;
  best_chunk_index: number;
  best_non_ai_prob: number;
  best_ai_prob: number;
  best_start_sec: number;
  best_end_sec: number;
  all_non_ai_probs: number[];
}

export interface DetectionResponse {
  filename: string;
  processing_status: string;
  detection_mode: DetectionMode;
  selected_model: {
    name: string;
    display_name: string;
    variant: string;
    num_classes: number;
    probability_mode: string;
  };
  pred_idx: number;
  label: "ai" | "human";
  ai_generated_probability: number;
  ai_generated_probability_score: number;
  human_probability_score: number;
  ai_music_detected: boolean;
  scores: {
    human: number;
    ai: number;
  };
  logits: number[];
  full_crop_summary?: FullCropSummary;
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T;
  }

  let message = `Request failed with status ${response.status}`;
  try {
    const errorData = (await response.json()) as { detail?: string };
    if (errorData.detail) {
      message = errorData.detail;
    }
  } catch {
    // Ignore malformed error payloads and keep the fallback message.
  }
  throw new Error(message);
}

export async function fetchModels(): Promise<ModelsResponse> {
  const response = await fetch(`${API_BASE_URL}/ai/models`);
  return parseResponse<ModelsResponse>(response);
}

export async function selectModel(modelName: string): Promise<SelectModelResponse> {
  const response = await fetch(`${API_BASE_URL}/ai/models/select`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model_name: modelName }),
  });
  return parseResponse<SelectModelResponse>(response);
}

export async function analyzeVideo(
  file: File,
  mode: DetectionMode,
): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const endpoint =
    mode === "full_crop" ? "/ai/detect/full-crop" : "/ai/detect/random-crop";

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body: formData,
  });
  return parseResponse<DetectionResponse>(response);
}
