const API_BASE = import.meta.env.VITE_API_BASE?.replace(/\/$/, "") || "http://localhost:8080";

export type PredictResponse = {
  class: string;
  probabilities: [number, number];
};

export async function health(): Promise<string> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.text();
}

export async function predict(file: File): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Predict failed: ${res.status}`);
  }
  return res.json();
}

export function getApiBase() {
  return API_BASE;
}
