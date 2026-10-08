const BASE = "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number) {
    super(`HTTP ${status}`);
    this.status = status;
  }
}

export type Transcription = {
  text: string;
  language_code: string;
  language_of_response: string;
  image_resolution_type: string;
};

async function post<T>(path: string, body: FormData): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { method: "POST", body });
  if (!r.ok) throw new ApiError(r.status);
  return (await r.json()) as T;
}

export function transcribe(blob: Blob) {
  const ext = blob.type.includes("mp4") ? "mp4" : "webm";
  const fd = new FormData();
  fd.append("audio", blob, `question.${ext}`);
  return post<Transcription>("/transcribe", fd);
}

// Mirrors the two payloads in app.py: first turn (with image) and follow-up.
export function askAI(p: {
  sessionId: string | null;
  image: File | null;
  t: Transcription;
}) {
  const firstTurn = !p.sessionId;
  const fd = new FormData();
  if (p.sessionId) fd.append("session_id", p.sessionId);
  fd.append("image_resolution_type", p.t.image_resolution_type);
  fd.append("user_prompt", p.t.text);
  fd.append("language_of_response", p.t.language_of_response);
  fd.append("send_image", String(firstTurn));
  if (firstTurn && p.image) fd.append("file", p.image);
  return post<{ ai_response: string; session_id: string }>(
    "/get_ai_response",
    fd,
  );
}

export function resetSession(id: string) {
  const fd = new FormData();
  fd.append("session_id", id);
  navigator.sendBeacon(`${BASE}/reset_session`, fd); // best effort
}
