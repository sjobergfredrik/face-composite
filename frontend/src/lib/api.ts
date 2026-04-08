const API_URL = (process.env.NEXT_PUBLIC_API_URL || "/api").replace(/\/$/, "");

function getApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_URL}${normalizedPath}`;
}

async function getErrorMessage(
  res: Response,
  fallback: string
): Promise<string> {
  const contentType = res.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    const err = await res.json().catch(() => ({ detail: fallback }));
    return err.detail || fallback;
  }

  const text = await res.text().catch(() => "");
  return text || res.statusText || fallback;
}

function toFetchError(error: unknown): Error {
  if (error instanceof Error) {
    return new Error(
      `Could not reach the face-swap API. ${error.message || "Check that Docker is running and the backend is healthy."}`
    );
  }

  return new Error(
    "Could not reach the face-swap API. Check that Docker is running and the backend is healthy."
  );
}

export interface FaceBBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedFace {
  index: number;
  bbox: FaceBBox;
  confidence: number;
}

export interface DetectResponse {
  faces: DetectedFace[];
  image_width: number;
  image_height: number;
}

export async function detectFaces(image: File): Promise<DetectResponse> {
  const form = new FormData();
  form.append("image", image);

  let res: Response;
  try {
    res = await fetch(getApiUrl("/detect-faces"), {
      method: "POST",
      body: form,
    });
  } catch (error) {
    throw toFetchError(error);
  }

  if (!res.ok) {
    throw new Error(await getErrorMessage(res, "Face detection failed"));
  }

  return res.json();
}

export async function swapFace(
  scene: File,
  actor: File,
  targetFaceIndex: number,
  model: string = "hyperswap_1c_256"
): Promise<Blob> {
  const form = new FormData();
  form.append("scene", scene);
  form.append("actor", actor);
  form.append("target_face_index", String(targetFaceIndex));
  form.append("model", model);

  let res: Response;
  try {
    res = await fetch(getApiUrl("/swap"), {
      method: "POST",
      body: form,
    });
  } catch (error) {
    throw toFetchError(error);
  }

  if (!res.ok) {
    throw new Error(await getErrorMessage(res, "Face swap failed"));
  }

  return res.blob();
}
