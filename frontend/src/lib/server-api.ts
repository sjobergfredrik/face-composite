const INTERNAL_API_URL = (
  process.env.INTERNAL_API_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://backend:8000"
).replace(/\/$/, "");

function getBackendUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${INTERNAL_API_URL}${normalizedPath}`;
}

export async function proxyMultipartRequest(
  request: Request,
  path: string
): Promise<Response> {
  try {
    const formData = await request.formData();
    const upstream = await fetch(getBackendUrl(path), {
      method: "POST",
      body: formData,
      cache: "no-store",
    });

    const headers = new Headers();
    const contentType = upstream.headers.get("content-type");

    if (contentType) {
      headers.set("content-type", contentType);
    }

    return new Response(upstream.body, {
      status: upstream.status,
      headers,
    });
  } catch (error) {
    const detail =
      error instanceof Error
        ? error.message
        : "Unknown error contacting backend.";

    return Response.json(
      {
        detail: `Backend request failed: ${detail}`,
      },
      { status: 502 }
    );
  }
}
