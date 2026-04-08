import { proxyMultipartRequest } from "@/lib/server-api";

export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  return proxyMultipartRequest(request, "/detect-faces");
}
