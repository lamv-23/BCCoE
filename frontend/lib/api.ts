const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface DocumentInfo {
  doc_id: string;
  display_name: string;
  source_page_url: string;
  local_path: string;
  version_label: string | null;
  last_updated: string | null;
  last_checked: string | null;
}

export interface RefreshResult {
  updated: string[];
  rebuilt: boolean;
  message: string;
}

export interface DocumentStatus {
  update_available: boolean;
  recently_updated_docs: string[];
  last_check: string | null;
  next_check: string | null;
  chunk_count: number;
  last_built: string | null;
}

export async function* streamChat(
  messages: Message[],
  model: string,
  maxChunks: number
): AsyncGenerator<{ event: string; data: string }> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, model, max_chunks: maxChunks }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.statusText}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    let event = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        event = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = line.slice(6).replace(/\\n/g, "\n");
        yield { event, data };
        event = "";
      }
    }
  }
}

export async function getDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${API_BASE}/api/documents`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function refreshDocuments(): Promise<RefreshResult> {
  const res = await fetch(`${API_BASE}/api/documents/refresh`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Refresh failed");
  return res.json();
}

export async function getDocumentStatus(): Promise<DocumentStatus> {
  const res = await fetch(`${API_BASE}/api/documents/status`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to fetch status");
  return res.json();
}

export async function acknowledgeUpdate(): Promise<void> {
  await fetch(`${API_BASE}/api/documents/status/acknowledge`, {
    method: "POST",
  });
}
