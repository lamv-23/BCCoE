"use client";

import { useEffect, useState } from "react";
import { X, RefreshCw } from "lucide-react";
import { acknowledgeUpdate, getDocumentStatus, DocumentStatus } from "@/lib/api";

export default function UpdateBanner() {
  const [status, setStatus] = useState<DocumentStatus | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const s = await getDocumentStatus();
        setStatus(s);
      } catch {
        // silently ignore — backend may be starting up
      }
    };
    poll();
    const id = setInterval(poll, 60_000);
    return () => clearInterval(id);
  }, []);

  if (!status?.update_available) return null;

  const names = status.recently_updated_docs.join(", ");

  const dismiss = async () => {
    await acknowledgeUpdate();
    setStatus((prev) => prev ? { ...prev, update_available: false } : prev);
  };

  return (
    <div className="flex items-center gap-3 bg-emerald-600 text-white px-4 py-2 text-sm">
      <RefreshCw className="w-4 h-4 shrink-0" />
      <span className="flex-1">
        <strong>Documents updated</strong> — knowledge base rebuilt
        {names ? ` with: ${names}` : ""}.
      </span>
      <button onClick={dismiss} aria-label="Dismiss" className="hover:opacity-70 transition-opacity">
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}
