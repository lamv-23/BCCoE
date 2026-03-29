"use client";

import { useEffect, useState } from "react";
import { RefreshCw, ExternalLink, ChevronDown, ChevronUp, Settings } from "lucide-react";
import { getDocuments, refreshDocuments, DocumentInfo, RefreshResult } from "@/lib/api";

interface Props {
  model: string;
  onModelChange: (m: string) => void;
  maxChunks: number;
  onMaxChunksChange: (n: number) => void;
  onClearChat: () => void;
}

const MODELS = [
  { label: "Claude Sonnet 4.6 (Recommended)", value: "claude-sonnet-4-6" },
  { label: "Claude Opus 4.6", value: "claude-opus-4-6" },
  { label: "Claude Haiku 4.5", value: "claude-haiku-4-5-20251001" },
];

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" });
}

export default function Sidebar({
  model,
  onModelChange,
  maxChunks,
  onMaxChunksChange,
  onClearChat,
}: Props) {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshResult, setRefreshResult] = useState<RefreshResult | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(true);

  const loadDocs = async () => {
    try {
      setDocs(await getDocuments());
    } catch {
      /* backend may still be starting */
    }
  };

  useEffect(() => {
    loadDocs();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    setRefreshResult(null);
    try {
      const result = await refreshDocuments();
      setRefreshResult(result);
      await loadDocs();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Refresh failed";
      setRefreshResult({ updated: [], rebuilt: false, message: msg });
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <aside className="w-72 shrink-0 flex flex-col bg-slate-50 border-r border-slate-200 overflow-y-auto">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-200">
        <h1 className="font-bold text-slate-800 text-base leading-tight">
          BCCoE CBA Guide Assistant
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Powered by official Australian CBA guidelines
        </p>
      </div>

      {/* Settings section */}
      <div className="border-b border-slate-200">
        <button
          className="w-full flex items-center justify-between px-5 py-3 text-sm font-medium text-slate-700 hover:bg-slate-100 transition-colors"
          onClick={() => setSettingsOpen((v) => !v)}
        >
          <span className="flex items-center gap-2">
            <Settings className="w-4 h-4" /> Settings
          </span>
          {settingsOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {settingsOpen && (
          <div className="px-5 pb-4 space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">AI Model</label>
              <select
                value={model}
                onChange={(e) => onModelChange(e.target.value)}
                className="w-full text-sm border border-slate-200 rounded-lg px-3 py-1.5 bg-white text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-400"
              >
                {MODELS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">
                Context chunks: <span className="font-bold text-slate-800">{maxChunks}</span>
              </label>
              <input
                type="range"
                min={10}
                max={25}
                value={maxChunks}
                onChange={(e) => onMaxChunksChange(Number(e.target.value))}
                className="w-full accent-blue-600"
              />
              <div className="flex justify-between text-xs text-slate-400 mt-0.5">
                <span>10</span><span>25</span>
              </div>
            </div>

            <button
              onClick={onClearChat}
              className="w-full text-xs text-slate-500 border border-slate-200 rounded-lg py-1.5 hover:bg-slate-100 transition-colors"
            >
              Clear chat history
            </button>
          </div>
        )}
      </div>

      {/* Documents section */}
      <div className="flex-1 px-5 py-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Documents
          </h2>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            title="Check for document updates"
            className="p-1 rounded-md hover:bg-slate-200 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-slate-500 ${refreshing ? "animate-spin" : ""}`} />
          </button>
        </div>

        {refreshResult && (
          <p className={`text-xs mb-3 px-2 py-1.5 rounded-md ${
            refreshResult.updated.length > 0
              ? "bg-emerald-50 text-emerald-700"
              : "bg-slate-100 text-slate-600"
          }`}>
            {refreshResult.message}
          </p>
        )}

        <ul className="space-y-3">
          {docs.map((doc) => (
            <li key={doc.doc_id} className="bg-white border border-slate-200 rounded-xl p-3 text-xs">
              <p className="font-medium text-slate-800 mb-1 leading-snug">{doc.display_name}</p>
              {doc.version_label && (
                <span className="inline-block bg-blue-50 text-blue-700 text-[10px] font-medium px-1.5 py-0.5 rounded-md mb-1">
                  v{doc.version_label}
                </span>
              )}
              <p className="text-slate-400">Updated: {formatDate(doc.last_updated)}</p>
              <a
                href={doc.source_page_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-blue-500 hover:text-blue-700 mt-1 transition-colors"
              >
                Source <ExternalLink className="w-3 h-3" />
              </a>
            </li>
          ))}
        </ul>
      </div>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-slate-200 text-[10px] text-slate-400">
        Documents checked weekly. Tap <RefreshCw className="inline w-3 h-3" /> to check now.
      </div>
    </aside>
  );
}
