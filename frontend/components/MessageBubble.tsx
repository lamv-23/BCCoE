"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Message } from "@/lib/api";

interface Props {
  message: Message;
  isStreaming?: boolean;
}

export default function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-white text-xs font-bold
          ${isUser ? "bg-blue-600" : "bg-slate-700"}`}
      >
        {isUser ? "You" : "AI"}
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed
          ${isUser
            ? "bg-blue-600 text-white rounded-tr-sm"
            : "bg-white border border-slate-200 text-slate-800 rounded-tl-sm shadow-sm"
          }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-1">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
            {isStreaming && (
              <span className="inline-block w-2 h-4 ml-0.5 bg-slate-400 animate-pulse rounded-sm" />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
