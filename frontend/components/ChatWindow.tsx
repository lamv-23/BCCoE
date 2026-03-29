"use client";

import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import { Message } from "@/lib/api";

interface Props {
  messages: Message[];
  streamingContent: string;
  isStreaming: boolean;
}

const SAMPLE_QUESTIONS = [
  "What is the standard discount rate for CBA?",
  "How do I calculate Net Present Value?",
  "What are the key steps in conducting a CBA?",
  "How should I handle uncertainty in my analysis?",
  "What costs should be included in a social CBA?",
  "What is sensitivity analysis?",
];

interface EmptyStateProps {
  onSelectQuestion: (q: string) => void;
}

function EmptyState({ onSelectQuestion }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 text-slate-500 px-6">
      <div className="text-center">
        <div className="text-4xl mb-3">📊</div>
        <h2 className="text-xl font-semibold text-slate-700 mb-1">CBA Guide Assistant</h2>
        <p className="text-sm">
          Ask anything about Australian government cost-benefit analysis guidelines.
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
        {SAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => onSelectQuestion(q)}
            className="text-left text-xs bg-white border border-slate-200 rounded-xl px-3 py-2.5
                       hover:border-blue-400 hover:bg-blue-50 transition-colors text-slate-600"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

interface ChatWindowProps extends Props {
  onSelectQuestion: (q: string) => void;
}

export default function ChatWindow({
  messages,
  streamingContent,
  isStreaming,
  onSelectQuestion,
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  if (messages.length === 0 && !isStreaming) {
    return <EmptyState onSelectQuestion={onSelectQuestion} />;
  }

  return (
    <div className="flex flex-col gap-5 px-4 py-6">
      {messages.map((msg, idx) => (
        <MessageBubble key={idx} message={msg} />
      ))}

      {isStreaming && streamingContent && (
        <MessageBubble
          message={{ role: "assistant", content: streamingContent }}
          isStreaming
        />
      )}

      <div ref={bottomRef} />
    </div>
  );
}
