"use client";

import { useState, useRef, KeyboardEvent } from "react";
import { Send } from "lucide-react";

interface Props {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export default function InputBar({ onSend, disabled }: Props) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const onInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="flex items-end gap-2 bg-white border border-slate-200 rounded-2xl px-4 py-3 shadow-sm
                    focus-within:border-blue-400 focus-within:ring-1 focus-within:ring-blue-200 transition-all">
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        onInput={onInput}
        rows={1}
        disabled={disabled}
        placeholder="Ask about cost-benefit analysis…"
        className="flex-1 resize-none outline-none text-sm text-slate-800 placeholder:text-slate-400
                   bg-transparent leading-relaxed disabled:opacity-50"
      />
      <button
        onClick={submit}
        disabled={disabled || !text.trim()}
        className="shrink-0 w-8 h-8 rounded-xl bg-blue-600 text-white flex items-center justify-center
                   hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        aria-label="Send"
      >
        <Send className="w-4 h-4" />
      </button>
    </div>
  );
}
