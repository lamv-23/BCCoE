"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import ChatWindow from "@/components/ChatWindow";
import InputBar from "@/components/InputBar";
import UpdateBanner from "@/components/UpdateBanner";
import { Message, streamChat } from "@/lib/api";

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingContent, setStreamingContent] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [model, setModel] = useState("gpt-4o-mini");
  const [maxChunks, setMaxChunks] = useState(18);

  const sendMessage = async (content: string) => {
    if (isStreaming) return;

    const userMsg: Message = { role: "user", content };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setIsStreaming(true);
    setStreamingContent("");

    let accumulated = "";
    try {
      for await (const { event, data } of streamChat(newMessages, model, maxChunks)) {
        if (event === "token") {
          accumulated += data;
          setStreamingContent(accumulated);
        } else if (event === "error") {
          accumulated = data;
          setStreamingContent(accumulated);
          break;
        } else if (event === "done") {
          break;
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Connection error";
      accumulated = `Sorry, something went wrong: ${msg}`;
      setStreamingContent(accumulated);
    } finally {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: accumulated },
      ]);
      setStreamingContent("");
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-100 overflow-hidden">
      <Sidebar
        model={model}
        onModelChange={setModel}
        maxChunks={maxChunks}
        onMaxChunksChange={setMaxChunks}
        onClearChat={() => setMessages([])}
      />

      <div className="flex flex-col flex-1 min-w-0">
        <UpdateBanner />

        <div className="flex-1 overflow-y-auto">
          <ChatWindow
            messages={messages}
            streamingContent={streamingContent}
            isStreaming={isStreaming}
            onSelectQuestion={sendMessage}
          />
        </div>

        <div className="px-4 py-4 bg-slate-100 border-t border-slate-200">
          <div className="max-w-3xl mx-auto">
            <InputBar onSend={sendMessage} disabled={isStreaming} />
            <p className="text-center text-[10px] text-slate-400 mt-2">
              Responses are based on official CBA guides. Always verify with source documents.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
