import { useState, useEffect, useRef } from "react";
import { Box } from "@mui/material";

import Sidebar from "./components/Sidebar";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";

import { askQuestion } from "./services/api";
import type { ChatMessage as Message } from "./types/chat";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);


  async function askQuestionWithRetry(
    question: string,
    retries = 2
  ): Promise<any> {
    let lastError: any;

    for (let i = 0; i <= retries; i++) {
      try {
        return await askQuestion(question);
      } catch (err) {
        lastError = err;
        await new Promise((r) => setTimeout(r, 1000 * (i + 1)));
      }
    }

    throw lastError;
  }

  const stages = [
    "Finding relevant documents...",
    "Analyzing context...",
    "Understanding query...",
    "Generating response...",
    "Finalizing answer...",
  ];

  function startProgress(update: (msg: string) => void) {
    let i = 0;

    const interval = setInterval(() => {
      update(stages[i]);
      i++;

      if (i >= stages.length) {
        clearInterval(interval);
      }
    }, 2000);

    return interval;
  }

  // -------------------------
  // update last assistant message
  // -------------------------
  function updateLastAssistantMessage(update: Partial<Message>) {
    setMessages((prev) => {
      const copy = [...prev];
      copy[copy.length - 1] = {
        ...copy[copy.length - 1],
        ...update,
      };
      return copy;
    });
  }


  async function handleSend(question: string) {
    const cleaned = question.trim();
    if (!cleaned) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: cleaned },
    ]);

    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "Finding relevant documents...",
        loading: true,
      },
    ]);

    setLoading(true);

    // progressive UX
    const interval = startProgress((msg) => {
      updateLastAssistantMessage({ content: msg });
    });

    // slow fallback message
    const slowTimer = setTimeout(() => {
      updateLastAssistantMessage({
        content:
          "Still processing large dataset... please wait a moment.",
      });
    }, 7000);

    try {
      const data = await askQuestionWithRetry(cleaned);

      clearInterval(interval);
      clearTimeout(slowTimer);

      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        };
        return copy;
      });
    } catch (err) {
      clearInterval(interval);
      clearTimeout(slowTimer);

      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = {
          role: "assistant",
          content:
            "Unable to process request right now. Please try again.",
          error: true,
        };
        return copy;
      });
    }

    setLoading(false);
  }

  return (
    <Box sx={{ display: "flex", height: "100vh", bgcolor: "#0b1120" }}>
      <Sidebar />

      <Box sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <Box sx={{ flex: 1, overflowY: "auto", p: 3 }}>
          {messages.map((m, i) => (
            <ChatMessage key={i} message={m} />
          ))}
          <div ref={messagesEndRef} />
        </Box>

        <ChatInput onSend={handleSend} loading={loading} />
      </Box>
    </Box>
  );
}