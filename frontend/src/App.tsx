import { useState, useEffect, useRef } from "react";
import { Box } from "@mui/material";

import Sidebar from "./components/Sidebar";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";

import { askQuestion } from "./services/api";
import type { ChatMessage as Message } from "./types/chat";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(question: string) {
    if (!question.trim()) return;

    const userMessage: Message = {
      role: "user",
      content: question,
    };

    setMessages((prev) => [...prev, userMessage]);

    const loadingMessage: Message = {
      role: "assistant",
      content: "",
      loading: true,
    };

    setMessages((prev) => [...prev, loadingMessage]);
    setLoading(true);

    try {
      const data = await askQuestion(question);

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        };
        return updated;
      });
    } catch {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: "Something went wrong.",
          error: true,
        };
        return updated;
      });
    }

    setLoading(false);
  }

  return (
    <Box sx={{ display: "flex", height: "100vh", bgcolor: "#0b1120" }}>
      <Sidebar />

      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Box
          sx={{
            flex: 1,
            overflowY: "auto",
            p: 3,
          }}
        >
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

export default App;