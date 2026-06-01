import { Box, Paper, Typography } from "@mui/material";
import type { ChatMessage as Message } from "../types/chat";

interface Props {
    message: Message;
}

function LoadingDots() {
    return (
        <span>
            Thinking
            <span className="dot">.</span>
            <span className="dot">.</span>
            <span className="dot">.</span>

            <style>
                {`
          .dot {
            animation: blink 1.4s infinite both;
          }

          .dot:nth-child(2) {
            animation-delay: 0.2s;
          }

          .dot:nth-child(3) {
            animation-delay: 0.4s;
          }

          @keyframes blink {
            0% { opacity: 0.2; }
            20% { opacity: 1; }
            100% { opacity: 0.2; }
          }
        `}
            </style>
        </span>
    );
}

export default function ChatMessage({ message }: Props) {
    const isUser = message.role === "user";

    const renderContent = () => {
        if (message.loading) {
            // if backend is already updating text → show it
            if (message.content && message.content.length > 0) {
                return message.content;
            }

            return <LoadingDots />;
        }

        return message.content;
    };

    return (
        <Box
            sx={{
                display: "flex",
                justifyContent: isUser ? "flex-end" : "flex-start",
                mb: 2,
            }}
        >
            <Paper
                elevation={2}
                sx={{
                    p: 2,
                    maxWidth: "75%",
                    bgcolor: isUser ? "#2563eb" : "#1e293b",
                    color: "white",
                    borderRadius: 3,
                }}
            >
                <Typography
                    variant="body2"
                    sx={{
                        whiteSpace: "pre-wrap",
                        lineHeight: 1.7,
                    }}
                >
                    {renderContent()}
                </Typography>
            </Paper>
        </Box>
    );
}