import { Box, Paper, Typography } from "@mui/material";
import type { ChatMessage as Message } from "../types/chat";

interface Props {
    message: Message;
}

export default function ChatMessage({ message }: Props) {
    const isUser = message.role === "user";

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
                    {message.loading ? "Thinking..." : message.content}
                </Typography>
            </Paper>
        </Box>
    );
}