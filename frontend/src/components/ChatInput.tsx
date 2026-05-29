import { useState } from "react";
import { Box, TextField, IconButton, Paper } from "@mui/material";
import SendIcon from "@mui/icons-material/Send";

interface Props {
    onSend: (message: string) => void;
    loading?: boolean;
}

export default function ChatInput({ onSend, loading }: Props) {
    const [message, setMessage] = useState("");

    function handleSend() {
        if (!message.trim()) return;
        onSend(message);
        setMessage("");
    }

    return (
        <Paper
            elevation={3}
            sx={{
                p: 2,
                borderTop: "1px solid #1f2937",
                bgcolor: "#0f172a",
            }}
        >
            <Box sx={{ display: "flex", gap: 2 }}>
                <TextField
                    fullWidth
                    multiline
                    maxRows={4}
                    placeholder="Message your documents..."
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    sx={{
                        "& .MuiOutlinedInput-root": {
                            color: "white",
                            bgcolor: "#1e293b",
                        },
                    }}
                />

                <IconButton
                    color="primary"
                    onClick={handleSend}
                    disabled={loading}
                >
                    <SendIcon />
                </IconButton>
            </Box>
        </Paper>
    );
}