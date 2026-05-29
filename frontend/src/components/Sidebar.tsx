import {
    Box,
    Typography,

} from "@mui/material";

export default function Sidebar() {

    return (
        <>
            <Box
                sx={{
                    width: 320,
                    bgcolor: "#111827",
                    color: "white",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    p: 3,
                    borderRight: "1px solid #1f2937",
                }}
            >
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 700 }}>
                        Sporveien RAG Assistant
                    </Typography>
                </Box>

                <Box>
                    <Typography variant="caption" sx={{ color: "#64748b" }}>
                        Powered by
                    </Typography>

                    <Typography variant="body2" sx={{ fontWeight: 700 }}>
                        Sporveien AI Team
                    </Typography>
                </Box>
            </Box>
        </>
    );
}