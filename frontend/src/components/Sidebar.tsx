import { useState } from "react";
import {
    Box,
    Typography,
    Button,
    Paper,
    Dialog,
    DialogTitle,
    DialogContent,
    TextField,
    DialogActions,
    Alert,
    CircularProgress,
} from "@mui/material";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import { uploadDocuments } from "../services/api";

export default function Sidebar() {
    const [showModal, setShowModal] = useState(false);
    const [dataSource, setDataSource] = useState("");
    const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);
    const [statusMessage, setStatusMessage] = useState("");
    const [statusType, setStatusType] = useState<"success" | "error" | "info">(
        "info"
    );

    function openModal() {
        setShowModal(true);
        setStatusMessage("");
    }

    function closeModal() {
        if (uploading) {
            alert(
                "Upload is still in progress. If you close now, you will not know the final status of the request."
            );
            return;
        }

        setShowModal(false);
        setSelectedFiles([]);
        setDataSource("");
        setStatusMessage("");
    }

    async function handleUpload() {
        if (!selectedFiles.length || !dataSource.trim()) {
            setStatusType("error");
            setStatusMessage("Please select files and enter data source path.");
            return;
        }

        try {
            setUploading(true);
            setStatusType("info");
            setStatusMessage("Upload in progress... Please wait.");

            const result = await uploadDocuments(selectedFiles, dataSource);

            if (result.success) {
                setStatusType("success");
                setStatusMessage(result.message || "Upload completed successfully.");
            } else {
                setStatusType("error");
                setStatusMessage(result.error || "Upload failed.");
            }
        } catch (err: any) {
            setStatusType("error");
            setStatusMessage(err.message || "Upload failed.");
        } finally {
            setUploading(false);
        }
    }

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

                    {/* <Paper
                        elevation={4}
                        sx={{
                            mt: 4,
                            p: 2,
                            bgcolor: "#1e293b",
                            borderRadius: 3,
                        }}
                    >
                        <Button
                            fullWidth
                            startIcon={<UploadFileIcon />}
                            variant="contained"
                            onClick={openModal}
                        >
                            Upload Documents
                        </Button>
                    </Paper> */}
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

            {/* <Dialog
                open={showModal}
                onClose={closeModal}
                fullWidth
                maxWidth="sm"
            > */}
            {/* <DialogTitle>Upload Documents</DialogTitle>

                <DialogContent>
                    <TextField
                        fullWidth
                        label="Data Source Path"
                        margin="normal"
                        value={dataSource}
                        onChange={(e) => setDataSource(e.target.value)}
                        disabled={uploading}
                    />

                    <input
                        type="file"
                        multiple
                        disabled={uploading}
                        onChange={(e) =>
                            setSelectedFiles(Array.from(e.target.files || []))
                        }
                    />

                    {selectedFiles.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                            {selectedFiles.map((file, idx) => (
                                <Typography key={idx} variant="body2">
                                    • {file.name}
                                </Typography>
                            ))}
                        </Box>
                    )}

                    {statusMessage && (
                        <Alert severity={statusType} sx={{ mt: 3 }}>
                            {statusMessage}
                        </Alert>
                    )}
                </DialogContent>

                <DialogActions>
                    <Button onClick={closeModal}>
                        Close
                    </Button>

                    <Button
                        variant="contained"
                        onClick={handleUpload}
                        disabled={uploading}
                        startIcon={
                            uploading ? <CircularProgress size={18} color="inherit" /> : null
                        }
                    >
                        {uploading ? "Uploading..." : "Upload"}
                    </Button>
                </DialogActions>
            </Dialog> */}
        </>
    );
}