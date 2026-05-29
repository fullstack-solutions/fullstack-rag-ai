export async function askQuestion(question: string) {
    const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
    });

    return await res.json();
}

export async function uploadDocuments(
    files: File[],
    dataSourceRef: string
) {
    const formData = new FormData();

    formData.append("data_source_ref", dataSourceRef);

    for (const file of files) {
        formData.append("files", file);
    }

    try {
        const res = await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData,
        });

        const data = await res.json();

        if (!res.ok || !data.success) {
            throw new Error(data.error || "Upload failed");
        }

        return data;

    } catch (error: any) {
        return {
            success: false,
            error: error.message || "Network error",
        };
    }
}