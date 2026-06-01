export async function askQuestion(question: string) {
    const controller = new AbortController();

    const timeout = setTimeout(() => {
        controller.abort();
    }, 60000);

    try {
        const res = await fetch("/api/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question }),
            signal: controller.signal,
        });

        clearTimeout(timeout);

        if (!res.ok) {
            throw new Error("API error");
        }

        return await res.json();
    } catch (err) {
        clearTimeout(timeout);
        throw err;
    }
}