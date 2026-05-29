export async function askQuestion(question: string) {
    const res = await fetch("/api/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
    });

    return await res.json();
}
