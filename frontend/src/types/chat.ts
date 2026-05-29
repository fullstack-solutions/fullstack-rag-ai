export interface Source {
    file: string;
    page: number;
    quote: string;
}

export interface ChatMessage {
    role: "user" | "assistant";
    content: string;
    sources?: Source[];

    loading?: boolean;
    error?: boolean;
}