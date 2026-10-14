
import { useState } from "react";
import { Bot, MessageCircle, Send, X } from "lucide-react";

const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export default function Chatbot({ analysis, details, explanation }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function sendMessage(event) {
    event.preventDefault();

    const message = input.trim();
    if (!message || loading) return;

    const previousMessages = messages;
    setInput("");
    setError("");
    setMessages((old) => [...old, { role: "user", content: message }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          history: previousMessages.slice(-10),
          analysis_context: analysis
            ? { analysis, details, explanation }
            : null,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to get a response.");
      }

      setMessages((old) => [
        ...old,
        { role: "assistant", content: data.answer },
      ]);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot reach the backend. Check your connection."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.root}>
      {open && (
        <section style={styles.panel}>
          <header style={styles.header}>
            <div style={styles.heading}>
              <Bot size={22} />
              <div>
                <strong>DeepShield Chat</strong>
                <div style={styles.subtitle}>Your AI assistant</div>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setOpen(false)}
              style={styles.iconButton}
              aria-label="Close chatbot"
            >
              <X size={20} />
            </button>
          </header>

          <div style={styles.messages} aria-live="polite">
            {messages.length === 0 && (
              <div style={styles.welcome}>
                <Bot size={32} />
                <strong>Hey! How can I help?</strong>
                <p>
                  Ask about deepfakes, detection results, confidence scores,
                  or media verification.
                </p>
              </div>
            )}

            {messages.map((item, index) => (
              <div
                key={`${index}-${item.role}`}
                style={{
                  ...styles.bubble,
                  ...(item.role === "user"
                    ? styles.userBubble
                    : styles.assistantBubble),
                }}
              >
                {item.content}
              </div>
            ))}

            {loading && (
              <div style={{ ...styles.bubble, ...styles.assistantBubble }}>
                Thinking...
              </div>
            )}

            {error && <p style={styles.error}>{error}</p>}
          </div>

          <form onSubmit={sendMessage} style={styles.form}>
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask DeepShield anything..."
              maxLength={2000}
              disabled={loading}
              aria-label="Chat message"
              style={styles.input}
            />

            <button
              type="submit"
              disabled={loading || !input.trim()}
              style={styles.sendButton}
              aria-label="Send message"
            >
              <Send size={18} />
            </button>
          </form>

          <div style={styles.disclaimer}>
            AI responses can be incorrect. Verify important claims.
          </div>
        </section>
      )}

      <button
        type="button"
        style={styles.launcher}
        onClick={() => setOpen((value) => !value)}
        aria-label={open ? "Close chatbot" : "Open chatbot"}
      >
        {open ? <X size={22} /> : <MessageCircle size={22} />}
        {!open && <span>Ask DeepShield</span>}
      </button>
    </div>
  );
}

const styles = {
  root: {
    position: "fixed",
    right: 24,
    bottom: 24,
    zIndex: 9999,
    fontFamily: "inherit",
  },
  panel: {
    width: "min(370px, calc(100vw - 32px))",
    height: "min(520px, calc(100dvh - 110px))",
    marginBottom: 12,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    color: "#e8edff",
    background: "#11182b",
    border: "1px solid rgba(145,162,255,.3)",
    borderRadius: 18,
    boxShadow: "0 20px 60px rgba(0,0,0,.4)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    padding: 16,
    background: "#19223b",
    borderBottom: "1px solid rgba(145,162,255,.2)",
  },
  heading: { display: "flex", alignItems: "center", gap: 10 },
  subtitle: { marginTop: 4, color: "#aab7d7", fontSize: 12 },
  iconButton: {
    display: "grid",
    placeItems: "center",
    color: "#e8edff",
    background: "transparent",
    border: 0,
    cursor: "pointer",
    padding: 5,
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: 12,
    padding: 15,
  },
  welcome: {
    margin: "auto 0",
    textAlign: "center",
    color: "#cbd5f3",
    lineHeight: 1.6,
    fontSize: 13,
  },
  bubble: {
    maxWidth: "88%",
    padding: "11px 13px",
    borderRadius: 13,
    whiteSpace: "pre-wrap",
    overflowWrap: "anywhere",
    fontSize: 13,
    lineHeight: 1.65,
  },
  userBubble: {
    alignSelf: "flex-end",
    background: "#697ce0",
    color: "#fff",
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    alignSelf: "flex-start",
    background: "#202b46",
    color: "#e5eaff",
    borderBottomLeftRadius: 4,
  },
  error: { color: "#ffaaaa", fontSize: 12 },
  form: {
    display: "flex",
    gap: 8,
    padding: 12,
    borderTop: "1px solid rgba(145,162,255,.2)",
  },
  input: {
    flex: 1,
    minWidth: 0,
    padding: "11px 12px",
    color: "#e8edff",
    background: "#202b46",
    border: "1px solid rgba(145,162,255,.25)",
    borderRadius: 10,
    outline: "none",
  },
  sendButton: {
    display: "grid",
    placeItems: "center",
    padding: "0 13px",
    color: "#0b1020",
    background: "#b9c3ff",
    border: 0,
    borderRadius: 10,
    cursor: "pointer",
  },
  disclaimer: {
    padding: "0 12px 12px",
    color: "#929fbd",
    fontSize: 10,
    textAlign: "center",
  },
  launcher: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 9,
    marginLeft: "auto",
    padding: "13px 17px",
    color: "#0b1020",
    background: "linear-gradient(110deg, #b9c3ff, #a5e6e0)",
    border: 0,
    borderRadius: 999,
    boxShadow: "0 8px 30px rgba(0,0,0,.3)",
    fontWeight: 700,
    cursor: "pointer",
  },
};