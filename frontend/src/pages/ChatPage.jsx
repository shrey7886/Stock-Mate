import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Trash2,
  Bot,
  User,
  Sparkles,
  ArrowRight,
  Loader2,
  MessageCircle,
} from "lucide-react";
import { chat } from "../services/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import FoleoLogo from "../components/FoleoLogo";

const QUICK_PROMPTS = [
  "How's my portfolio health?",
  "What sectors am I overweight in?",
  "Should I rebalance?",
  "Explain LTCG tax for stocks",
  "Suggest SIP strategy for 10k/month",
];

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-5 py-3.5 chat-bubble-ai w-fit shadow-sm">
      <div className="typing-dot" style={{ animationDelay: "0s" }} />
      <div className="typing-dot" style={{ animationDelay: "0.15s" }} />
      <div className="typing-dot" style={{ animationDelay: "0.3s" }} />
    </div>
  );
}


function MessageBubble({ role, text, insights, healthScore, nextSteps, onNextStep }) {
  const isUser = role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div className={`flex items-start gap-4 max-w-[85%] md:max-w-[70%]`}>
        {!isUser && (
          <div className="mt-1 shrink-0">
            <FoleoLogo size="sm" />
          </div>
        )}
        <div className="space-y-3">
          <div
            className={`px-6 py-4 text-[15px] leading-relaxed shadow-sm ${
              isUser
                ? "chat-bubble-user text-[var(--color-text-primary)] whitespace-pre-wrap"
                : "chat-bubble-ai text-[var(--color-text-secondary)] font-light"
            }`}
          >
            {isUser ? text : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
                  ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3 last:mb-0 space-y-1 my-2" {...props} />,
                  ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-3 last:mb-0 space-y-1 my-2" {...props} />,
                  li: ({node, ...props}) => <li className="marker:text-[var(--color-text-muted)]" {...props} />,
                  strong: ({node, ...props}) => <strong className="font-semibold text-[var(--color-text-primary)]" {...props} />,
                  h1: ({node, ...props}) => <h1 className="text-xl font-bold mb-3 mt-5 text-[var(--color-text-primary)] first:mt-0" {...props} />,
                  h2: ({node, ...props}) => <h2 className="text-lg font-bold mb-2 mt-4 text-[var(--color-text-primary)] first:mt-0" {...props} />,
                  h3: ({node, ...props}) => <h3 className="text-base font-bold mb-2 mt-3 text-[var(--color-text-primary)] first:mt-0" {...props} />,
                  a: ({node, ...props}) => <a className="text-[var(--color-brand)] underline hover:opacity-80" target="_blank" rel="noopener noreferrer" {...props} />,
                  table: ({node, ...props}) => <div className="overflow-x-auto mb-3"><table className="min-w-full divide-y divide-[var(--color-border-subtle)] text-sm" {...props} /></div>,
                  th: ({node, ...props}) => <th className="px-3 py-2 text-left text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-wider bg-[var(--color-surface)]/50" {...props} />,
                  td: ({node, ...props}) => <td className="px-3 py-2 whitespace-nowrap border-b border-[var(--color-border-subtle)]/50" {...props} />,
                  blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-[var(--color-brand)]/50 pl-4 py-1 italic bg-[var(--color-brand)]/5 rounded-r-lg mb-3" {...props} />,
                  code: ({node, inline, className, children, ...props}) => {
                    const match = /language-(\w+)/.exec(className || "");
                    return !inline ? (
                      <div className="overflow-x-auto bg-[var(--color-surface)] border border-[var(--color-border)] rounded-xl my-3 scrollbar-thin">
                        <div className="px-3 py-1.5 bg-[var(--color-surface-overlay)] border-b border-[var(--color-border)] text-[11px] font-mono text-[var(--color-text-muted)] tracking-wider uppercase font-bold">
                          {match ? match[1] : "code"}
                        </div>
                        <pre className="p-3 text-sm font-mono text-[var(--color-text-secondary)] whitespace-pre">
                          <code className={className} {...props}>{children}</code>
                        </pre>
                      </div>
                    ) : (
                      <code className="bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)] px-1.5 py-0.5 rounded text-[13px] font-medium font-mono text-[var(--color-text-primary)]" {...props}>
                        {children}
                      </code>
                    )
                  }
                }}
              >
                {text}
              </ReactMarkdown>
            )}
          </div>

          {/* Inline health score */}
          {healthScore != null && (
            <div className="flex items-center gap-2.5 px-5 py-3 rounded-2xl bg-[var(--color-brand)]/10 border border-[var(--color-brand)]/20 w-fit cursor-default shadow-sm">
              <Sparkles size={13} className="text-[var(--color-brand)]" />
              <span className="text-[13px] text-[var(--color-brand)] font-bold uppercase tracking-wider">
                Portfolio Health: {healthScore}/100
              </span>
            </div>
          )}

          {/* Insights */}
          {insights && insights.length > 0 && (
            <div className="space-y-2 mt-2">
              {insights.map((ins, i) => (
                <div key={i} className="flex items-start gap-3 px-5 py-3.5 rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] shadow-sm">
                  <ArrowRight size={14} className="text-[var(--color-brand)] mt-0.5 shrink-0" />
                  <span className="text-[14px] font-medium text-[var(--color-text-secondary)] leading-relaxed">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ p: ({node, ...props}) => <span {...props} />, strong: ({node, ...props}) => <strong className="font-semibold text-[var(--color-text-primary)]" {...props} /> }}>
                      {ins}
                    </ReactMarkdown>
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Next steps */}
          {nextSteps && nextSteps.length > 0 && (
            <div className="flex flex-wrap gap-2.5 mt-2">
              {nextSteps.map((step, i) => (
                <button
                  key={i}
                  onClick={() => onNextStep(step)}
                  className="px-4 py-2 rounded-full bg-[var(--color-surface-overlay)] border border-[var(--color-border)] text-[13px] font-medium text-[var(--color-text-secondary)] shadow-sm hover:text-[var(--color-text-primary)] hover:border-[var(--color-brand)]/50 hover:bg-[var(--color-brand)]/5 transition-all duration-300"
                >
                  {step}
                </button>
              ))}
            </div>
          )}
        </div>
        {isUser && (
          <div className="mt-1 p-2.5 rounded-2xl bg-white/10 dark:bg-white/5 border border-black/10 dark:border-white/10 shrink-0 shadow-sm">
            <User size={16} className="text-[var(--color-text-muted)]" />
          </div>
        )}
      </div>
    </motion.div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [sessionId] = useState(() => crypto.randomUUID());
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, sending, scrollToBottom]);

  const sendMessage = async (text) => {
    if (!text.trim() || sending) return;
    const userMsg = { role: "user", text: text.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);

    try {
      const data = await chat.send(text.trim(), sessionId);
      const aiMsg = {
        role: "ai",
        text: data.answer || data.response || "I couldn't process that. Try again!",
        insights: data.insights,
        healthScore: data.health?.score ?? data.health_score,
        nextSteps: data.next_steps,
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: `Oops, something went wrong: ${err.message}. Try again? 🤷`,
        },
      ]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleClear = async () => {
    try {
      await chat.clearSession();
    } catch {
      // best-effort
    }
    setMessages([]);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden bg-[var(--color-surface)]">
      {/* Header */}
      <div className="flex items-center justify-between px-8 py-5 border-b border-[var(--color-border)] bg-[var(--color-surface)]/80 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center gap-4">
          <div className="p-2.5 rounded-2xl bg-[var(--color-brand)]/10 border border-[var(--color-brand)]/20">
            <MessageCircle size={18} className="text-[var(--color-brand)]" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-[var(--color-text-primary)] tracking-tight">Assistant</h1>
          </div>
        </div>
        <button
          onClick={handleClear}
          className="p-3 rounded-xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-red-500 hover:bg-red-500/10 hover:border-red-500/20 transition-all duration-300 shadow-sm"
          title="Clear conversation"
        >
          <Trash2 size={16} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 md:px-12 py-8 space-y-8 scroll-smooth">
        {messages.length === 0 && !sending && (
          <div className="flex flex-col items-center justify-center h-full text-center px-6">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 30 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ duration: 0.6, ease: [0.25, 0.1, 0.25, 1] }}
              className="max-w-2xl"
            >
              <div className="flex items-center justify-center mx-auto mb-8">
                <FoleoLogo size="lg" className="scale-125" />
              </div>
              <h2 className="text-4xl md:text-5xl font-display text-[var(--color-text-primary)] mb-6 leading-tight">
                <span className="italic font-normal">What</span>{" "}
                <span className="font-light tracking-tight">can I help with?</span>
              </h2>
              <p className="text-[17px] text-[var(--color-text-secondary)] mb-12 font-light leading-relaxed">
                Ask about your portfolio, market trends, tax implications, or get personalized investment advice tailored to the Indian market.
              </p>

              {/* Quick prompts */}
              <div className="flex flex-wrap justify-center gap-3">
                {QUICK_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => sendMessage(prompt)}
                    className="px-5 py-3 rounded-full bg-[var(--color-surface-overlay)] border border-[var(--color-border)] text-sm font-medium text-[var(--color-text-secondary)] shadow-sm hover:text-[var(--color-text-primary)] hover:border-[var(--color-brand)]/50 hover:bg-[var(--color-brand)]/5 hover:-translate-y-0.5 transition-all duration-300"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </motion.div>
          </div>
        )}

        <AnimatePresence mode="popLayout">
          {messages.map((msg, i) => (
            <MessageBubble
              key={i}
              {...msg}
              onNextStep={(step) => sendMessage(step)}
            />
          ))}
        </AnimatePresence>

        {sending && (
          <div className="flex justify-start">
            <div className="flex items-start gap-4">
              <div className="mt-1 shrink-0">
                <FoleoLogo size="sm" />
              </div>
              <TypingIndicator />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-6 md:px-12 py-6 border-t border-[var(--color-border)] bg-[var(--color-surface-base)] relative z-20 shadow-[0_-10px_40px_rgba(0,0,0,0.05)]">
        <form onSubmit={handleSubmit} className="flex items-center gap-4 max-w-4xl mx-auto">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask Foleo anything..."
            disabled={sending}
            className="flex-1 px-6 py-4 rounded-full bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text-primary)] text-base placeholder-[var(--color-text-muted)] focus:outline-none focus:border-[var(--color-brand)] focus:ring-4 focus:ring-[var(--color-brand)]/10 shadow-sm transition-all duration-300 disabled:opacity-60"
            autoFocus
          />
          <button
            type="submit"
            disabled={!input.trim() || sending}
            className="p-4 rounded-full bg-[var(--color-brand)] text-white hover:bg-[var(--color-brand-light)] hover:scale-105 disabled:opacity-40 disabled:scale-100 disabled:hover:bg-[var(--color-brand)] disabled:cursor-not-allowed transition-all duration-300 shadow-xl shadow-[var(--color-brand)]/20"
          >
            {sending ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} className="ml-0.5" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
