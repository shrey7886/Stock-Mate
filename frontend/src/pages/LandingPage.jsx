import { useNavigate, Link } from "react-router-dom";
import FoleoLogo from "../components/FoleoLogo";
import { motion } from "framer-motion";
import {
  MessageCircle,
  Shield,
  ArrowRight,
  Sparkles,
  BarChart3,
  Bot,
  ArrowUpRight,
} from "lucide-react";
import { useTheme } from "../context/ThemeContext";

const FEATURES = [
  {
    icon: Bot,
    title: "AI-Powered Chat",
    desc: "Ask anything about your portfolio, markets, or investing strategies. Get intelligent, context-aware answers instantly.",
  },
  {
    icon: BarChart3,
    title: "Portfolio Analytics",
    desc: "Real-time portfolio health scores, sector analysis, risk metrics, and performance tracking — all in one dashboard.",
  },
  {
    icon: Shield,
    title: "RAG Knowledge Base",
    desc: "Backed by a curated knowledge base covering Indian tax law, SIP strategies, technical analysis, and behavioral finance.",
  },
];

const CHAT_PREVIEW = [
  { role: "user", text: "What's my portfolio health looking like?" },
  {
    role: "ai",
    text: "Your portfolio scores 78/100 — solid! 💪 Your tech allocation is a bit heavy at 42%. Consider diversifying into pharma or FMCG to improve your risk-adjusted returns.",
  },
  { role: "user", text: "Should I add more to my HDFC Bank SIP?" },
  {
    role: "ai",
    text: "HDFC Bank is trading at a P/E of 19.2 — slightly below its 5-year average. Increasing your SIP here could be a smart move for long-term accumulation. 📈",
  },
];

const ease = [0.25, 0.1, 0.25, 1];

const staggerContainer = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.15 }
  }
};

const staggerItem = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { duration: 0.8, ease } }
};

export default function LandingPage() {
  const navigate = useNavigate();
  const { theme } = useTheme();

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5, ease: [0.25, 0.1, 0.25, 1] }}
      className="min-h-screen bg-[var(--color-surface)] overflow-hidden transition-colors duration-500"
    >
      {/* ═══ HERO (sky gradient, Origin-inspired) ═══ */}
      <section className="hero-sky min-h-screen flex flex-col relative">
        <div className="hero-sky-light absolute inset-0 pointer-events-none" />

        {/* Atmospheric cloud layers (mostly visible in dark mode via opacity) */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden mix-blend-screen opacity-60">
          <div className="absolute top-[15%] left-[-10%] w-[60%] h-[40%] rounded-full bg-[var(--color-brand)]/10 blur-[120px] float-slow" />
          <div className="absolute top-[30%] right-[-5%] w-[50%] h-[35%] rounded-full bg-white/5 blur-[100px] float-medium" />
          <div className="absolute bottom-[20%] left-[20%] w-[40%] h-[30%] rounded-full bg-[var(--color-brand)]/5 blur-[80px] float-slow" style={{ animationDelay: "3s" }} />
        </div>

        {/* Nav */}
        <nav className="relative z-50 flex items-center justify-between px-6 md:px-16 lg:px-24 py-6 pointer-events-auto">
          <div className="flex items-center mt-1">
            <FoleoLogo size="lg" className="drop-shadow-sm" />
          </div>
          <div className="flex items-center gap-4">
            <Link
              to="/login"
              className="px-5 py-2.5 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors inline-block"
              style={{ letterSpacing: "1.5px", fontSize: "11px", fontWeight: 600, textTransform: "uppercase" }}
            >
              Log In
            </Link>
            <Link
              to="/login"
              className="px-6 py-3 font-semibold rounded-full bg-[var(--color-text-primary)] text-[var(--color-surface)] hover:scale-105 transition-all shadow-xl shadow-black/10 flex items-center gap-2"
              style={{ letterSpacing: "0.5px", fontSize: "12px" }}
            >
              GET STARTED <ArrowRight size={14} />
            </Link>
          </div>
        </nav>

        {/* Hero content */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="show"
          className="flex-1 flex flex-col items-center justify-center text-center px-6 relative z-10 -mt-20"
        >
          {/* Badge pill */}
          <motion.div variants={staggerItem} className="badge-pill mb-12 shadow-sm">
            AI-POWERED WEALTH INTELLIGENCE
          </motion.div>

          {/* Main headline — Origin serif style */}
          <motion.h1 variants={staggerItem} className="font-display leading-[1.05] max-w-5xl">
            <span className="block text-6xl md:text-8xl lg:text-[7rem] italic font-normal text-[var(--color-text-primary)] mb-2">Own</span>{" "}
            <span className="block text-6xl md:text-8xl lg:text-[7rem] font-light text-[var(--color-text-primary)]/90 tracking-tight">your wealth.</span>
          </motion.h1>

          {/* Sub headline */}
          <motion.p variants={staggerItem} className="mt-8 text-lg md:text-xl text-[var(--color-text-secondary)] max-w-2xl leading-relaxed font-light">
            Foleo is your personal AI Financial Advisor.
            <br className="hidden md:block" />
            Track your investments, analyze your portfolio and optimize
            <br className="hidden md:block" />
            your financial future—all in one place.
          </motion.p>

          {/* CTA */}
          <motion.button
            variants={staggerItem}
            onClick={() => navigate("/login")}
            className="mt-12 px-10 py-4 rounded-full bg-[var(--color-brand)] text-white font-semibold text-sm hover:bg-[var(--color-brand-light)] hover:-translate-y-1 transition-all shadow-2xl shadow-[var(--color-brand)]/20 flex items-center gap-2"
            style={{ letterSpacing: "0.5px", fontSize: "13px" }}
          >
            START FOR FREE <ArrowRight size={15} />
          </motion.button>

          {/* Origin-style chat input preview */}
          <motion.div variants={staggerItem} className="mt-16 w-full max-w-2xl group cursor-pointer" onClick={() => navigate("/login")}>
            <div className="origin-input flex items-center gap-4 px-8 py-5 shadow-2xl shadow-black/5">
              <span className="flex-1 text-left text-[var(--color-text-muted)] text-base font-light group-hover:text-[var(--color-text-secondary)] transition-colors">Ask Foleo anything...</span>
              <div className="w-12 h-12 rounded-full bg-[var(--color-brand)]/10 flex items-center justify-center shrink-0 group-hover:scale-110 group-hover:bg-[var(--color-brand)]/20 transition-all duration-300">
                <ArrowUpRight size={20} className="text-[var(--color-brand)]" />
              </div>
            </div>
            <p className="mt-5 text-sm text-[var(--color-text-muted)]/70 font-light tracking-wide uppercase">Track everything. Ask anything.</p>
          </motion.div>
        </motion.div>
      </section>

      {/* ═══ SOCIAL PROOF ═══ */}
      <section className="py-20 px-6 border-y border-[var(--color-border-subtle)] bg-[var(--color-surface-base)]">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 1 }}
          className="flex flex-wrap items-center justify-center gap-12 text-[var(--color-text-muted)] text-xs tracking-[2.5px] uppercase font-bold"
        >
          <span>RAG-Powered Intelligence</span>
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-border)]" />
          <span>14 Financial Intents</span>
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-border)]" />
          <span>Zerodha Integration</span>
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-border)]" />
          <span>Real-Time Analytics</span>
        </motion.div>
      </section>

      {/* ═══ CHAT PREVIEW ═══ */}
      <section className="px-6 md:px-12 py-32 bg-[var(--color-surface)]">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 50 }}
          whileInView={{ opacity: 1, scale: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 1, ease }}
          className="max-w-3xl mx-auto glass-card p-8 md:p-12 shadow-2xl"
        >
          <div className="flex items-center gap-4 mb-10 border-b border-[var(--color-border-subtle)] pb-6">
            <div className="w-10 h-10 rounded-2xl bg-[var(--color-brand)]/10 flex items-center justify-center">
              <MessageCircle size={18} className="text-[var(--color-brand)]" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-[var(--color-text-primary)]">Foleo AI</h3>
              <span className="text-sm text-[var(--color-text-muted)] font-light">Preview Conversation</span>
            </div>
          </div>

          <div className="space-y-6">
            {CHAT_PREVIEW.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: msg.role === "user" ? 20 : -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.15, ease }}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={`max-w-[85%] px-5 py-4 text-[15px] leading-relaxed shadow-sm ${msg.role === "user" ? "chat-bubble-user text-[var(--color-text-primary)]" : "chat-bubble-ai text-[var(--color-text-secondary)]"}`}>
                  {msg.text}
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-10 flex items-center gap-4 px-6 py-4 rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)]">
            <span className="text-[15px] text-[var(--color-text-muted)] flex-1 font-light">Ask Foleo anything...</span>
            <div className="p-2.5 rounded-xl bg-[var(--color-brand)]/10">
              <ArrowRight size={16} className="text-[var(--color-brand)]" />
            </div>
          </div>
        </motion.div>
      </section>

      {/* ═══ FEATURES ═══ */}
      <section id="features" className="px-6 md:px-12 py-32 bg-[var(--color-surface-base)] border-t border-[var(--color-border-subtle)]">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.8, ease }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-6xl font-display text-[var(--color-text-primary)]">Everything you need</h2>
          <p className="mt-6 text-[var(--color-text-secondary)] text-xl max-w-2xl mx-auto font-light leading-relaxed">
            A complete AI wealth management toolkit, designed for the modern investor.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, delay: i * 0.15, ease }}
              className="glass-card glass-card-hover p-10 bg-[var(--color-surface)]"
            >
              <div className="w-14 h-14 rounded-2xl bg-[var(--color-brand)]/10 flex items-center justify-center mb-6 border border-[var(--color-brand)]/20 shadow-sm">
                <f.icon size={26} className="text-[var(--color-brand)]" />
              </div>
              <h3 className="text-xl font-semibold text-[var(--color-text-primary)] mb-4">{f.title}</h3>
              <p className="text-[15px] text-[var(--color-text-secondary)] leading-relaxed font-light">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ═══ CTA ═══ */}
      <section className="px-6 md:px-12 py-40 border-t border-[var(--color-border-subtle)] bg-[var(--color-surface)] relative overflow-hidden">
        {/* Glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-[var(--color-brand)]/5 rounded-full blur-[150px] pointer-events-none" />

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease }}
          className="max-w-4xl mx-auto text-center relative z-10"
        >
          <h2 className="text-5xl md:text-7xl font-display text-[var(--color-text-primary)] mb-6 leading-tight">
            <span className="italic block font-normal">Smart</span> investing starts here.
          </h2>
          <p className="text-[var(--color-text-secondary)] text-xl mb-12 font-light">
            Join Foleo and let AI supercharge your investment decisions.
          </p>
          <button
            onClick={() => navigate("/login")}
            className="px-12 py-5 rounded-full bg-[var(--color-brand)] text-white font-semibold hover:bg-[var(--color-brand-light)] hover:-translate-y-1 hover:scale-105 transition-all shadow-2xl shadow-[var(--color-brand)]/20 text-sm"
            style={{ letterSpacing: "1px" }}
          >
            GET STARTED — IT'S FREE
          </button>
        </motion.div>
      </section>

      {/* ═══ FOOTER ═══ */}
      <footer className="border-t border-[var(--color-border)] px-6 md:px-12 py-10 bg-[var(--color-surface-base)]">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6 max-w-6xl mx-auto">
          <div className="flex items-center gap-3">
            <FoleoLogo size="md" className="pr-1" />
            <span className="text-sm font-medium text-[var(--color-text-secondary)]">Foleo &copy; {new Date().getFullYear()}</span>
          </div>
          <p className="text-xs text-[var(--color-text-muted)] uppercase tracking-wide">Not financial advice. For educational purposes only.</p>
        </div>
      </footer>
    </motion.div>
  );
}
