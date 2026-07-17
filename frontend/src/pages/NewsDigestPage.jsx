import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Newspaper, ExternalLink } from "lucide-react";
import { news } from "../services/api";

const ease = [0.25, 0.1, 0.25, 1];

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } },
};
const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, ease } },
};

function Shimmer({ className = "" }) {
  return <div className={`animate-shimmer rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)] ${className}`} />;
}

function ArticleRow({ article }) {
  return (
    <a
      href={article.link || undefined}
      target="_blank"
      rel="noopener noreferrer"
      className="block px-4 py-3 rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-overlay)] hover:border-[var(--color-brand)]/40 transition-all group"
    >
      <p className="text-sm text-[var(--color-text-primary)] leading-relaxed font-medium flex items-start gap-2">
        <span className="flex-1">{article.title}</span>
        <ExternalLink size={13} className="text-[var(--color-text-muted)] group-hover:text-[var(--color-brand)] shrink-0 mt-0.5" />
      </p>
      {(article.publisher || article.published_at) && (
        <p className="text-xs text-[var(--color-text-muted)] mt-2 uppercase tracking-wider">
          {[article.publisher, article.published_at].filter(Boolean).join(" · ")}
        </p>
      )}
    </a>
  );
}

function SymbolNewsCard({ item }) {
  return (
    <motion.div variants={itemVariants} className="glass-card p-6 md:p-8">
      <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-[3px] mb-5">{item.symbol}</h2>
      <div className="space-y-3">
        {item.articles.map((article, i) => (
          <ArticleRow key={i} article={article} />
        ))}
      </div>
    </motion.div>
  );
}

export default function NewsDigestPage() {
  const [digestData, setDigestData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchDigest = async () => {
    setLoading(true);
    try {
      const data = await news.digest();
      setDigestData(data);
    } catch {
      setDigestData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDigest();
  }, []);

  const items = digestData?.items || [];
  const isEmpty = !loading && (!digestData || digestData.data_status !== "live" || items.length === 0);

  return (
    <div className="p-6 md:p-12 max-w-[1400px] mx-auto space-y-10 min-h-full">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-5xl font-display leading-[1.1] text-[var(--color-text-primary)]"
          >
            <span className="italic font-normal">News</span>{" "}
            <span className="font-light tracking-tight">digest</span>
          </motion.h1>
          <p className="text-sm text-[var(--color-text-secondary)] mt-3 font-light tracking-wide uppercase">
            Latest headlines for your holdings
          </p>
        </div>
      </div>

      {loading ? (
        <div className="space-y-6">
          {[...Array(3)].map((_, i) => <Shimmer key={i} className="h-56" />)}
        </div>
      ) : isEmpty ? (
        <div className="glass-card p-16 flex flex-col items-center text-center gap-6">
          <div className="p-6 rounded-3xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] shadow-sm">
            <Newspaper size={36} className="text-[var(--color-text-muted)]" />
          </div>
          <p className="text-base text-[var(--color-text-secondary)] max-w-md font-light leading-relaxed mx-auto">
            No news available right now.
          </p>
        </div>
      ) : (
        <motion.div variants={containerVariants} initial="hidden" animate="show" className="space-y-6">
          {items.map((item) => (
            <SymbolNewsCard key={item.symbol} item={item} />
          ))}
        </motion.div>
      )}
    </div>
  );
}
