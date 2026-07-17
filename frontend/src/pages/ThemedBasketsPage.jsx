import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { LayoutGrid } from "lucide-react";
import { baskets } from "../services/api";

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

function SymbolChip({ symbol, held }) {
  return (
    <span
      className={`inline-flex items-center px-3 py-1.5 rounded-full text-xs font-semibold border ${
        held
          ? "bg-[var(--color-brand)]/10 text-[var(--color-brand)] border-[var(--color-brand)]/30"
          : "bg-[var(--color-surface-overlay)] text-[var(--color-text-secondary)] border-[var(--color-border-subtle)]"
      }`}
    >
      {symbol}
    </span>
  );
}

function BasketCard({ basket }) {
  const heldSet = new Set(basket.held_symbols || []);
  return (
    <motion.div variants={itemVariants} className="glass-card glass-card-hover p-6 md:p-8">
      <h2 className="text-lg font-display text-[var(--color-text-primary)] mb-2">{basket.theme}</h2>
      {basket.description && (
        <p className="text-sm text-[var(--color-text-secondary)] font-light mb-5 leading-relaxed">{basket.description}</p>
      )}
      <div className="flex flex-wrap gap-2">
        {basket.symbols.map((symbol) => (
          <SymbolChip key={symbol} symbol={symbol} held={heldSet.has(symbol)} />
        ))}
      </div>
    </motion.div>
  );
}

export default function ThemedBasketsPage() {
  const [basketsData, setBasketsData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchBaskets = async () => {
    setLoading(true);
    try {
      const data = await baskets.list();
      setBasketsData(data);
    } catch {
      setBasketsData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBaskets();
  }, []);

  const items = basketsData?.baskets || [];
  const isEmpty = !loading && (!basketsData || basketsData.data_status !== "live" || items.length === 0);

  return (
    <div className="p-6 md:p-12 max-w-[1400px] mx-auto space-y-10 min-h-full">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-5xl font-display leading-[1.1] text-[var(--color-text-primary)]"
          >
            <span className="italic font-normal">Themed</span>{" "}
            <span className="font-light tracking-tight">baskets</span>
          </motion.h1>
          <p className="text-sm text-[var(--color-text-secondary)] mt-3 font-light tracking-wide uppercase">
            Curated stock groupings by theme
          </p>
        </div>
      </div>

      {loading ? (
        <div className="grid md:grid-cols-2 gap-6">
          {[...Array(4)].map((_, i) => <Shimmer key={i} className="h-48" />)}
        </div>
      ) : isEmpty ? (
        <div className="glass-card p-16 flex flex-col items-center text-center gap-6">
          <div className="p-6 rounded-3xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] shadow-sm">
            <LayoutGrid size={36} className="text-[var(--color-text-muted)]" />
          </div>
          <p className="text-base text-[var(--color-text-secondary)] max-w-md font-light leading-relaxed mx-auto">
            No themed baskets available right now.
          </p>
        </div>
      ) : (
        <motion.div variants={containerVariants} initial="hidden" animate="show" className="grid md:grid-cols-2 gap-6">
          {items.map((basket) => (
            <BasketCard key={basket.theme} basket={basket} />
          ))}
        </motion.div>
      )}
    </div>
  );
}
