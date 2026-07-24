import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { LayoutGrid, X, BellPlus } from "lucide-react";
import { baskets, portfolio, alerts } from "../services/api";

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
  return (
    <div
      className={`animate-shimmer rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)] ${className}`}
    />
  );
}

function SetAlertForm({ symbol, defaultPrice }) {
  const [direction, setDirection] = useState("above");
  const [targetPrice, setTargetPrice] = useState(
    defaultPrice ? String(defaultPrice) : "",
  );
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const handleSave = async () => {
    const price = Number(targetPrice);
    if (!price || price <= 0) {
      setFeedback({ type: "error", text: "Enter a valid target price." });
      return;
    }
    setSaving(true);
    setFeedback(null);
    try {
      await alerts.create({ symbol, target_price: price, direction });
      setFeedback({ type: "success", text: "Alert saved." });
    } catch (err) {
      setFeedback({ type: "error", text: err.message });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-4 rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] space-y-3">
      <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-[var(--color-text-muted)]">
        <BellPlus size={14} /> Set Price Alert
      </div>
      <div className="flex items-center gap-2">
        <div className="flex rounded-xl overflow-hidden border border-[var(--color-border)] shrink-0">
          {["above", "below"].map((d) => (
            <button
              key={d}
              onClick={() => setDirection(d)}
              className={`px-3 py-2 text-xs font-semibold capitalize transition-colors ${
                direction === d
                  ? "bg-[var(--color-brand)] text-white"
                  : "bg-[var(--color-surface)] text-[var(--color-text-secondary)]"
              }`}
            >
              {d}
            </button>
          ))}
        </div>
        <input
          type="number"
          value={targetPrice}
          onChange={(e) => setTargetPrice(e.target.value)}
          placeholder="Target price"
          className="flex-1 min-w-0 px-3 py-2 rounded-xl bg-[var(--color-surface)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-brand)]"
        />
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-4 py-2 rounded-xl bg-[var(--color-brand)] text-white text-xs font-semibold disabled:opacity-50 shrink-0"
        >
          {saving ? "Saving..." : "Save"}
        </button>
      </div>
      {feedback && (
        <p
          className={`text-xs font-medium ${feedback.type === "error" ? "text-red-500" : "text-[var(--color-brand)]"}`}
        >
          {feedback.text}
        </p>
      )}
    </div>
  );
}

function StockFinancialsPanel({ symbol, data, loading, onClose }) {
  const f = data?.fundamentals;
  const holding = data?.holding;
  const dataStatus = data?.data_status;

  const formatCurrency = (val) => {
    if (val == null) return "—";
    const n = Number(val);
    if (isNaN(n)) return "—";
    return "₹" + n.toLocaleString("en-IN", { maximumFractionDigits: 0 });
  };

  const formatLarge = (val) => {
    if (val == null) return "—";
    const n = Number(val);
    if (isNaN(n)) return "—";
    if (n >= 1e12) return `₹${(n / 1e12).toFixed(2)}T`;
    if (n >= 1e9) return `₹${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)}Cr`;
    return formatCurrency(n);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/40 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 10, scale: 0.98 }}
        transition={{ duration: 0.3, ease }}
        onClick={(e) => e.stopPropagation()}
        className="glass-card w-full max-w-lg max-h-[85vh] overflow-y-auto p-8 md:p-10 relative"
      >
        <button
          onClick={onClose}
          className="absolute top-6 right-6 p-2 rounded-full bg-[var(--color-surface-overlay)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
        >
          <X size={16} />
        </button>

        <h2 className="text-2xl font-display font-semibold text-[var(--color-text-primary)]">
          {symbol}
        </h2>

        {!loading && (
          <div className="mt-5">
            <SetAlertForm symbol={symbol} defaultPrice={holding?.last_price} />
          </div>
        )}

        {loading ? (
          <div className="space-y-3 mt-6">
            {[...Array(4)].map((_, i) => (
              <Shimmer key={i} className="h-12" />
            ))}
          </div>
        ) : f ? (
          <div className="mt-6 space-y-6">
            {f.long_name && (
              <p className="text-sm text-[var(--color-text-secondary)]">
                {f.long_name}
              </p>
            )}

            {holding && (
              <div className="grid grid-cols-2 gap-3 p-4 rounded-2xl bg-[var(--color-brand)]/5 border border-[var(--color-brand)]/15">
                <div>
                  <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                    Qty
                  </p>
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">
                    {holding.quantity}
                  </p>
                </div>
                <div>
                  <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                    Avg Price
                  </p>
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">
                    {formatCurrency(holding.average_price)}
                  </p>
                </div>
                <div>
                  <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                    LTP
                  </p>
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">
                    {formatCurrency(holding.last_price)}
                  </p>
                </div>
                <div>
                  <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                    P&L
                  </p>
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">
                    {formatCurrency(holding.pnl)}
                  </p>
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                  Sector
                </p>
                <p className="text-sm font-medium text-[var(--color-text-primary)] mt-1">
                  {f.sector || "—"}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                  Market Cap
                </p>
                <p className="text-sm font-medium text-[var(--color-text-primary)] mt-1">
                  {formatLarge(f.market_cap)}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                  P/E (Trailing)
                </p>
                <p className="text-sm font-medium text-[var(--color-text-primary)] mt-1">
                  {f.pe_ratio != null ? f.pe_ratio.toFixed(2) : "—"}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                  P/E (Forward)
                </p>
                <p className="text-sm font-medium text-[var(--color-text-primary)] mt-1">
                  {f.forward_pe != null ? f.forward_pe.toFixed(2) : "—"}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                  Dividend Yield
                </p>
                <p className="text-sm font-medium text-[var(--color-text-primary)] mt-1">
                  {f.dividend_yield != null
                    ? `${(f.dividend_yield * 100).toFixed(2)}%`
                    : "—"}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                  Beta
                </p>
                <p className="text-sm font-medium text-[var(--color-text-primary)] mt-1">
                  {f.beta != null ? f.beta.toFixed(2) : "—"}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                  52W High
                </p>
                <p className="text-sm font-medium text-[var(--color-text-primary)] mt-1">
                  {formatCurrency(f.fifty_two_week_high)}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-wider">
                  52W Low
                </p>
                <p className="text-sm font-medium text-[var(--color-text-primary)] mt-1">
                  {formatCurrency(f.fifty_two_week_low)}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <p className="text-sm text-[var(--color-text-muted)] mt-8 text-center py-8">
            {dataStatus === "market_data_unavailable"
              ? "Fundamentals are temporarily unavailable for this stock."
              : "No fundamentals data available."}
          </p>
        )}
      </motion.div>
    </motion.div>
  );
}

function SymbolChip({ symbol, held, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center px-3 py-1.5 rounded-full text-xs font-semibold border transition-all cursor-pointer hover:shadow-md ${
        held
          ? "bg-[var(--color-brand)]/10 text-[var(--color-brand)] border-[var(--color-brand)]/30 hover:bg-[var(--color-brand)]/15"
          : "bg-[var(--color-surface-overlay)] text-[var(--color-text-secondary)] border-[var(--color-border-subtle)] hover:bg-[var(--color-surface-overlay)]/80"
      }`}
    >
      {symbol}
    </button>
  );
}

function BasketCard({ basket, onStockClick }) {
  const heldSet = new Set(basket.held_symbols || []);
  return (
    <motion.div
      variants={itemVariants}
      className="glass-card glass-card-hover p-6 md:p-8"
    >
      <h2 className="text-lg font-display text-[var(--color-text-primary)] mb-2">
        {basket.theme}
      </h2>
      {basket.description && (
        <p className="text-sm text-[var(--color-text-secondary)] font-light mb-5 leading-relaxed">
          {basket.description}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        {basket.symbols.map((symbol) => (
          <SymbolChip
            key={symbol}
            symbol={symbol}
            held={heldSet.has(symbol)}
            onClick={() => onStockClick(symbol)}
          />
        ))}
      </div>
    </motion.div>
  );
}

export default function ThemedBasketsPage() {
  const [basketsData, setBasketsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [stockFinancials, setStockFinancials] = useState(null);
  const [stockFinancialsLoading, setStockFinancialsLoading] = useState(false);

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

  const handleSelectStock = async (symbol) => {
    setSelectedSymbol(symbol);
    setStockFinancialsLoading(true);
    try {
      const data = await portfolio.stockFinancials(symbol);
      setStockFinancials(data);
    } catch {
      setStockFinancials(null);
    } finally {
      setStockFinancialsLoading(false);
    }
  };

  const handleCloseStockPanel = () => {
    setSelectedSymbol(null);
    setStockFinancials(null);
  };

  const items = basketsData?.baskets || [];
  const isEmpty =
    !loading &&
    (!basketsData || basketsData.data_status !== "live" || items.length === 0);

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
            Top performing stocks by theme
          </p>
        </div>
      </div>

      {loading ? (
        <div className="grid md:grid-cols-2 gap-6">
          {[...Array(4)].map((_, i) => (
            <Shimmer key={i} className="h-48" />
          ))}
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
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="grid md:grid-cols-2 gap-6"
        >
          {items.map((basket) => (
            <BasketCard
              key={basket.theme}
              basket={basket}
              onStockClick={handleSelectStock}
            />
          ))}
        </motion.div>
      )}

      <AnimatePresence>
        {selectedSymbol && (
          <StockFinancialsPanel
            symbol={selectedSymbol}
            data={stockFinancials}
            loading={stockFinancialsLoading}
            onClose={handleCloseStockPanel}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
