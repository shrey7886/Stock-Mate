import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Star,
  Plus,
  X,
  RefreshCw,
  AlertCircle,
  Search,
  TrendingUp,
} from "lucide-react";
import { watchlist } from "../services/api";

function Shimmer({ className = "" }) {
  return <div className={`animate-shimmer rounded-xl bg-white/5 ${className}`} />;
}

export default function WatchlistPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [addSymbol, setAddSymbol] = useState("");
  const [adding, setAdding] = useState(false);

  const fetchWatchlist = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await watchlist.get();
      setItems(data.watchlist || data.symbols || data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWatchlist();
  }, []);

  const handleAdd = async (e) => {
    e.preventDefault();
    const sym = addSymbol.trim().toUpperCase();
    if (!sym) return;
    setAdding(true);
    try {
      await watchlist.add(sym);
      setAddSymbol("");
      fetchWatchlist();
    } catch (err) {
      setError(err.message);
    } finally {
      setAdding(false);
    }
  };

  const handleRemove = async (symbol) => {
    try {
      await watchlist.remove(symbol);
      setItems((prev) => prev.filter((s) => (typeof s === "string" ? s : s.symbol) !== symbol));
    } catch (err) {
      setError(err.message);
    }
  };

  const normalizeItem = (item) => {
    if (typeof item === "string") return { symbol: item };
    return item;
  };

  return (
    <div className="p-6 md:p-8 max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-2xl md:text-3xl font-display font-bold text-white"
          >
            Watchlist
          </motion.h1>
          <p className="text-sm text-neutral-500 mt-1">
            Track stocks you're interested in
          </p>
        </div>
        <button
          onClick={fetchWatchlist}
          className="p-2.5 rounded-xl bg-white/5 border border-white/10 text-neutral-400 hover:text-white hover:bg-white/10 transition-all"
          title="Refresh"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center gap-3 px-5 py-4 rounded-xl bg-red-500/10 border border-red-500/20"
        >
          <AlertCircle size={18} className="text-red-400 shrink-0" />
          <p className="text-sm text-red-400">{error}</p>
        </motion.div>
      )}

      {/* Add symbol form */}
      <motion.form
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        onSubmit={handleAdd}
        className="flex items-center gap-3"
      >
        <div className="relative flex-1">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500" />
          <input
            type="text"
            value={addSymbol}
            onChange={(e) => setAddSymbol(e.target.value)}
            placeholder="Add stock symbol (e.g. RELIANCE, TCS)"
            className="w-full pl-9 pr-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white text-sm placeholder-neutral-500 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/25 transition-all"
          />
        </div>
        <button
          type="submit"
          disabled={!addSymbol.trim() || adding}
          className="flex items-center gap-2 px-5 py-3 rounded-xl bg-emerald-500 text-black font-semibold text-sm hover:bg-emerald-400 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-lg shadow-emerald-500/20"
        >
          <Plus size={16} />
          Add
        </button>
      </motion.form>

      {/* Watchlist items */}
      {loading ? (
        <div className="space-y-3">
          {[...Array(4)].map((_, i) => <Shimmer key={i} className="h-16" />)}
        </div>
      ) : items.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-20"
        >
          <Star size={40} className="text-neutral-700 mx-auto mb-4" />
          <p className="text-neutral-400">Your watchlist is empty</p>
          <p className="text-sm text-neutral-600 mt-1">
            Add stock symbols above to start tracking
          </p>
        </motion.div>
      ) : (
        <div className="space-y-2">
          <AnimatePresence mode="popLayout">
            {items.map((item, i) => {
              const { symbol } = normalizeItem(item);
              return (
                <motion.div
                  key={symbol}
                  layout
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20, height: 0 }}
                  transition={{ duration: 0.25, delay: i * 0.03 }}
                  className="flex items-center justify-between px-5 py-4 rounded-xl bg-white/[0.02] border border-white/5 hover:border-emerald-500/10 transition-all group"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                      <TrendingUp size={16} className="text-emerald-400" />
                    </div>
                    <span className="text-sm font-medium text-white">{symbol}</span>
                  </div>
                  <button
                    onClick={() => handleRemove(symbol)}
                    className="p-2 rounded-lg text-neutral-500 hover:text-red-400 hover:bg-red-500/10 transition-all opacity-0 group-hover:opacity-100"
                    title="Remove"
                  >
                    <X size={14} />
                  </button>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
