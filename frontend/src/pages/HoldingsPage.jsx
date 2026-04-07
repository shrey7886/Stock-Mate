import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Briefcase,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  AlertCircle,
  Search,
} from "lucide-react";
import { portfolio } from "../services/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

function Shimmer({ className = "" }) {
  return <div className={`animate-shimmer rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)] ${className}`} />;
}

export default function HoldingsPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");

  const fetchHoldings = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await portfolio.summary();
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHoldings();
  }, []);

  const holdings = data?.holdings || [];
  const filtered = holdings.filter((h) =>
    (h.symbol || h.tradingsymbol || "").toLowerCase().includes(search.toLowerCase())
  );

  const formatCurrency = (val) =>
    val != null ? "₹" + Number(val).toLocaleString("en-IN", { maximumFractionDigits: 0 }) : "—";

  // P&L bar chart data
  const chartData = holdings
    .map((h) => ({
      name: h.symbol || h.tradingsymbol || "—",
      pnl: h.pnl ?? (h.ltp - h.average_price) * h.quantity ?? 0,
    }))
    .sort((a, b) => b.pnl - a.pnl)
    .slice(0, 10);

  return (
    <div className="p-6 md:p-12 max-w-[1400px] mx-auto space-y-10">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-3xl md:text-5xl font-display font-semibold text-[var(--color-text-primary)] leading-[1.1]"
          >
            <span className="italic font-normal">Your</span> Holdings
          </motion.h1>
          <p className="text-sm text-[var(--color-text-secondary)] mt-3 tracking-wide uppercase font-medium">
            {holdings.length} stock{holdings.length !== 1 ? "s" : ""} in your portfolio
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="relative">
            <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search stocks..."
              className="pl-11 pr-4 py-3 rounded-xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] text-[var(--color-text-primary)] text-sm placeholder-[var(--color-text-muted)] focus:outline-none focus:border-[var(--color-brand)] focus:ring-4 focus:ring-[var(--color-brand)]/10 transition-all w-56 sm:w-64 font-light shadow-sm"
            />
          </div>
          <button
            onClick={fetchHoldings}
            className="p-3.5 rounded-xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface-raised)] transition-all shadow-sm"
            title="Refresh"
          >
            <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 px-6 py-5 rounded-2xl bg-red-500/10 border border-red-500/20 shadow-sm">
          <AlertCircle size={20} className="text-red-500 shrink-0" />
          <p className="text-[15px] font-medium text-red-500">{error}</p>
        </div>
      )}

      {/* P&L bar chart */}
      {!loading && chartData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="glass-card p-10 shadow-lg"
        >
          <h2 className="text-xs font-bold text-[var(--color-text-muted)] mb-8 uppercase tracking-[3px]">
            Top Holdings by P&L
          </h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 20 }}>
              <XAxis type="number" tick={{ fill: "var(--color-text-muted)", fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: "var(--color-text-secondary)", fontSize: 13, fontWeight: 600 }}
                axisLine={false}
                tickLine={false}
                width={70}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--color-surface-raised)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "12px",
                  fontSize: "13px",
                  color: "var(--color-text-primary)",
                  boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)",
                }}
                itemStyle={{ color: "var(--color-text-primary)" }}
                formatter={(value) => [formatCurrency(value), "P&L"]}
              />
              <Bar dataKey="pnl" radius={[0, 6, 6, 0]} barSize={20}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.pnl >= 0 ? "var(--color-profit)" : "var(--color-loss)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Holdings table */}
      {loading ? (
        <div className="space-y-4">
          {[...Array(6)].map((_, i) => <Shimmer key={i} className="h-20" />)}
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="glass-card overflow-hidden shadow-lg p-0"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-[var(--color-surface-overlay)]/50">
                <tr className="border-b border-[var(--color-border-subtle)] text-[11px] text-[var(--color-text-muted)] uppercase tracking-[2px]">
                  <th className="text-left px-8 py-5 font-bold">Stock</th>
                  <th className="text-right px-8 py-5 font-bold">Qty</th>
                  <th className="text-right px-8 py-5 font-bold">Avg Price</th>
                  <th className="text-right px-8 py-5 font-bold">LTP</th>
                  <th className="text-right px-8 py-5 font-bold">Current Value</th>
                  <th className="text-right px-8 py-5 font-bold">P&L</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-border-subtle)]">
                {filtered.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="text-center py-20 text-[15px] font-light text-[var(--color-text-muted)] bg-[var(--color-surface)]">
                      {search ? "No matching holdings found." : "No holdings found. Link your broker to see data."}
                    </td>
                  </tr>
                ) : (
                  filtered.map((h, i) => {
                    const pnl = h.pnl ?? (h.ltp - h.average_price) * h.quantity;
                    const pnlPct = h.pnl_pct ?? (h.average_price ? ((h.ltp - h.average_price) / h.average_price) * 100 : 0);
                    const isPositive = pnl >= 0;
                    return (
                      <motion.tr
                        key={h.symbol || h.tradingsymbol || i}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.2, delay: i * 0.03 }}
                        className="hover:bg-[var(--color-surface-overlay)] transition-colors duration-200 group"
                      >
                        <td className="px-8 py-5">
                          <div className="flex items-center gap-4">
                            <div className="w-10 h-10 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)] flex items-center justify-center text-[13px] font-bold text-[var(--color-text-secondary)] shadow-sm">
                              {(h.symbol || h.tradingsymbol || "?").slice(0, 2)}
                            </div>
                            <div>
                              <p className="text-[15px] font-semibold text-[var(--color-text-primary)]">
                                {h.symbol || h.tradingsymbol}
                              </p>
                              {h.exchange && (
                                <p className="text-[11px] font-medium text-[var(--color-text-muted)] uppercase tracking-wider mt-0.5">{h.exchange}</p>
                              )}
                            </div>
                          </div>
                        </td>
                        <td className="text-right px-8 py-5 text-[15px] text-[var(--color-text-secondary)] font-medium tabular-nums">
                          {h.quantity}
                        </td>
                        <td className="text-right px-8 py-5 text-[15px] text-[var(--color-text-secondary)] font-medium tabular-nums">
                          {formatCurrency(h.average_price)}
                        </td>
                        <td className="text-right px-8 py-5 text-[15px] text-[var(--color-text-primary)] font-semibold tabular-nums">
                          {formatCurrency(h.ltp)}
                        </td>
                        <td className="text-right px-8 py-5 text-[15px] text-[var(--color-text-primary)] font-bold tabular-nums">
                          {formatCurrency(h.current_value ?? h.ltp * h.quantity)}
                        </td>
                        <td className="text-right px-8 py-5">
                          <div className="flex flex-col items-end gap-1">
                            <div className="flex items-center gap-1.5">
                              {isPositive ? (
                                <TrendingUp size={14} className="text-[var(--color-profit)]" />
                              ) : (
                                <TrendingDown size={14} className="text-[var(--color-loss)]" />
                              )}
                              <span className={`text-[15px] font-bold tabular-nums ${isPositive ? "text-[var(--color-profit)]" : "text-[var(--color-loss)]"}`}>
                                {formatCurrency(Math.abs(pnl))}
                              </span>
                            </div>
                            <span className={`text-[12px] font-medium tabular-nums ${isPositive ? "text-[var(--color-profit)]/70" : "text-[var(--color-loss)]/70"}`}>
                              ({isPositive ? "+" : ""}{pnlPct.toFixed(1)}%)
                            </span>
                          </div>
                        </td>
                      </motion.tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  );
}
