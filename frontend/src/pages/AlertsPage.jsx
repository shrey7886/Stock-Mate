import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Bell, Trash2, RefreshCw, AlertCircle, TrendingUp, TrendingDown } from "lucide-react";
import { alerts } from "../services/api";

function Shimmer({ className = "" }) {
  return <div className={`animate-shimmer rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)] ${className}`} />;
}

export default function AlertsPage() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAlerts = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await alerts.list();
      setData(res || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, []);

  const handleDelete = async (id) => {
    try {
      await alerts.remove(id);
      fetchAlerts();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="p-6 md:p-12 max-w-4xl mx-auto space-y-10">
      <div className="flex items-center justify-between">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-3xl md:text-5xl font-display font-semibold text-[var(--color-text-primary)] leading-[1.1]"
          >
            <span className="italic font-normal">My</span> Alerts
          </motion.h1>
          <p className="text-sm text-[var(--color-text-secondary)] mt-3 tracking-wide uppercase font-medium">
            {data.length} alert{data.length !== 1 ? "s" : ""}
          </p>
        </div>
        <button
          onClick={fetchAlerts}
          className="p-3.5 rounded-xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface-raised)] transition-all shadow-sm"
          title="Refresh"
        >
          <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-3 px-6 py-5 rounded-2xl bg-red-500/10 border border-red-500/20 shadow-sm">
          <AlertCircle size={20} className="text-red-500 shrink-0" />
          <p className="text-[15px] font-medium text-red-500">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="space-y-4">
          {[...Array(4)].map((_, i) => <Shimmer key={i} className="h-20" />)}
        </div>
      ) : data.length === 0 ? (
        <div className="glass-card p-16 text-center text-[15px] font-light text-[var(--color-text-muted)]">
          <Bell size={28} className="mx-auto mb-4 text-[var(--color-text-muted)]" />
          No alerts yet. Set one from a stock's fundamentals panel.
        </div>
      ) : (
        <div className="space-y-4">
          {data.map((a, i) => (
            <motion.div
              key={a.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: i * 0.03 }}
              className="glass-card p-6 flex items-center justify-between gap-4"
            >
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-2xl border shadow-sm ${a.is_triggered ? "bg-[var(--color-brand)]/10 border-[var(--color-brand)]/20" : "bg-[var(--color-surface-overlay)] border-[var(--color-border)]"}`}>
                  {a.direction === "above" ? (
                    <TrendingUp size={18} className={a.is_triggered ? "text-[var(--color-brand)]" : "text-[var(--color-text-secondary)]"} />
                  ) : (
                    <TrendingDown size={18} className={a.is_triggered ? "text-[var(--color-brand)]" : "text-[var(--color-text-secondary)]"} />
                  )}
                </div>
                <div>
                  <p className="text-[15px] font-semibold text-[var(--color-text-primary)]">
                    {a.symbol} <span className="font-normal text-[var(--color-text-secondary)]">{a.direction}</span> ₹{a.target_price}
                  </p>
                  <p className="text-[12px] text-[var(--color-text-muted)] mt-0.5 uppercase tracking-wide font-medium">
                    {a.is_triggered ? `Triggered ${a.triggered_at ? new Date(a.triggered_at).toLocaleString() : ""}` : "Active"}
                  </p>
                </div>
              </div>
              <button
                onClick={() => handleDelete(a.id)}
                className="p-2.5 text-[var(--color-text-muted)] hover:text-red-500 transition-colors duration-300 rounded-xl hover:bg-red-500/10"
                title="Delete alert"
              >
                <Trash2 size={16} />
              </button>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
