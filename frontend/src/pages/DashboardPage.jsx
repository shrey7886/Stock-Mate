import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  MessageCircle,
  Briefcase,
  Sparkles,
  ArrowRight,
  RefreshCw,
  AlertCircle,
  Link2,
} from "lucide-react";
import { portfolio, chat } from "../services/api";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const COLORS = ["#10B981", "#06B6D4", "#8B5CF6", "#F59E0B", "#EF4444", "#EC4899", "#6366F1", "#14B8A6"];

const ease = [0.25, 0.1, 0.25, 1];

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } }
};
const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, ease } }
};

function StatCard({ icon: Icon, label, value, sub, trend }) {
  const isPositive = trend === "up";
  return (
    <motion.div variants={itemVariants} className="glass-card glass-card-hover p-8 group">
      <div className="flex items-start justify-between mb-6">
        <div className="p-3 rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)] shadow-sm group-hover:scale-110 transition-transform duration-300">
          <Icon size={20} className="text-[var(--color-text-secondary)] group-hover:text-[var(--color-text-primary)] transition-colors" strokeWidth={1.8} />
        </div>
        {trend && (
          <span className={`flex items-center gap-1.5 text-xs font-bold px-3 py-1.5 rounded-full ${isPositive ? "bg-[var(--color-profit)]/10 text-[var(--color-profit)]" : "bg-[var(--color-loss)]/10 text-[var(--color-loss)]"}`}>
            {isPositive ? <TrendingUp size={13} strokeWidth={2.5} /> : <TrendingDown size={13} strokeWidth={2.5} />}
            {sub}
          </span>
        )}
      </div>
      <p className="text-3xl font-display font-semibold text-[var(--color-text-primary)] tracking-tight">{value}</p>
      <p className="text-xs text-[var(--color-text-secondary)] mt-2 uppercase tracking-[1.5px] font-semibold">{label}</p>
    </motion.div>
  );
}

function HealthGauge({ score }) {
  const clampedScore = Math.max(0, Math.min(100, score || 0));
  const color =
    clampedScore >= 70 ? "var(--color-brand)" : clampedScore >= 40 ? "#F59E0B" : "var(--color-loss)";
  const circumference = 2 * Math.PI * 58;
  const offset = circumference - (clampedScore / 100) * circumference;

  return (
    <div className="relative w-40 h-40 mx-auto group">
      <svg viewBox="0 0 130 130" className="w-full h-full -rotate-90 drop-shadow-sm">
        <circle cx="65" cy="65" r="58" fill="none" stroke="var(--color-border)" strokeWidth="8" />
        <motion.circle
          cx="65" cy="65" r="58" fill="none"
          stroke={color} strokeWidth="8" strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.5, ease: [0.25, 0.1, 0.25, 1] }}
          className="group-hover:opacity-80 transition-opacity"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-display font-semibold text-[var(--color-text-primary)] tracking-tight">{clampedScore}</span>
        <span className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-[3px] font-bold mt-1">Health</span>
      </div>
    </div>
  );
}

function InsightCard({ insight }) {
  return (
    <div className="flex items-start gap-4 p-5 rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] hover:border-[var(--color-brand)]/30 hover:shadow-md transition-all duration-300">
      <div className="p-2 rounded-xl bg-[var(--color-brand)]/10 mt-1 shrink-0">
        <Sparkles size={16} className="text-[var(--color-brand)]" />
      </div>
      <div>
        <p className="text-[15px] text-[var(--color-text-primary)] leading-relaxed font-light">{insight.message || insight}</p>
        {insight.action && (
          <span className="inline-block mt-3 text-sm text-[var(--color-brand)] font-medium flex items-center gap-1 hover:gap-2 transition-all cursor-pointer">
            {insight.action} <ArrowRight size={14} />
          </span>
        )}
      </div>
    </div>
  );
}

function NoBrokerState() {
  return (
    <motion.div variants={itemVariants} className="glass-card p-16 flex flex-col items-center text-center gap-6 shadow-xl">
      <div className="p-6 rounded-3xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] shadow-sm">
        <Link2 size={36} className="text-[var(--color-text-muted)]" />
      </div>
      <div>
        <h2 className="text-3xl font-display text-[var(--color-text-primary)] mb-4">
          <span className="italic font-normal">Connect</span>{" "}
          <span className="font-light text-[var(--color-text-primary)]/90">your broker</span>
        </h2>
        <p className="text-base text-[var(--color-text-secondary)] max-w-md font-light leading-relaxed mx-auto">
          Link your Zerodha account to see live portfolio data, P&amp;L, holdings, and AI-powered insights.
        </p>
      </div>
      <Link
        to="/broker"
        className="mt-4 flex items-center gap-2 px-8 py-4 rounded-full bg-[var(--color-text-primary)] text-[var(--color-surface)] font-semibold text-sm hover:scale-105 transition-transform shadow-xl hover:shadow-2xl"
      >
        <Link2 size={16} />
        Connect Broker
      </Link>
    </motion.div>
  );
}

function Shimmer({ className = "" }) {
  return <div className={`animate-shimmer rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)] ${className}`} />;
}

function _to_float(v) {
  const n = parseFloat(v);
  return isNaN(n) ? 0 : n;
}

export default function DashboardPage() {
  const [portfolioData, setPortfolioData] = useState(null);
  const [insights, setInsights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [insightsLoading, setInsightsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await portfolio.summary();
      setPortfolioData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchInsights = async () => {
    setInsightsLoading(true);
    try {
      const data = await chat.proactiveInsights();
      setInsights(data.insights || []);
    } catch {
      // non-critical
    } finally {
      setInsightsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    fetchInsights();
  }, []);

  const formatCurrency = (val) => {
    if (val == null) return "—";
    const n = Number(val);
    if (isNaN(n)) return "—";
    return "₹" + n.toLocaleString("en-IN", { maximumFractionDigits: 0 });
  };

  const formatPct = (val) => {
    if (val == null) return "";
    const n = Number(val);
    return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
  };

  const isLinked = portfolioData?.linked === true;
  const isLive = portfolioData?.data_status === "live";
  const needsRelink = portfolioData?.action_required === "relink_broker";
  const pnl = portfolioData?.total_pnl ?? 0;
  const pnlPct = portfolioData?.total_pnl_pct ?? 0;
  const totalValue = portfolioData?.total_current_value ?? 0;
  const holdings = portfolioData?.holdings || [];
  const healthScore = portfolioData?.health_score;

  const sectorMap = {};
  holdings.forEach((h) => {
    const key = h.exchange || h.sector || "Other";
    const val = _to_float(h.last_price) * _to_float(h.quantity);
    sectorMap[key] = (sectorMap[key] || 0) + val;
  });
  const pieData = Object.entries(sectorMap)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }));

  return (
    <div className="p-6 md:p-12 max-w-[1400px] mx-auto space-y-10 min-h-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-5xl font-display leading-[1.1] text-[var(--color-text-primary)]"
          >
            <span className="italic font-normal">Your</span>{" "}
            <span className="font-light tracking-tight">portfolio</span>
          </motion.h1>
          <p className="text-sm text-[var(--color-text-secondary)] mt-3 font-light tracking-wide uppercase">
            {isLive
              ? `Live Data · ACC: ${portfolioData.account_id}`
              : isLinked
              ? `Broker linked · ${portfolioData?.data_status || 'syncing'}`
              : "Connect a broker to see live data"}
          </p>
        </div>
        <button
          onClick={() => { fetchData(); fetchInsights(); }}
          className="p-3 rounded-xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-border-subtle)] transition-all duration-300 shadow-sm"
          title="Refresh Dashboard"
        >
          <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {/* Errors / Warnings */}
      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex items-center gap-3 px-6 py-5 rounded-2xl bg-red-500/10 border border-red-500/20 shadow-sm">
            <AlertCircle size={20} className="text-red-500 shrink-0" />
            <p className="text-[15px] text-red-500 font-medium">{error}</p>
          </motion.div>
        )}
        {!loading && needsRelink && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex items-center justify-between px-6 py-5 rounded-2xl bg-amber-500/10 border border-amber-500/20 shadow-sm">
            <div className="flex items-center gap-4">
              <AlertCircle size={20} className="text-amber-500 shrink-0" />
              <p className="text-[15px] text-amber-600 dark:text-amber-400 font-medium">Session expired. Reconnect to see live data.</p>
            </div>
            <Link to="/broker" className="shrink-0 text-[13px] font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400 hover:text-amber-700 underline">Reconnect</Link>
          </motion.div>
        )}
      </AnimatePresence>

      {loading ? (
        <div className="space-y-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => <Shimmer key={i} className="h-36" />)}
          </div>
          <div className="grid lg:grid-cols-2 gap-8">
            <Shimmer className="h-80" />
            <Shimmer className="h-80" />
          </div>
        </div>
      ) : !isLinked && !needsRelink ? (
        <NoBrokerState />
      ) : (
        <motion.div variants={containerVariants} initial="hidden" animate="show" className="space-y-8">
          
          {/* Stats grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            <StatCard icon={Briefcase} label="Current Value" value={formatCurrency(totalValue)} />
            <StatCard icon={pnl >= 0 ? TrendingUp : TrendingDown} label="Total P&L" value={formatCurrency(pnl)} sub={formatPct(pnlPct)} trend={pnl >= 0 ? "up" : "down"} />
            <StatCard icon={Activity} label="Holdings" value={holdings.length} />
            <StatCard icon={Activity} label="Invested" value={formatCurrency(portfolioData?.total_invested)} />
          </div>

          {/* Dash mid row */}
          <div className="grid lg:grid-cols-5 gap-6">
            {/* Health */}
            <motion.div variants={itemVariants} className="glass-card p-10 lg:col-span-2 flex flex-col items-center text-center justify-center relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-[var(--color-brand)]/5 rounded-full blur-[80px] -mr-32 -mt-32 pointer-events-none" />
              <h2 className="text-xs font-bold text-[var(--color-text-muted)] mb-8 uppercase tracking-[3px] self-start w-full text-left">Portfolio Health</h2>
              {healthScore != null ? (
                <>
                  <HealthGauge score={healthScore} />
                  <p className="text-sm text-[var(--color-text-secondary)] mt-8 leading-relaxed max-w-[250px]">
                    {healthScore >= 70 ? "Your portfolio is in great shape!" : healthScore >= 40 ? "Room for improvement — diversification may help." : "Consider rebalancing your portfolio."}
                  </p>
                </>
              ) : (
                <p className="text-sm text-[var(--color-text-muted)] py-12">Health score unavailable</p>
              )}
            </motion.div>

            {/* Allocation */}
            <motion.div variants={itemVariants} className="glass-card p-10 lg:col-span-3 flex flex-col">
              <h2 className="text-xs font-bold text-[var(--color-text-muted)] mb-6 uppercase tracking-[3px]">Allocation</h2>
              {pieData.length > 0 ? (
                <div className="flex-1 flex flex-col sm:flex-row items-center gap-8">
                  <div className="w-full sm:w-1/2 h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={pieData} cx="50%" cy="50%" innerRadius={70} outerRadius={100} paddingAngle={4} dataKey="value" stroke="none">
                          {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="w-full sm:w-1/2 flex flex-col gap-4">
                    {pieData.map((d, i) => (
                      <div key={d.name} className="flex items-center justify-between p-3 rounded-xl bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)]">
                        <div className="flex items-center gap-3">
                          <div className="w-3.5 h-3.5 rounded-full shadow-sm" style={{ background: COLORS[i % COLORS.length] }} />
                          <span className="text-[13px] font-medium text-[var(--color-text-primary)]">{d.name}</span>
                        </div>
                        <span className="text-[13px] text-[var(--color-text-secondary)]">{formatCurrency(d.value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-sm text-[var(--color-text-muted)] text-center m-auto py-12">No allocation data</p>
              )}
            </motion.div>
          </div>

          {/* Holdings Table */}
          {holdings.length > 0 && (
            <motion.div variants={itemVariants} className="glass-card p-0 overflow-hidden">
              <div className="p-8 pb-6 border-b border-[var(--color-border-subtle)]">
                <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-[3px]">Holdings ({holdings.length})</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-[var(--color-surface-overlay)]/50">
                    <tr className="text-[11px] text-[var(--color-text-muted)] uppercase tracking-[2px]">
                      <th className="text-left font-bold px-8 py-5">Symbol</th>
                      <th className="text-right font-bold px-8 py-5">Qty</th>
                      <th className="text-right font-bold px-8 py-5">Avg Price</th>
                      <th className="text-right font-bold px-8 py-5">LTP</th>
                      <th className="text-right font-bold px-8 py-5">Current Value</th>
                      <th className="text-right font-bold px-8 py-5">P&L</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--color-border-subtle)]">
                    {holdings.map((h, i) => {
                      const qty = _to_float(h.quantity);
                      const avgPrice = _to_float(h.average_price);
                      const ltp = _to_float(h.last_price);
                      const currentVal = qty * ltp;
                      const invested = qty * avgPrice;
                      const pnlVal = currentVal - invested;
                      const pnlPctVal = invested > 0 ? (pnlVal / invested) * 100 : 0;
                      const pos = pnlVal >= 0;
                      return (
                        <tr key={i} className="hover:bg-[var(--color-surface-overlay)] transition-colors duration-200 group">
                          <td className="px-8 py-5 font-semibold text-[var(--color-text-primary)]">
                            {h.tradingsymbol || h.symbol || "—"}
                            {h.exchange && <span className="ml-2 px-2 py-0.5 rounded text-[10px] bg-[var(--color-border)] text-[var(--color-text-secondary)] font-medium uppercase tracking-wider">{h.exchange}</span>}
                          </td>
                          <td className="px-8 py-5 text-right text-[var(--color-text-secondary)] font-medium">{qty}</td>
                          <td className="px-8 py-5 text-right text-[var(--color-text-secondary)] font-medium">{formatCurrency(avgPrice)}</td>
                          <td className="px-8 py-5 text-right text-[var(--color-text-secondary)] font-medium">{formatCurrency(ltp)}</td>
                          <td className="px-8 py-5 text-right text-[var(--color-text-primary)] font-bold">{formatCurrency(currentVal)}</td>
                          <td className={`px-8 py-5 text-right font-bold ${pos ? "text-[var(--color-profit)]" : "text-[var(--color-loss)]"}`}>
                            {formatCurrency(pnlVal)}
                            <span className="block text-[11px] font-medium opacity-80 mt-1">{formatPct(pnlPctVal)}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}

          {/* Bottom section: Insights & Actions */}
          <div className="grid lg:grid-cols-3 gap-6">
            <motion.div variants={itemVariants} className="glass-card p-8 lg:col-span-2">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-xl bg-[var(--color-brand)]/10">
                    <Sparkles size={16} className="text-[var(--color-brand)]" />
                  </div>
                  <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-[3px]">AI Insights</h2>
                </div>
                <Link to="/chat" className="flex items-center justify-center gap-2 px-4 py-2 rounded-full border border-[var(--color-border)] text-[11px] font-bold tracking-widest uppercase text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface-overlay)] transition-all">
                  Chat <ArrowRight size={13} />
                </Link>
              </div>
              
              {insightsLoading ? (
                <div className="space-y-4">
                  {[...Array(2)].map((_, i) => <Shimmer key={i} className="h-24" />)}
                </div>
              ) : insights.length > 0 ? (
                <div className="space-y-4">
                  {insights.map((ins, i) => <InsightCard key={i} insight={ins} />)}
                </div>
              ) : (
                <div className="py-12 flex flex-col items-center justify-center text-center border border-dashed border-[var(--color-border-subtle)] rounded-2xl">
                   <MessageCircle size={32} className="text-[var(--color-text-muted)]/50 mb-4" />
                   <p className="text-[15px] text-[var(--color-text-muted)] max-w-sm">No insights yet. Complete a portfolio sync and chat with Foleo AI to generate some.</p>
                </div>
              )}
            </motion.div>

            {/* Actions */}
            <motion.div variants={itemVariants} className="flex flex-col gap-6">
              {[
                { to: "/chat", icon: MessageCircle, title: "AI Assistant", desc: "Ask questions, get advice" },
                { to: "/broker", icon: Link2, title: "Broker Connections", desc: "Manage sync and permissions" },
              ].map((action) => (
                <Link key={action.to} to={action.to} className="flex-1 flex flex-col justify-center glass-card hover:border-[var(--color-brand)]/30 hover:scale-[1.02] transition-all p-8 group">
                  <div className="p-3.5 rounded-2xl bg-[var(--color-surface-overlay)] w-max mb-5 group-hover:bg-[var(--color-brand)]/10 group-hover:scale-110 transition-all shadow-sm">
                    <action.icon size={22} className="text-[var(--color-text-secondary)] group-hover:text-[var(--color-brand)] transition-colors" />
                  </div>
                  <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">{action.title}</h3>
                  <p className="text-[13px] text-[var(--color-text-secondary)]">{action.desc}</p>
                </Link>
              ))}
            </motion.div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

