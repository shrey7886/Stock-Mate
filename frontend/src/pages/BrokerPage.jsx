import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Link2,
  Unlink,
  Shield,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  ExternalLink,
  Star,
  Loader2,
  Plus,
} from "lucide-react";
import { zerodha } from "../services/api";

function Shimmer({ className = "" }) {
  return <div className={`animate-shimmer rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)] ${className}`} />;
}

export default function BrokerPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [accounts, setAccounts] = useState([]);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [linking, setLinking] = useState(false);

  useEffect(() => {
    const s = searchParams.get("success");
    const err = searchParams.get("error");
    if (s === "true") {
      setSuccess("Zerodha account linked successfully!");
      setSearchParams({}, { replace: true });
    } else if (err) {
      setError(err);
      setSearchParams({}, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [statusData, accountsData] = await Promise.all([
        zerodha.status().catch(() => null),
        zerodha.accounts().catch(() => ({ accounts: [] })),
      ]);
      setStatus(statusData);
      setAccounts(accountsData?.accounts || accountsData || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleLink = async () => {
    setLinking(true);
    try {
      const data = await zerodha.start();
      if (data.login_url || data.url) {
        window.location.href = data.login_url || data.url;
      }
    } catch (err) {
      setError(err.message);
      setLinking(false);
    }
  };

  const handleUnlink = async (accountId) => {
    try {
      await zerodha.unlink(accountId);
      fetchData();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleSetPrimary = async (accountId) => {
    try {
      await zerodha.setPrimary(accountId);
      fetchData();
    } catch (err) {
      setError(err.message);
    }
  };

  const isLinked = accounts.length > 0 || status?.is_linked;

  return (
    <div className="p-6 md:p-12 max-w-5xl mx-auto space-y-10">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-5xl font-display leading-[1.1] text-[var(--color-text-primary)]"
          >
            <span className="italic font-normal">Your</span>{" "}
            <span className="font-light tracking-tight">broker</span>
          </motion.h1>
          <p className="text-[14px] text-[var(--color-text-secondary)] mt-3 font-light tracking-wide uppercase">
            Connect your zerodha account to view live data
          </p>
        </div>
        <button
          onClick={fetchData}
          className="p-3 rounded-xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface-raised)] transition-all duration-300 shadow-sm"
          title="Refresh Data"
        >
          <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      <AnimatePresence mode="popLayout">
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-3 px-6 py-5 rounded-2xl bg-red-500/10 border border-red-500/20 shadow-sm"
          >
            <AlertCircle size={20} className="text-red-500 shrink-0" />
            <p className="text-[15px] text-red-500 font-medium">{error}</p>
          </motion.div>
        )}

        {success && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-3 px-6 py-5 rounded-2xl bg-[var(--color-brand)]/10 border border-[var(--color-brand)]/20 shadow-sm"
          >
            <CheckCircle2 size={20} className="text-[var(--color-brand)] shrink-0" />
            <p className="text-[15px] text-[var(--color-brand)] font-medium">{success}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {loading ? (
        <div className="space-y-6">
          <Shimmer className="h-40" />
          <Shimmer className="h-28" />
        </div>
      ) : (
        <div className="space-y-8">
          {/* Status card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.25, 0.1, 0.25, 1] }}
            className="glass-card p-10 shadow-xl"
          >
            <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
              <div className={`p-5 rounded-[2rem] border shadow-sm shrink-0 ${isLinked ? "bg-[var(--color-brand)]/10 border-[var(--color-brand)]/20" : "bg-[var(--color-surface-overlay)] border-[var(--color-border)]"}`}>
                {isLinked ? (
                  <CheckCircle2 size={36} className="text-[var(--color-brand)]" />
                ) : (
                  <Link2 size={36} className="text-[var(--color-text-muted)]" />
                )}
              </div>
              <div className="flex-1 space-y-2.5">
                <h2 className="text-2xl md:text-3xl font-display font-semibold text-[var(--color-text-primary)]">
                  {isLinked ? "Zerodha Connected" : (
                    <><span className="italic font-normal">Connect</span>{" "}<span className="font-light tracking-tight">Zerodha</span></>
                  )}
                </h2>
                <p className="text-base text-[var(--color-text-secondary)] max-w-lg font-light leading-relaxed">
                  {isLinked
                    ? `${accounts.length} account${accounts.length !== 1 ? "s" : ""} linked. Your portfolio data syncs automatically.`
                    : "Link your Zerodha account to sync live holdings, P&L, and enable AI-powered portfolio analytics."}
                </p>
                {!isLinked && (
                  <button
                    onClick={handleLink}
                    disabled={linking}
                    className="mt-6 flex items-center gap-2 px-8 py-4 rounded-full bg-[var(--color-brand)] text-white font-semibold text-sm hover:bg-[var(--color-brand-light)] hover:-translate-y-1 hover:shadow-2xl disabled:opacity-50 disabled:hover:-translate-y-0 transition-all duration-300 shadow-xl shadow-[var(--color-brand)]/20 w-fit"
                  >
                    {linking ? (
                      <Loader2 size={16} className="animate-spin" />
                    ) : (
                      <ExternalLink size={16} />
                    )}
                    {linking ? "Authenticating..." : "Link Zerodha Account"}
                  </button>
                )}
              </div>
            </div>
          </motion.div>

          {/* Linked Accounts */}
          {accounts.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.15, ease: [0.25, 0.1, 0.25, 1] }}
              className="space-y-4"
            >
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-[3px]">Linked Accounts</h2>
                <button
                  onClick={handleLink}
                  disabled={linking}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-full border border-[var(--color-border)] text-xs font-semibold text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface-overlay)] transition-all duration-300"
                >
                  <Plus size={14} />
                  Add Another
                </button>
              </div>

              <AnimatePresence>
                {accounts.map((acc, i) => {
                  const id = acc.account_id || acc.client_id || acc.id;
                  const isPrimary = acc.is_primary;
                  return (
                    <motion.div
                      key={id || i}
                      initial={{ opacity: 0, y: 15 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -15, scale: 0.98 }}
                      transition={{ duration: 0.4, delay: i * 0.1 }}
                      className="glass-card hover:scale-[1.01] transition-transform p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4"
                    >
                      <div className="flex items-center gap-5">
                        <div className="p-3 rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border-subtle)] shadow-sm">
                          <Shield size={18} className="text-[var(--color-text-secondary)]" />
                        </div>
                        <div>
                          <div className="flex items-center gap-3">
                            <span className="text-lg font-semibold text-[var(--color-text-primary)]">{id}</span>
                            {isPrimary && (
                              <span className="px-3 py-1 rounded-full bg-[var(--color-brand)]/10 text-[10px] font-bold text-[var(--color-brand)] uppercase tracking-wider relative overflow-hidden">
                                <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
                                Primary
                              </span>
                            )}
                          </div>
                          {acc.user_name && (
                            <p className="text-sm font-medium text-[var(--color-text-secondary)] mt-1">{acc.user_name}</p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center justify-end gap-3 w-full sm:w-auto mt-2 sm:mt-0 pt-4 sm:pt-0 border-t border-[var(--color-border-subtle)] sm:border-0">
                        {!isPrimary && (
                          <button
                            onClick={() => handleSetPrimary(id)}
                            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium text-[var(--color-text-secondary)] hover:text-amber-500 hover:bg-amber-500/10 border border-transparent hover:border-amber-500/20 transition-all duration-300"
                            title="Set as primary"
                          >
                            <Star size={16} /> <span className="sm:hidden">Set Primary</span>
                          </button>
                        )}
                        <button
                          onClick={() => handleUnlink(id)}
                          className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium text-[var(--color-text-secondary)] hover:text-red-500 hover:bg-red-500/10 border border-transparent hover:border-red-500/20 transition-all duration-300"
                          title="Unlink account"
                        >
                          <Unlink size={16} /> <span className="sm:hidden">Unlink</span>
                        </button>
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </motion.div>
          )}

          {/* Security note */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="flex items-start gap-4 p-6 rounded-2xl bg-[var(--color-surface-overlay)] border border-[var(--color-border)] shadow-sm"
          >
            <div className="p-2 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] shrink-0">
              <Shield size={16} className="text-[var(--color-text-secondary)]" />
            </div>
            <p className="text-[13px] text-[var(--color-text-secondary)] leading-relaxed font-medium">
              Your credentials are encrypted end-to-end and never stored in plain text.
              Foleo connects securely via Zerodha's official KiteConnect API using OAuth 2.0.
            </p>
          </motion.div>
        </div>
      )}
    </div>
  );
}


