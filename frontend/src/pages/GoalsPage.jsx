import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Target,
  Plus,
  Trash2,
  RefreshCw,
  AlertCircle,
  Calculator,
  IndianRupee,
  TrendingUp,
} from "lucide-react";
import { goals } from "../services/api";

function Shimmer({ className = "" }) {
  return <div className={`animate-shimmer rounded-xl bg-white/5 ${className}`} />;
}

function ProgressBar({ current, target }) {
  const pct = target > 0 ? Math.min(100, (current / target) * 100) : 0;
  return (
    <div className="w-full h-2 rounded-full bg-white/5 overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-cyan-400"
      />
    </div>
  );
}

function SIPCalculator() {
  const [monthly, setMonthly] = useState(10000);
  const [years, setYears] = useState(10);
  const [rate, setRate] = useState(12);

  const months = years * 12;
  const r = rate / 100 / 12;
  const futureValue = r > 0
    ? monthly * ((Math.pow(1 + r, months) - 1) / r) * (1 + r)
    : monthly * months;
  const invested = monthly * months;
  const wealth = futureValue - invested;

  const formatCurrency = (val) =>
    "₹" + Math.round(val).toLocaleString("en-IN");

  return (
    <div className="glass-card p-6 space-y-5">
      <div className="flex items-center gap-2 mb-1">
        <Calculator size={18} className="text-emerald-400" />
        <h2 className="text-sm font-medium text-neutral-300">SIP Calculator</h2>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs text-neutral-500">Monthly Investment</label>
            <span className="text-xs text-emerald-400 font-medium tabular-nums">
              {formatCurrency(monthly)}
            </span>
          </div>
          <input
            type="range"
            min={500}
            max={100000}
            step={500}
            value={monthly}
            onChange={(e) => setMonthly(Number(e.target.value))}
            className="w-full accent-emerald-500"
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs text-neutral-500">Time Period</label>
            <span className="text-xs text-emerald-400 font-medium tabular-nums">{years} years</span>
          </div>
          <input
            type="range"
            min={1}
            max={30}
            step={1}
            value={years}
            onChange={(e) => setYears(Number(e.target.value))}
            className="w-full accent-emerald-500"
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs text-neutral-500">Expected Return (p.a.)</label>
            <span className="text-xs text-emerald-400 font-medium tabular-nums">{rate}%</span>
          </div>
          <input
            type="range"
            min={1}
            max={30}
            step={0.5}
            value={rate}
            onChange={(e) => setRate(Number(e.target.value))}
            className="w-full accent-emerald-500"
          />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 pt-4 border-t border-white/5">
        <div>
          <p className="text-xs text-neutral-500 mb-1">Invested</p>
          <p className="text-sm font-semibold text-white tabular-nums">{formatCurrency(invested)}</p>
        </div>
        <div>
          <p className="text-xs text-neutral-500 mb-1">Wealth Gained</p>
          <p className="text-sm font-semibold text-emerald-400 tabular-nums">{formatCurrency(wealth)}</p>
        </div>
        <div>
          <p className="text-xs text-neutral-500 mb-1">Maturity Value</p>
          <p className="text-sm font-semibold text-white tabular-nums">{formatCurrency(futureValue)}</p>
        </div>
      </div>
    </div>
  );
}

export default function GoalsPage() {
  const [userGoals, setUserGoals] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showForm, setShowForm] = useState(false);

  // Form fields
  const [goalName, setGoalName] = useState("");
  const [targetAmount, setTargetAmount] = useState("");
  const [currentAmount, setCurrentAmount] = useState("");
  const [deadline, setDeadline] = useState("");
  const [saving, setSaving] = useState(false);

  const fetchGoals = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await goals.get();
      setUserGoals(data.goals || data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGoals();
  }, []);

  const handleSave = async (e) => {
    e.preventDefault();
    if (!goalName.trim() || !targetAmount) return;
    setSaving(true);
    try {
      await goals.set({
        name: goalName.trim(),
        target_amount: Number(targetAmount),
        current_amount: Number(currentAmount) || 0,
        deadline: deadline || undefined,
      });
      setGoalName("");
      setTargetAmount("");
      setCurrentAmount("");
      setDeadline("");
      setShowForm(false);
      fetchGoals();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleClear = async () => {
    try {
      await goals.clear();
      setUserGoals([]);
    } catch (err) {
      setError(err.message);
    }
  };

  const formatCurrency = (val) =>
    val != null ? "₹" + Number(val).toLocaleString("en-IN", { maximumFractionDigits: 0 }) : "—";

  const goalList = Array.isArray(userGoals) ? userGoals : userGoals ? [userGoals] : [];

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-2xl md:text-3xl font-display font-bold text-white"
          >
            Goals
          </motion.h1>
          <p className="text-sm text-neutral-500 mt-1">
            Set financial targets and track progress
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowForm(!showForm)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-500 text-black font-semibold text-sm hover:bg-emerald-400 transition-all shadow-lg shadow-emerald-500/20"
          >
            <Plus size={14} />
            New Goal
          </button>
          <button
            onClick={fetchGoals}
            className="p-2.5 rounded-xl bg-white/5 border border-white/10 text-neutral-400 hover:text-white hover:bg-white/10 transition-all"
            title="Refresh"
          >
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 px-5 py-4 rounded-xl bg-red-500/10 border border-red-500/20">
          <AlertCircle size={18} className="text-red-400 shrink-0" />
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Add goal form */}
      {showForm && (
        <motion.form
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          onSubmit={handleSave}
          className="glass-card p-6 space-y-4"
        >
          <h2 className="text-sm font-medium text-neutral-300">Create Goal</h2>
          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-neutral-500 mb-1.5">Goal Name</label>
              <input
                type="text"
                value={goalName}
                onChange={(e) => setGoalName(e.target.value)}
                placeholder="e.g. Emergency Fund"
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm placeholder-neutral-500 focus:outline-none focus:border-emerald-500/50 transition-all"
                required
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-500 mb-1.5">Target Amount (₹)</label>
              <input
                type="number"
                value={targetAmount}
                onChange={(e) => setTargetAmount(e.target.value)}
                placeholder="500000"
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm placeholder-neutral-500 focus:outline-none focus:border-emerald-500/50 transition-all"
                required
                min="1"
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-500 mb-1.5">Current Amount (₹)</label>
              <input
                type="number"
                value={currentAmount}
                onChange={(e) => setCurrentAmount(e.target.value)}
                placeholder="0"
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm placeholder-neutral-500 focus:outline-none focus:border-emerald-500/50 transition-all"
                min="0"
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-500 mb-1.5">Deadline</label>
              <input
                type="date"
                value={deadline}
                onChange={(e) => setDeadline(e.target.value)}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-emerald-500/50 transition-all"
              />
            </div>
          </div>
          <div className="flex gap-3 pt-2">
            <button
              type="submit"
              disabled={saving}
              className="px-5 py-2.5 rounded-xl bg-emerald-500 text-black font-semibold text-sm hover:bg-emerald-400 disabled:opacity-50 transition-all"
            >
              {saving ? "Saving..." : "Save Goal"}
            </button>
            <button
              type="button"
              onClick={() => setShowForm(false)}
              className="px-5 py-2.5 rounded-xl bg-white/5 border border-white/10 text-neutral-300 text-sm hover:bg-white/10 transition-all"
            >
              Cancel
            </button>
          </div>
        </motion.form>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Goals list */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-neutral-400">Your Goals</h2>
            {goalList.length > 0 && (
              <button
                onClick={handleClear}
                className="flex items-center gap-1 text-xs text-neutral-500 hover:text-red-400 transition-colors"
              >
                <Trash2 size={12} />
                Clear All
              </button>
            )}
          </div>

          {loading ? (
            <div className="space-y-3">
              {[...Array(3)].map((_, i) => <Shimmer key={i} className="h-24" />)}
            </div>
          ) : goalList.length === 0 ? (
            <div className="text-center py-16">
              <Target size={40} className="text-neutral-700 mx-auto mb-4" />
              <p className="text-neutral-400">No goals set yet</p>
              <p className="text-sm text-neutral-600 mt-1">
                Create a financial goal to start tracking
              </p>
            </div>
          ) : (
            goalList.map((g, i) => {
              const current = g.current_amount || 0;
              const target = g.target_amount || 1;
              const pct = Math.min(100, (current / target) * 100);
              return (
                <motion.div
                  key={g.name || i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.08 }}
                  className="glass-card p-5 space-y-3"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-sm font-semibold text-white">{g.name}</h3>
                      {g.deadline && (
                        <p className="text-xs text-neutral-500 mt-0.5">
                          Deadline: {new Date(g.deadline).toLocaleDateString("en-IN")}
                        </p>
                      )}
                    </div>
                    <span className="text-xs font-medium text-emerald-400 tabular-nums">
                      {pct.toFixed(0)}%
                    </span>
                  </div>
                  <ProgressBar current={current} target={target} />
                  <div className="flex items-center justify-between text-xs text-neutral-500">
                    <span>{formatCurrency(current)} saved</span>
                    <span>Target: {formatCurrency(target)}</span>
                  </div>
                </motion.div>
              );
            })
          )}
        </div>

        {/* SIP Calculator */}
        <div>
          <SIPCalculator />
        </div>
      </div>
    </div>
  );
}
