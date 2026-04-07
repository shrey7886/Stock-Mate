import { useState } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, AlertCircle, Loader2, Lock, ArrowRight } from "lucide-react";
import { auth } from "../services/api";

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");
  const navigate = useNavigate();

  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [status, setStatus] = useState("idle"); // idle, loading, success, error
  const [errorMsg, setErrorMsg] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!token) {
      setErrorMsg("Invalid or missing reset token.");
      setStatus("error");
      return;
    }
    if (password.length < 6) {
      setErrorMsg("Password must be at least 6 characters long.");
      setStatus("error");
      return;
    }
    if (password !== confirmPassword) {
      setErrorMsg("Passwords do not match.");
      setStatus("error");
      return;
    }

    setStatus("loading");
    try {
      await auth.resetPassword(token, password);
      setStatus("success");
    } catch (err) {
      setErrorMsg(err.message || "Failed to reset password. Token may be expired.");
      setStatus("error");
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.6, ease: [0.25, 0.1, 0.25, 1] }}
      className="min-h-screen bg-[var(--color-surface)] flex flex-col items-center justify-center relative overflow-hidden transition-colors duration-500 p-6"
    >
      {/* Ambient glow */}
      <div className="absolute top-[-20%] left-[50%] -translate-x-1/2 w-[600px] h-[600px] bg-[var(--color-brand)]/5 rounded-full blur-[120px] pointer-events-none" />

      <motion.div
        initial={{ scale: 0.95, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.25, 0.1, 0.25, 1], delay: 0.1 }}
        className="w-full max-w-md bg-[var(--color-surface-overlay)] border border-[var(--color-border)] rounded-3xl p-8 md:p-10 shadow-2xl relative z-10"
      >
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-display font-semibold text-[var(--color-text-primary)] mb-2">Create New Password</h1>
          <p className="text-[var(--color-text-secondary)] text-sm">Enter your new strong password below.</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <AnimatePresence mode="popLayout">
            {status === "error" && (
              <motion.div
                initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                animate={{ opacity: 1, height: "auto", marginBottom: 24 }}
                exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                className="flex items-center gap-3 p-4 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-500 text-sm overflow-hidden"
              >
                <AlertCircle size={18} className="shrink-0" />
                <p>{errorMsg}</p>
              </motion.div>
            )}
            
            {status === "success" && (
              <motion.div
                initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                animate={{ opacity: 1, height: "auto", marginBottom: 24 }}
                exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                className="flex flex-col items-center justify-center gap-3 p-6 rounded-2xl bg-green-500/10 border border-green-500/20 text-green-500 text-center overflow-hidden"
              >
                <CheckCircle size={32} className="shrink-0" />
                <p className="font-medium text-base">Password Reset Successfully!</p>
                <p className="text-sm opacity-80">You can now sign in with your new password.</p>
                <button
                  type="button"
                  onClick={() => navigate("/login")}
                  className="mt-4 px-6 py-2.5 rounded-full bg-[var(--color-brand)] text-white hover:bg-[var(--color-brand-light)] font-semibold transition-colors w-full"
                >
                  Return to Login
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          {status !== "success" && (
            <>
              {/* Token Error Fallback */}
              {!token && status !== "success" && (
                <div className="p-4 bg-orange-500/10 border border-orange-500/20 text-orange-500 rounded-2xl text-sm mb-6 flex items-start gap-3">
                  <AlertCircle size={18} className="shrink-0 mt-0.5" />
                  <p>Warning: No reset token detected in URL. The password reset will likely fail.</p>
                </div>
              )}

              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-2">
                    New Password
                  </label>
                  <div className="relative">
                    <Lock size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]" />
                    <input
                      type="password"
                      value={password}
                      onChange={(e) => {
                        setPassword(e.target.value);
                        if (status === "error") setStatus("idle");
                      }}
                      className="w-full pl-11 pr-4 py-3.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-2xl text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-brand)] focus:ring-1 focus:ring-[var(--color-brand)] transition-all"
                      placeholder="••••••••"
                      required
                      disabled={status === "loading"}
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-2">
                    Confirm Password
                  </label>
                  <div className="relative">
                    <Lock size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]" />
                    <input
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => {
                        setConfirmPassword(e.target.value);
                        if (status === "error") setStatus("idle");
                      }}
                      className="w-full pl-11 pr-4 py-3.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-2xl text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-brand)] focus:ring-1 focus:ring-[var(--color-brand)] transition-all"
                      placeholder="••••••••"
                      required
                      disabled={status === "loading"}
                    />
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={status === "loading"}
                className="w-full flex items-center justify-center gap-2 py-4 rounded-2xl bg-[var(--color-brand)] text-white font-bold hover:bg-[var(--color-brand-light)] hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-70 disabled:hover:scale-100 shadow-xl shadow-[var(--color-brand)]/20 mt-6"
              >
                {status === "loading" ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : (
                  <>
                    Save Password <ArrowRight size={18} />
                  </>
                )}
              </button>
            </>
          )}
        </form>

        <div className="mt-8 text-center border-t border-[var(--color-border)] pt-6">
          <Link to="/login" className="text-sm font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors">
            Remembered your password? Log in
          </Link>
        </div>
      </motion.div>
    </motion.div>
  );
}
