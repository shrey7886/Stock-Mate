import { NavLink, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import {
  LayoutDashboard,
  MessageCircle,
  Link2,
  LogOut,
  Menu,
  X,
  Sparkles,
  Moon,
  Sun
} from "lucide-react";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import FoleoLogo from "./FoleoLogo";

const NAV_ITEMS = [
  { to: "/dashboard", icon: LayoutDashboard, label: "Portfolio" },
  { to: "/chat", icon: MessageCircle, label: "Assistant" },
  { to: "/broker", icon: Link2, label: "Broker" },
];

function SidebarLink({ to, icon: Icon, label, onClick }) {
  return (
    <NavLink
      to={to}
      onClick={onClick}
      className={({ isActive }) =>
        `group flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-medium transition-all duration-300 ${isActive
          ? "bg-[var(--color-surface-overlay)] text-[var(--color-text-primary)] shadow-sm"
          : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface-raised)]"
        }`
      }
    >
      <Icon size={17} strokeWidth={1.8} />
      <span>{label}</span>
    </NavLink>
  );
}

export default function AppShell({ children }) {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  const isDark = theme === "dark";

  const sidebar = (
    <div className="flex flex-col h-full bg-[var(--color-surface)]">
      {/* Logo */}
      <div className="px-6 py-8 flex items-center">
        <FoleoLogo size="lg" className="drop-shadow-sm" />
      </div>

      {/* Nav links */}
      <nav className="flex-1 px-4 space-y-1 mt-4">
        <p className="px-4 mb-4 text-[10px] font-bold uppercase tracking-[2px] text-[var(--color-text-muted)]">Menu</p>
        {NAV_ITEMS.map((item) => (
          <SidebarLink
            key={item.to}
            {...item}
            onClick={() => setMobileOpen(false)}
          />
        ))}
      </nav>

      {/* User section */}
      <div className="px-5 py-5 border-t border-[var(--color-border)]">
        <div className="flex items-center justify-between mb-4">
          <span className="text-[11px] font-bold uppercase tracking-[2px] text-[var(--color-text-muted)] ml-2">Theme</span>
          <button
            onClick={toggleTheme}
            className="p-2 rounded-xl bg-[var(--color-surface-overlay)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-border)] transition-all duration-300 shadow-sm border border-[var(--color-border)] flex items-center gap-2"
            title="Toggle Theme"
          >
            {isDark ? <Sun size={14} /> : <Moon size={14} />}
          </button>
        </div>
        <div className="flex items-center justify-between bg-[var(--color-surface-card)] rounded-2xl p-3 border border-[var(--color-border-subtle)]">
          <div className="min-w-0 flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-[var(--color-surface-overlay)] flex items-center justify-center text-sm font-semibold text-[var(--color-text-primary)] uppercase shrink-0">
              {(user?.email || "U").charAt(0)}
            </div>
            <p className="text-sm font-medium text-[var(--color-text-secondary)] truncate max-w-[100px]">
              {user?.email || user?.userId || "User"}
            </p>
          </div>
          <button
            onClick={handleLogout}
            className="p-2.5 text-[var(--color-text-muted)] hover:text-red-500 transition-colors duration-300 rounded-xl hover:bg-red-500/10"
            title="Logout"
          >
            <LogOut size={16} />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-[var(--color-surface)] text-[var(--color-text-primary)]">
      {/* Desktop sidebar */}
      <aside className="hidden lg:flex w-72 flex-col border-r border-[var(--color-border)] shadow-[var(--color-border)_1px_0_0_0]">
        {sidebar}
      </aside>

      {/* Mobile overlay */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/40 backdrop-blur-md z-40 lg:hidden"
              onClick={() => setMobileOpen(false)}
            />
            <motion.aside
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", damping: 28, stiffness: 320 }}
              className="fixed left-0 top-0 bottom-0 w-72 z-50 bg-[var(--color-surface)] border-r border-[var(--color-border)] lg:hidden shadow-2xl"
            >
              {sidebar}
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 bg-[var(--color-surface)]">
        {/* Top bar (mobile) */}
        <header className="lg:hidden flex items-center justify-between px-5 py-4 border-b border-[var(--color-border)] bg-[var(--color-surface)]/80 backdrop-blur-xl sticky top-0 z-30">
          <button
            onClick={() => setMobileOpen(true)}
            className="p-2.5 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors rounded-xl bg-[var(--color-surface-overlay)]"
          >
            <Menu size={20} />
          </button>
          <div className="flex items-center mt-1">
            <FoleoLogo size="sm" className="drop-shadow-sm" />
          </div>
          <button
            onClick={toggleTheme}
            className="p-2.5 text-[var(--color-text-secondary)] transition-colors rounded-xl bg-[var(--color-surface-overlay)]"
          >
            {isDark ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto relative z-10">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
              className="min-h-full"
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
