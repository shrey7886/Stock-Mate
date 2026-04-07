import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "./context/AuthContext";
import AppShell from "./components/AppShell";
import LandingPage from "./pages/LandingPage";
import LoginPage from "./pages/LoginPage";
import DashboardPage from "./pages/DashboardPage";
import ChatPage from "./pages/ChatPage";
import BrokerPage from "./pages/BrokerPage";
import ResetPasswordPage from "./pages/ResetPasswordPage";

function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return null;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return children;
}

function PublicOnly({ children }) {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return null;
  if (isAuthenticated) return <Navigate to="/dashboard" replace />;
  return children;
}

export default function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/" element={<PublicOnly><LandingPage /></PublicOnly>} />
      <Route path="/login" element={<PublicOnly><LoginPage /></PublicOnly>} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />

      {/* Protected — wrapped in AppShell */}
      <Route
        path="/dashboard"
        element={<ProtectedRoute><AppShell><DashboardPage /></AppShell></ProtectedRoute>}
      />
      <Route
        path="/chat"
        element={<ProtectedRoute><AppShell><ChatPage /></AppShell></ProtectedRoute>}
      />
      <Route
        path="/broker"
        element={<ProtectedRoute><AppShell><BrokerPage /></AppShell></ProtectedRoute>}
      />

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
