import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

/**
 * Route guard – renders children only when authenticated.
 * Unauthenticated visitors are sent to /login with a `from` state
 * so they can be redirected back after login.
 */
export const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    // Show a minimal loading state while session is being restored
    return (
      <div className="flex items-center justify-center min-h-screen bg-[var(--bg)]">
        <div className="w-10 h-10 rounded-full border-4 border-[var(--border)] border-t-[var(--accent)] animate-spin" />
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
};
