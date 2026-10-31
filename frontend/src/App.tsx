import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QAProvider } from './context/QAContext';
import { AuthProvider } from './context/AuthContext';
import { ProtectedRoute } from './components/auth/ProtectedRoute';
import { AppShell } from './components/layout/AppShell';
import { AskPage } from './pages/AskPage';
import { HistoryPage } from './pages/HistoryPage';
import { EvidencePage } from './pages/EvidencePage';
import { SettingsPage } from './pages/SettingsPage';
import { AuthPage } from './pages/AuthPage';

export default function App() {
  return (
    <AuthProvider>
      <QAProvider>
        <BrowserRouter>
          <Routes>
            {/* Public auth routes */}
            <Route path="/login" element={<AuthPage />} />
            <Route path="/signup" element={<AuthPage />} />

            {/* Protected app routes */}
            <Route
              path="/*"
              element={
                <ProtectedRoute>
                  <AppShell>
                    <Routes>
                      <Route path="/" element={<Navigate to="/ask" replace />} />
                      <Route path="/ask" element={<AskPage />} />
                      <Route path="/history" element={<HistoryPage />} />
                      <Route path="/evidence" element={<EvidencePage />} />
                      <Route path="/settings" element={<SettingsPage />} />
                      <Route path="*" element={<Navigate to="/ask" replace />} />
                    </Routes>
                  </AppShell>
                </ProtectedRoute>
              }
            />
          </Routes>
        </BrowserRouter>
      </QAProvider>
    </AuthProvider>
  );
}
