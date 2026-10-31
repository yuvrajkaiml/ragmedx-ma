import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { rehydrateRefreshSchedule } from './services/authService';

// Fix #8: re-arm the proactive token refresh timer on every page load so an
// already-authenticated user does not silently hit 401 mid-session.
rehydrateRefreshSchedule();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
