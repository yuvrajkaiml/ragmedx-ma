import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

export const Sidebar: React.FC = () => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await signOut();
    navigate('/login');
  };

  const initials = user?.name
    ? user.name.split(' ').map((w) => w[0]).join('').toUpperCase().slice(0, 2)
    : user?.isGuest ? 'G' : '?';
  const navItems = [
    {
      to: '/ask',
      label: 'Ask',
      subtitle: 'Ask a clinical question',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
        </svg>
      ),
    },
    {
      to: '/history',
      label: 'History',
      subtitle: 'Past questions & answers',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
    },
    {
      to: '/evidence',
      label: 'Evidence Library',
      subtitle: 'Cited sources & references',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 01-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
      ),
    },
    {
      to: '/settings',
      label: 'Settings',
      subtitle: 'Preferences & thresholds',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      ),
    },
  ];

  return (
    <aside className="w-64 bg-[var(--bg-2)] border-r border-[var(--border)] min-h-screen flex flex-col justify-between shrink-0 hidden md:flex select-none z-20">
      <div>
        {/* Brand Header with Spec Heartbeat */}
        <div className="p-5 flex items-center space-x-3 border-b border-[var(--border)] bg-[var(--bg-2)]">
          <div className="w-10 h-10 rounded-full bg-[var(--accent)] flex items-center justify-center text-[#050b0e] shrink-0 shadow-md animate-spec-heartbeat">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
          </div>
          <div>
            <div className="flex items-center space-x-1">
              <h1 className="font-bold text-[var(--text)] text-base leading-tight tracking-tight">RAGMedX-MA</h1>
              <svg className="w-4 h-4 text-[var(--accent)] animate-icon-float" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <p className="text-xs text-[var(--text-muted)] font-medium">Medical Evidence QA</p>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1.5" aria-label="Main Navigation">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `group relative flex items-start space-x-3 p-3 rounded-xl transition-all duration-200 ease-out focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] overflow-hidden ${
                  isActive
                    ? 'bg-[var(--surface-2)] text-[var(--accent-bright)] font-bold border-l-4 border-[var(--accent)] shadow-sm animate-active-sheen-spec'
                    : 'text-[var(--text-muted)] hover:bg-[var(--surface)] hover:text-[var(--text)] hover:translate-x-1 font-medium'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <div className={`mt-0.5 shrink-0 transition-all duration-200 ${isActive ? 'text-[var(--accent-bright)] scale-105' : 'text-[var(--text-faint)] group-hover:text-[var(--accent)] group-hover:scale-110'}`}>
                    {item.icon}
                  </div>
                  <div>
                    <div className={`text-sm leading-snug ${isActive ? 'text-[var(--accent-bright)] font-bold' : 'text-[var(--text)] font-semibold'}`}>{item.label}</div>
                    <div className={`text-xs ${isActive ? 'text-[var(--mint)]' : 'text-[var(--text-muted)] group-hover:text-[var(--text)]'} font-normal transition-colors duration-150`}>
                      {item.subtitle}
                    </div>
                  </div>
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* User Avatar & Logout */}
      <div className="p-3 m-3 bg-[var(--surface-2)] rounded-xl border border-[var(--border)] space-y-3">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-full bg-gradient-to-br from-[var(--accent-deep)] to-[var(--accent)] flex items-center justify-center text-[#041214] font-bold text-sm shrink-0 shadow-sm">
            {initials}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-[var(--text)] truncate">
              {user?.isGuest ? 'Guest' : user?.name || 'User'}
            </div>
            {user?.email && (
              <div className="text-[11px] text-[var(--text-faint)] truncate">{user.email}</div>
            )}
          </div>
        </div>

        <button
          onClick={handleLogout}
          className="w-full flex items-center justify-center space-x-2 py-2 rounded-lg text-xs font-medium text-[var(--text-muted)] hover:text-[#f87171] hover:bg-[rgba(248,113,113,0.08)] border border-[var(--border)] transition-all duration-200 cursor-pointer"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
          </svg>
          <span>Sign out</span>
        </button>
      </div>
    </aside>
  );
};
