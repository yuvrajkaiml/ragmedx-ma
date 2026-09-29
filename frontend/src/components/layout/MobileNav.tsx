import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

export const MobileNav: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    setIsOpen(false);
    await signOut();
    navigate('/login');
  };

  const initials = user?.name
    ? user.name.split(' ').map((w) => w[0]).join('').toUpperCase().slice(0, 2)
    : user?.isGuest ? 'G' : '?';

  const navItems = [
    { to: '/ask', label: 'Ask' },
    { to: '/history', label: 'History' },
    { to: '/evidence', label: 'Evidence Library' },
    { to: '/settings', label: 'Settings' },
  ];

  return (
    <header className="bg-[var(--bg-2)] border-b border-[var(--border)] md:hidden sticky top-0 z-30">
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-full bg-[var(--accent)] flex items-center justify-center text-[#050b0e]">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
          </div>
          <span className="font-bold text-[var(--text)] text-sm">RAGMedX-MA</span>
        </div>

        <div className="flex items-center space-x-2">
          {/* User avatar pill */}
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-[var(--accent-deep)] to-[var(--accent)] flex items-center justify-center text-[#041214] font-bold text-[10px]">
            {initials}
          </div>

          <button
            onClick={() => setIsOpen(!isOpen)}
            className="p-2 rounded-lg text-[var(--text-muted)] hover:bg-[var(--surface-2)] focus:outline-none"
            aria-label="Toggle Navigation"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {isOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {isOpen && (
        <nav className="px-4 pb-3 pt-1 space-y-1 border-t border-[var(--border)] bg-[var(--bg-2)]">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={() => setIsOpen(false)}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-lg text-sm font-medium ${
                  isActive ? 'bg-[var(--surface-2)] text-[var(--accent-bright)] font-bold' : 'text-[var(--text-muted)] hover:bg-[var(--surface)] hover:text-[var(--text)]'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}

          {/* User info + logout */}
          <div className="pt-2 mt-2 border-t border-[var(--border)]">
            <div className="flex items-center space-x-2 px-3 py-1.5">
              <span className="text-xs text-[var(--text-muted)]">
                {user?.isGuest ? 'Guest' : user?.name || 'User'}
              </span>
            </div>
            <button
              onClick={handleLogout}
              className="w-full flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium text-[var(--text-muted)] hover:text-[#f87171] hover:bg-[rgba(248,113,113,0.08)] transition-colors cursor-pointer"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
              <span>Sign out</span>
            </button>
          </div>
        </nav>
      )}
    </header>
  );
};
