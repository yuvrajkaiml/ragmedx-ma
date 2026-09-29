import React from 'react';
import { Sidebar } from './Sidebar';
import { MobileNav } from './MobileNav';
import { Disclaimer } from '../common/Disclaimer';
import { PageTransition } from './PageTransition';

interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  return (
    <div className="flex flex-col md:flex-row min-h-screen bg-[var(--bg)] text-[var(--text)] font-sans antialiased relative overflow-x-hidden">
      {/* Aurora Background (Fixed Layer, 3 Deep Teal/Cyan/Mint Blobs, blur 90px, opacity 0.35) */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0" aria-hidden="true">
        <div className="absolute top-[-15%] left-[-10%] w-[40vmax] h-[40vmax] rounded-full bg-[#0f766e] blur-[90px] opacity-35 animate-aurora-spec-1" />
        <div className="absolute top-[25%] right-[-15%] w-[40vmax] h-[40vmax] rounded-full bg-[#0e7490] blur-[90px] opacity-35 animate-aurora-spec-2" />
        <div className="absolute bottom-[-15%] left-[15%] w-[40vmax] h-[40vmax] rounded-full bg-[#115e59] blur-[90px] opacity-35 animate-aurora-spec-3" />
      </div>

      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 relative z-10">
        <MobileNav />
        <main className="flex-1 p-4 md:p-8 max-w-7xl mx-auto w-full">
          <PageTransition>{children}</PageTransition>
        </main>
        <Disclaimer variant="footer" />
      </div>
    </div>
  );
};
