import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { HistoryFilters } from '../components/history/HistoryFilters';
import { HistoryList } from '../components/history/HistoryList';
import { SessionDetailModal } from '../components/history/SessionDetailModal';
import { HistoryFilterType, HistorySessionItem } from '../types/history';
import { useQA } from '../context/QAContext';

export const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const { historySessions, reopenSession } = useQA();
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState<HistoryFilterType>('All');
  const [selectedSession, setSelectedSession] = useState<HistorySessionItem | null>(null);

  const filteredSessions = historySessions.filter((session) => {
    const matchesSearch =
      session.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      session.answerSummary.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter =
      activeFilter === 'All' || session.state.toLowerCase() === activeFilter.toLowerCase();

    return matchesSearch && matchesFilter;
  });

  const handleReopenSession = (session: HistorySessionItem) => {
    reopenSession(session);
    navigate('/ask');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-3">
        <div className="w-9 h-9 rounded-xl bg-[var(--surface-2)] text-[var(--accent)] flex items-center justify-center border border-[var(--border)] shrink-0">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <div>
          <h2 className="text-2xl font-bold text-[var(--text)] tracking-tight">History</h2>
          <p className="text-xs text-[var(--text-muted)]">
            Review past questions and their adjudicated answers.
          </p>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <HistoryFilters
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        activeFilter={activeFilter}
        onFilterChange={setActiveFilter}
      />

      {/* Sessions List or Empty State */}
      <HistoryList
        sessions={filteredSessions}
        onSelectSession={(session) => setSelectedSession(session)}
      />

      {/* Session Detail Modal */}
      <SessionDetailModal
        session={selectedSession}
        isOpen={Boolean(selectedSession)}
        onClose={() => setSelectedSession(null)}
        onReopen={handleReopenSession}
      />
    </div>
  );
};
