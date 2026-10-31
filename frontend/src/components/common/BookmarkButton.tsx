import React, { useState } from 'react';

interface BookmarkButtonProps {
  initialBookmarked?: boolean;
  onToggleBookmark?: (isBookmarked: boolean) => void;
  className?: string;
}

export const BookmarkButton: React.FC<BookmarkButtonProps> = ({
  initialBookmarked = false,
  onToggleBookmark,
  className = '',
}) => {
  const [isBookmarked, setIsBookmarked] = useState<boolean>(initialBookmarked);

  const handleClick = () => {
    const nextState = !isBookmarked;
    setIsBookmarked(nextState);
    if (onToggleBookmark) {
      onToggleBookmark(nextState);
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      aria-pressed={isBookmarked}
      aria-label={isBookmarked ? 'Remove bookmark' : 'Bookmark answer'}
      title={isBookmarked ? 'Bookmarked' : 'Bookmark answer'}
      className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer select-none ${
        isBookmarked
          ? 'bg-[var(--accent)] text-[#041214] shadow-xs font-semibold'
          : 'bg-[var(--surface-2)] hover:bg-[var(--surface)] text-[var(--text-muted)] hover:text-[var(--text)] border border-[var(--border)]'
      } ${className}`}
    >
      <svg
        className="w-3.5 h-3.5"
        fill={isBookmarked ? 'currentColor' : 'none'}
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
      </svg>
      <span>{isBookmarked ? 'Bookmarked' : 'Bookmark'}</span>
    </button>
  );
};
