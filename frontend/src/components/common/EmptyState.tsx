import React from 'react';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center bg-[var(--surface)] rounded-xl border border-[var(--border)] shadow-[var(--card-shadow)] my-4">
      {icon && <div className="p-3 mb-3 bg-[var(--surface-2)] text-[var(--accent)] rounded-full border border-[var(--border)]">{icon}</div>}
      <h3 className="text-base font-semibold text-[var(--text)] mb-1">{title}</h3>
      <p className="text-sm text-[var(--text-muted)] max-w-md mb-4">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
};
