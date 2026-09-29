import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  icon,
  className = '',
  disabled,
  ...props
}) => {
  const baseStyle =
    'inline-flex items-center justify-center font-medium rounded-xl transition-all duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-[var(--accent)] active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:active:scale-100 select-none cursor-pointer';

  const sizeStyles = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-5 py-2.5 text-base',
  };

  const variantStyles = {
    primary: 'bg-gradient-to-r from-[#0f766e] to-[#14b8a6] hover:brightness-110 text-[#041214] font-bold shadow-[0_0_16px_rgba(20,184,166,0.35)] border border-transparent',
    secondary: 'bg-[var(--surface-2)] hover:bg-[rgba(20,184,166,0.08)] text-[var(--text)] hover:border-[var(--border-strong)] border border-[var(--border)]',
    outline: 'bg-[var(--surface-2)] hover:bg-[rgba(20,184,166,0.08)] text-[var(--text)] border border-[var(--border)] hover:border-[var(--border-strong)]',
    ghost: 'bg-transparent hover:bg-[rgba(20,184,166,0.08)] text-[var(--mint)] hover:text-[var(--accent-bright)] border border-transparent',
  };

  return (
    <button
      className={`${baseStyle} ${sizeStyles[size]} ${variantStyles[variant]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {icon && <span className="mr-2 inline-flex items-center">{icon}</span>}
      {children}
    </button>
  );
};
