import type { ReactNode } from 'react';

export function MetricCard({ label, value, suffix, delta, icon, tone = 'cyan' }: {
  label: string;
  value: string | number;
  suffix?: string;
  delta?: string;
  icon?: ReactNode;
  tone?: 'cyan' | 'green' | 'amber' | 'red' | 'violet';
}) {
  return (
    <div className={`metric-card metric-${tone}`}>
      <div className="metric-top"><span>{label}</span>{icon && <i>{icon}</i>}</div>
      <div className="metric-value">{value}<small>{suffix}</small></div>
      {delta && <div className="metric-delta">{delta}</div>}
      <div className="metric-glow" />
    </div>
  );
}
