import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import type { ReactNode } from 'react';

export function Panel({ title, subtitle, extra, children, className = '' }: {
  title?: string;
  subtitle?: string;
  extra?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || extra) && (
        <div className="panel-head">
          <div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>
          {extra}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function Chart({ option, height = 260, appearance }: { option: object; height?: number; appearance?: 'dark' }) {
  return <ReactECharts theme={appearance} option={option as EChartsOption} style={{ height }} opts={{ renderer: 'canvas' }} />;
}

export const chartText = '#9fb2c8';
export const chartGrid = 'rgba(143, 183, 221, .12)';
