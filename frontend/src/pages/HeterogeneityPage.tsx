import { Segmented, Tag } from 'antd';
import { useState } from 'react';
import { Chart, chartGrid, chartText, Panel } from '../components/ChartPanel';
import { MetricCard } from '../components/MetricCard';
import { PageHeader } from '../components/PageHeader';
import { domainAccuracy, domains, featureDistances } from '../mock/data';

export function HeterogeneityPage() {
  const [view, setView] = useState('处理后');
  const heatmap = featureDistances.flatMap((row, x) => row.map((value, y) => [x, y, view === '处理后' ? Number((value * .52).toFixed(2)) : value]));
  const scatterData = domains.flatMap((domain, domainIndex) => Array.from({ length: 28 }, (_, index) => {
    const compact = view === '处理后' ? .48 : 1;
    const centerX = [15, 38, 62, 84][domainIndex];
    const centerY = [62, 28, 68, 34][domainIndex];
    return [centerX + Math.sin(index * 2.1 + domainIndex) * 14 * compact, centerY + Math.cos(index * 1.6) * 12 * compact, domainIndex];
  }));
  const scatterOption = {
    tooltip: { formatter: (params: { value: number[] }) => `${domains[params.value[2]].name}<br/>统一特征投影` },
    grid: { left: 34, right: 20, top: 20, bottom: 28 },
    xAxis: { min: 0, max: 100, axisLabel: { show: false }, axisLine: { lineStyle: { color: chartGrid } }, splitLine: { lineStyle: { color: chartGrid } } },
    yAxis: { min: 0, max: 100, axisLabel: { show: false }, axisLine: { lineStyle: { color: chartGrid } }, splitLine: { lineStyle: { color: chartGrid } } },
    series: domains.map((domain, domainIndex) => ({ name: domain.name, type: 'scatter', data: scatterData.filter((point) => point[2] === domainIndex), symbolSize: 9, itemStyle: { color: domain.color, opacity: .75 } })),
  };
  const heatmapOption = {
    tooltip: { position: 'top' },
    grid: { left: 85, right: 20, top: 12, bottom: 70 },
    xAxis: { type: 'category', data: domains.map((d) => d.shortName), axisLabel: { color: chartText, rotate: 22 }, axisLine: { lineStyle: { color: chartGrid } } },
    yAxis: { type: 'category', data: domains.map((d) => d.shortName), axisLabel: { color: chartText }, axisLine: { lineStyle: { color: chartGrid } } },
    visualMap: { min: 0, max: 1, calculable: false, orient: 'horizontal', left: 'center', bottom: 0, textStyle: { color: chartText }, inRange: { color: ['#10283b', '#317da8', '#44d8ff', '#ffbd52'] } },
    series: [{ type: 'heatmap', data: heatmap, label: { show: true, color: '#e8f5ff' }, itemStyle: { borderColor: '#0c1a29', borderWidth: 3 } }],
  };
  const barOption = {
    tooltip: { trigger: 'axis' }, legend: { data: ['处理前', '处理后'], textStyle: { color: chartText } },
    grid: { left: 50, right: 20, top: 38, bottom: 55 },
    xAxis: { type: 'category', data: domainAccuracy.map((d) => d.name), axisLabel: { color: chartText, rotate: 18 }, axisLine: { lineStyle: { color: chartGrid } } },
    yAxis: { type: 'value', min: 40, max: 100, axisLabel: { color: chartText, formatter: '{value}%' }, splitLine: { lineStyle: { color: chartGrid } } },
    series: [{ name: '处理前', type: 'bar', data: domainAccuracy.map((d) => d.before), itemStyle: { color: '#344f68', borderRadius: [3, 3, 0, 0] } }, { name: '处理后', type: 'bar', data: domainAccuracy.map((d) => d.after), itemStyle: { color: '#31d6b0', borderRadius: [3, 3, 0, 0] } }],
  };
  return <div className="page">
    <PageHeader eyebrow="HETEROGENEITY ANALYSIS" title="跨域异构态势分析" description="同时观察域间特征空间差异与域内数据分布不均，呈现两阶段异构处理前后的变化。" actions={<Segmented options={['处理前', '处理后']} value={view} onChange={(value) => setView(String(value))} />} />
    <div className="metrics-grid four"><MetricCard label="域间平均距离" value={view === '处理后' ? '0.35' : '0.67'} delta="特征空间一致性提升" /><MetricCard label="最差域准确率" value={view === '处理后' ? '80.7' : '59.3'} suffix="%" delta="+21.4%" tone="green" /><MetricCard label="类别覆盖率" value="94.2" suffix="%" delta="第一阶段扩充后" tone="violet" /><MetricCard label="节点公平性" value="0.88" delta="域内差距下降 36%" tone="amber" /></div>
    <div className="analysis-grid">
      <Panel title="统一特征空间投影" subtitle={`${view} · 合成二维投影`} extra={<div className="chart-legend">{domains.map((d) => <span key={d.id}><i style={{ background: d.color }} />{d.shortName}</span>)}</div>}><Chart option={scatterOption} height={340} /></Panel>
      <Panel title="域间特征距离矩阵" subtitle="数值越高表示差异越明显"><Chart option={heatmapOption} height={340} /></Panel>
    </div>
    <div className="analysis-bottom-grid">
      <Panel title="各域模型效果变化" subtitle="异构解决方案启用前后"><Chart option={barOption} height={300} /></Panel>
      <Panel title="域内数据分布" subtitle="节点样本量与缺失类别">
        <div className="distribution-list">{domains.map((d, index) => <div key={d.id}><div className="distribution-head"><span><i style={{ background: d.color }} />{d.name}</span><Tag>{d.modality}</Tag></div><div className="stacked-bar">{[20 + index * 3, 27 - index, 13 + index * 2, 24 - index * 2, 16].map((value, i) => <span key={i} style={{ width: `${value}%`, background: `${d.color}${['ee','bb','88','66','44'][i]}` }} />)}</div><small>最大类别占比 {34 + index * 6}% · 缺失类别 {index + 1} 个</small></div>)}</div>
      </Panel>
    </div>
  </div>;
}
