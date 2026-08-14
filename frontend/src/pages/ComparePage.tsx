import { ArrowDownOutlined, ArrowUpOutlined, SwapOutlined } from '@ant-design/icons';
import { Select, Tag } from 'antd';
import { Chart, chartGrid, chartText, Panel } from '../components/ChartPanel';
import { PageHeader } from '../components/PageHeader';
import { domainAccuracy, roundMetrics } from '../mock/data';

const compareTrend = {
  tooltip: { trigger: 'axis' }, legend: { data: ['无防御：攻击成功率', '有防御：攻击成功率', '有防御：干净准确率'], textStyle: { color: chartText } },
  grid: { left: 48, right: 24, top: 45, bottom: 28 },
  xAxis: { type: 'category', data: roundMetrics.map((d) => d.round), axisLabel: { color: chartText }, axisLine: { lineStyle: { color: chartGrid } } },
  yAxis: { type: 'value', min: 0, max: 100, axisLabel: { color: chartText, formatter: '{value}%' }, splitLine: { lineStyle: { color: chartGrid } } },
  series: [
    { name: '无防御：攻击成功率', type: 'line', symbol: 'none', data: roundMetrics.map((_, i) => Math.min(91, 17 + i * 2.7)), lineStyle: { color: '#f0526d', width: 2 } },
    { name: '有防御：攻击成功率', type: 'line', symbol: 'none', data: roundMetrics.map((d) => d.attackSuccess), lineStyle: { color: '#29e3ae', width: 3 } },
    { name: '有防御：干净准确率', type: 'line', symbol: 'none', data: roundMetrics.map((d) => d.accuracy), lineStyle: { color: '#44d8ff', width: 2, type: 'dashed' } },
  ],
};

export function ComparePage() {
  return <div className="page">
    <PageHeader eyebrow="EXPERIMENT COMPARISON" title="攻防与保护效果对照" description="将独立实验按相同场景、随机种子和训练参数对齐，比较收益、代价和域级差异。" />
    <div className="compare-selector"><div><small>实验 A · 基准</small><Select value="后门攻击 / 无防御" options={[{ value: '后门攻击 / 无防御' }]} /><Tag color="error">RUN-0821-A</Tag></div><SwapOutlined /><div><small>实验 B · 对照</small><Select value="后门攻击 / 两阶段防御" options={[{ value: '后门攻击 / 两阶段防御' }]} /><Tag color="green">RUN-0821-B</Tag></div><div className="compare-context"><small>共同条件</small><b>4 域 · 20 节点 · 30 轮</b><span>随机种子 20260815</span></div></div>
    <div className="compare-kpis">{[
      ['攻击成功率','78.2%','31.4%','-46.8%','good'],['干净准确率','86.3%','84.9%','-1.4%','cost'],['恶意检出率','—','91.7%','+91.7%','good'],['正常节点误报率','—','4.1%','+4.1%','cost'],['最差域准确率','61.8%','80.7%','+18.9%','good'],
    ].map(([label,a,b,change,tone]) => <div key={label}><span>{label}</span><div><em>{a}</em><i>→</i><b>{b}</b></div><small className={tone === 'good' ? 'good' : 'cost'}>{String(change).startsWith('-') ? <ArrowDownOutlined /> : <ArrowUpOutlined />}{change}</small></div>)}</div>
    <div className="compare-chart-grid"><Panel title="攻击效果与模型可用性" subtitle="两个独立运行记录，按轮次对齐"><Chart option={compareTrend} height={340} /></Panel><Panel title="域级收益" subtitle="各域防御后准确率提升"><div className="domain-benefits">{domainAccuracy.map((domain, index) => <div key={domain.name}><div><span>{domain.name}</span><b>+{(domain.after-domain.before).toFixed(1)}%</b></div><div className="benefit-track"><i style={{ width: `${domain.before}%` }} /><em style={{ width: `${domain.after}%` }} /></div><small>无防御 {domain.before}%　·　有防御 {domain.after}%</small><Tag color={index === 3 ? 'gold' : 'cyan'}>{index === 3 ? '提升最大' : '稳定改善'}</Tag></div>)}</div></Panel></div>
    <Panel title="阶段贡献拆解" subtitle="分别观察特征统计阶段与正常训练阶段防御"><div className="stage-contribution"><div><span>仅正常训练阶段防御</span><b>攻击成功率 45.8%</b><ProgressBar value={54} tone="amber" /></div><div><span>仅特征统计阶段防御</span><b>攻击成功率 39.2%</b><ProgressBar value={63} tone="violet" /></div><div className="highlight"><span>两阶段联合防御</span><b>攻击成功率 31.4%</b><ProgressBar value={76} tone="green" /></div></div></Panel>
  </div>;
}

function ProgressBar({ value, tone }: { value: number; tone: string }) { return <div className={`simple-progress ${tone}`}><i style={{ width: `${value}%` }} /><span>防御有效度 {value}%</span></div>; }
