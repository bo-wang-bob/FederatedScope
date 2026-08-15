import { EyeOutlined, LockOutlined, SafetyCertificateOutlined, ScanOutlined } from '@ant-design/icons';
import { Alert, Progress, Segmented, Table, Tag } from 'antd';
import { useState } from 'react';
import { Chart, chartGrid, chartText, Panel } from '../components/ChartPanel';
import { MetricCard } from '../components/MetricCard';
import { PageHeader } from '../components/PageHeader';
import { membershipMetrics, propertyMetrics, reconstructionMetrics } from '../mock/data';
import type { PrivacyMetric } from '../types';

type AttackView = '成员关系推断' | '属性推断' | '数据重建';

const scoreOption = {
  tooltip: { trigger: 'axis' }, legend: { data: ['成员样本', '非成员样本'], textStyle: { color: chartText } },
  grid: { left: 45, right: 20, top: 38, bottom: 35 },
  xAxis: { type: 'category', data: ['0-.1','.1-.2','.2-.3','.3-.4','.4-.5','.5-.6','.6-.7','.7-.8','.8-.9','.9-1'], axisLabel: { color: chartText }, axisLine: { lineStyle: { color: chartGrid } } },
  yAxis: { type: 'value', axisLabel: { color: chartText }, splitLine: { lineStyle: { color: chartGrid } } },
  series: [{ name: '成员样本', type: 'bar', data: [1,2,3,5,8,14,22,28,34,17], itemStyle: { color: '#44d8ff' } }, { name: '非成员样本', type: 'bar', data: [27,34,29,18,12,7,4,2,1,0], itemStyle: { color: '#8b7cff' } }],
};

const confusionOption = {
  tooltip: {}, grid: { left: 85, right: 25, top: 18, bottom: 45 },
  xAxis: { type: 'category', data: ['预测 A', '预测 B', '预测 C'], axisLabel: { color: chartText }, axisLine: { lineStyle: { color: chartGrid } } },
  yAxis: { type: 'category', data: ['真实 C', '真实 B', '真实 A'], axisLabel: { color: chartText }, axisLine: { lineStyle: { color: chartGrid } } },
  visualMap: { min: 0, max: 48, show: false, inRange: { color: ['#11283c', '#225c7b', '#44d8ff'] } },
  series: [{ type: 'heatmap', data: [[0,0,42],[1,0,8],[2,0,3],[0,1,6],[1,1,38],[2,1,10],[0,2,4],[1,2,7],[2,2,46]], label: { show: true, color: '#fff' }, itemStyle: { borderColor: '#0c1826', borderWidth: 4 } }],
};

function PrivacyMetricCards({ metrics }: { metrics: PrivacyMetric[] }) {
  return <div className="privacy-metric-grid">{metrics.map((metric) => {
    const format = (value: number) => `${value}${metric.unit ?? ''}`;
    const change = metric.after - metric.before;
    const beneficial = metric.lowerIsBetter ? change < 0 : change >= 0;
    return <div key={metric.name}><span>{metric.name}</span><div className="privacy-values"><b>{format(metric.before)}</b><i>→</i><strong>{format(metric.after)}</strong></div><small className={beneficial ? 'text-success' : 'text-warning'}>{change > 0 ? '+' : ''}{change.toFixed(metric.unit ? 1 : 2)}{metric.unit ?? ''}</small><em>保护前 / 保护后</em></div>;
  })}</div>;
}

export function PrivacyPage() {
  const [attack, setAttack] = useState<AttackView>('成员关系推断');
  const metrics = attack === '成员关系推断' ? membershipMetrics : attack === '属性推断' ? propertyMetrics : reconstructionMetrics;
  return <div className="page">
    <PageHeader eyebrow="PRIVACY ATTACK EFFECTS" title="隐私攻击效果评估" description="分别模拟三类隐私攻击，直观展示本地隐私保护前后的风险变化与模型可用性代价。" actions={<Tag color="green" icon={<SafetyCertificateOutlined />}>后门攻击已关闭</Tag>} />
    <div className="privacy-header-bar"><div className="privacy-observer"><div><EyeOutlined /></div><span><small>模拟攻击观察方</small><b>好奇聚合方</b><em>仅访问服务端可见参数与统计摘要</em></span></div><div className="privacy-separator" /><div><small>目标范围</small><b>4 个域 · 8 个目标节点</b></div><div><small>结果来源</small><Tag color="cyan">前端固定种子模拟</Tag></div><div><small>保护对照</small><b>独立实验 A / B</b></div></div>
    <Segmented block size="large" options={[{ label: <span><EyeOutlined /> 成员关系推断</span>, value: '成员关系推断' }, { label: <span><LockOutlined /> 属性推断</span>, value: '属性推断' }, { label: <span><ScanOutlined /> 数据重建</span>, value: '数据重建' }]} value={attack} onChange={(value) => setAttack(value as AttackView)} className="privacy-tabs" />
    <PrivacyMetricCards metrics={metrics} />
    {attack === '成员关系推断' && <MembershipContent />}
    {attack === '属性推断' && <PropertyContent />}
    {attack === '数据重建' && <ReconstructionContent />}
  </div>;
}

function MembershipContent() {
  const rows = Array.from({ length: 6 }, (_, index) => ({ id: `SAMPLE-${1024 + index * 37}`, truth: index % 3 !== 0, probability: [.91,.86,.72,.59,.34,.18][index] }));
  return <div className="privacy-content-grid"><Panel title="成员得分分布" subtitle="阈值 0.56 · 分离程度越高表示泄露风险越大"><Chart option={scoreOption} height={300} /></Panel><Panel title="目标样本判断" subtitle="匿名样本，仅用于效果演示"><Table rowKey="id" size="small" pagination={false} dataSource={rows} columns={[{ title: '样本编号', dataIndex: 'id' },{ title: '真实成员', dataIndex: 'truth', render: (v: boolean) => v ? '是' : '否' },{ title: '推断概率', dataIndex: 'probability', render: (v: number) => <Progress percent={v * 100} size="small" strokeColor={v > .56 ? '#ffbd52' : '#44d8ff'} /> },{ title: '结果', render: (_: unknown, row: typeof rows[number]) => <Tag color={(row.probability > .56) === row.truth ? 'success' : 'error'}>{(row.probability > .56) === row.truth ? '正确' : '错误'}</Tag> }]} /></Panel></div>;
}

function PropertyContent() {
  const rows = ['OH-DT-C02','OH-DT-C05','OH-TS-C03','OH-ED-C01','OH-ED-C04','OH-FR-C02'].map((id, index) => ({ id, truth: `任务类型 ${['A','B','C'][index % 3]}`, prediction: `任务类型 ${['A','B','C','B','A','C'][index]}`, confidence: [91,82,78,66,54,48][index] }));
  return <div className="privacy-content-grid"><Panel title="合成属性混淆矩阵" subtitle="仅使用任务类型 A / B / C 等抽象属性"><Chart option={confusionOption} height={300} /></Panel><Panel title="节点属性推断结果" subtitle="保护前模拟结果"><Table rowKey="id" size="small" pagination={false} dataSource={rows} columns={[{ title: '目标节点', dataIndex: 'id' },{ title: '真实属性', dataIndex: 'truth' },{ title: '预测属性', dataIndex: 'prediction' },{ title: '置信度', dataIndex: 'confidence', render: (v: number) => `${v}%` },{ title: '结果', render: (_: unknown, row: typeof rows[number]) => <Tag color={row.truth === row.prediction ? 'warning' : 'success'}>{row.truth === row.prediction ? '泄露' : '未命中'}</Tag> }]} /></Panel></div>;
}

function ReconstructionContent() {
  return <><Alert type="info" showIcon message="展示内容为合成抽象特征，不包含真实军事图像或可复用的攻击步骤。" /><div className="reconstruction-grid"><FeatureImage title="合成参考特征" variant="reference" score="原始演示样本" /><FeatureImage title="未保护重建效果" variant="recovered" score="相似度 0.81" /><FeatureImage title="启用保护后" variant="protected" score="相似度 0.27" /><Panel title="重建质量变化" subtitle="保护后信息可辨识度显著降低"><div className="quality-list">{[['特征相似度',81,27],['结构相似度',76,21],['标签恢复率',68,17]].map(([name,before,after]) => <div key={String(name)}><span>{name}</span><div><Progress percent={Number(before)} showInfo={false} strokeColor="#ffbd52" /><b>{before}%</b></div><div><Progress percent={Number(after)} showInfo={false} strokeColor="#29e3ae" /><b>{after}%</b></div></div>)}</div><div className="quality-legend"><span><i className="before" />未保护</span><span><i className="after" />已保护</span></div></Panel></div></>;
}

function FeatureImage({ title, variant, score }: { title: string; variant: string; score: string }) {
  return <Panel title={title} subtitle={score}><div className={`feature-image ${variant}`}><div className="feature-grid-art">{Array.from({ length: 64 }, (_, index) => <i key={index} style={{ opacity: variant === 'protected' ? .12 + ((index * 17) % 20) / 100 : .2 + ((index * 13) % 70) / 100 }} />)}</div><div className="feature-target"><span /><span /><span /></div></div></Panel>;
}
