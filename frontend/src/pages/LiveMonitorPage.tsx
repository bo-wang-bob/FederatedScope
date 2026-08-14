import { ClockCircleOutlined, PauseOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { Button, Progress, Segmented, Space, Table, Tag, Timeline } from 'antd';
import { Chart, chartGrid, chartText, Panel } from '../components/ChartPanel';
import { MetricCard } from '../components/MetricCard';
import { PageHeader } from '../components/PageHeader';
import { FederationTopology } from '../components/topology/FederationTopology';
import { domains, events, nodeRows, roundMetrics } from '../mock/data';
import { phases, useAppStore } from '../store/useAppStore';

const trendOption = {
  tooltip: { trigger: 'axis' },
  legend: { data: ['全局准确率', '更新范数', '攻击成功率'], textStyle: { color: chartText } },
  grid: { left: 44, right: 44, top: 38, bottom: 25 },
  xAxis: { type: 'category', data: roundMetrics.map((item) => item.round), axisLabel: { color: chartText }, axisLine: { lineStyle: { color: chartGrid } } },
  yAxis: [{ type: 'value', axisLabel: { color: chartText, formatter: '{value}%' }, splitLine: { lineStyle: { color: chartGrid } } }, { type: 'value', axisLabel: { color: chartText }, splitLine: { show: false } }],
  series: [
    { name: '全局准确率', type: 'line', smooth: true, symbol: 'none', data: roundMetrics.map((d) => d.accuracy), lineStyle: { color: '#44d8ff', width: 3 } },
    { name: '更新范数', type: 'line', yAxisIndex: 1, symbol: 'none', data: roundMetrics.map((_, i) => (1.8 + Math.sin(i) * .35).toFixed(2)), lineStyle: { color: '#8b7cff' } },
    { name: '攻击成功率', type: 'line', smooth: true, symbol: 'none', data: roundMetrics.map((d) => d.attackSuccess), lineStyle: { color: '#ffbd52', type: 'dashed' } },
  ],
};

export function LiveMonitorPage() {
  const { phaseIndex, running, toggleRunning, setPhaseIndex, round } = useAppStore();
  return <div className="page">
    <PageHeader eyebrow="LIVE EXPERIMENT MONITOR" title="实验运行监控" description="将后端单机训练轮次投影为中央、域级和节点级事件，实时跟踪训练与防御状态。" actions={<Space><Tag color="processing"><span className="live-dot" /> LIVE</Tag><Button icon={running ? <PauseOutlined /> : <PlayCircleOutlined />} onClick={toggleRunning}>{running ? '暂停' : '继续'}</Button></Space>} />
    <div className="stage-strip">{phases.map((phase, index) => <button key={phase} className={index === phaseIndex ? 'active' : index < phaseIndex ? 'done' : ''} onClick={() => setPhaseIndex(index)}><i>{index < phaseIndex ? '✓' : index + 1}</i><span>{phase}</span></button>)}</div>
    <div className="metrics-grid five"><MetricCard label="当前轮次" value={round} suffix=" / 30" delta="预计剩余 06:42" /><MetricCard label="活跃节点" value="16" suffix=" / 20" delta="3 等待 · 1 过滤" tone="green" /><MetricCard label="域级上传" value="3" suffix=" / 4" delta="指挥决策域等待中" tone="violet" /><MetricCard label="全局准确率" value="85.7" suffix="%" delta="本轮 +0.8%" tone="cyan" /><MetricCard label="攻击成功率" value="31.4" suffix="%" delta="防御后下降 42.7%" tone="amber" /></div>
    <div className="live-grid">
      <Panel title="分层数据流" subtitle={`当前：${phases[phaseIndex]} · 第 ${round} 轮`} className="live-topology"><FederationTopology compact /></Panel>
      <Panel title="实时事件" subtitle="最近的训练与安全事件" extra={<ClockCircleOutlined />}>
        <Timeline className="event-timeline" items={events.map((event) => ({ color: event.level === 'danger' ? 'red' : event.level === 'warning' ? 'orange' : event.level === 'success' ? 'green' : 'blue', children: <div className="event-item"><span>{event.time} · {event.source}</span><p>{event.message}</p></div> }))} />
      </Panel>
    </div>
    <div className="live-bottom-grid">
      <Panel title="训练与攻击趋势" subtitle="轮次级真实指标 / 模拟效果"><Chart option={trendOption} height={300} /></Panel>
      <Panel title="域级上传进度" subtitle="域内节点先聚合，再上传中央服务器">
        <div className="upload-list">{domains.map((domain, index) => <div key={domain.id}><div><span><i style={{ background: domain.color }} />{domain.name}</span><em>{index === 3 ? '等待节点' : '已完成'}</em></div><Progress percent={index === 3 ? 68 : 100} strokeColor={domain.color} /><small>{index === 3 ? '4 / 5 个节点已上传' : '域级摘要已发送至中央服务器'}</small></div>)}</div>
      </Panel>
    </div>
    <Panel title="防御决策明细" subtitle="模拟真值不参与风险判断" extra={<Segmented size="small" options={['全部阶段', '特征统计阶段', '正常训练阶段']} />}>
      <Table rowKey="id" size="small" pagination={false} dataSource={nodeRows.filter((node) => node.assessment !== '通过')} columns={[
        { title: '阶段', render: (_: unknown, __: unknown, i: number) => i % 2 ? '正常训练阶段' : '特征统计阶段' },
        { title: '轮次', render: () => round }, { title: '域', dataIndex: 'domainId', render: (id: string) => domains.find((d) => d.id === id)?.name },
        { title: '节点', dataIndex: 'id' }, { title: '风险分数', dataIndex: 'risk', render: (value: number) => <b className="text-danger">{value.toFixed(2)}</b> },
        { title: '处理结果', dataIndex: 'assessment', render: (value: string) => <Tag color={value === '过滤' ? 'error' : 'warning'}>{value}</Tag> },
        { title: '判定依据', render: () => '更新方向偏移 · 跨指标一致性异常' },
      ]} />
    </Panel>
  </div>;
}
