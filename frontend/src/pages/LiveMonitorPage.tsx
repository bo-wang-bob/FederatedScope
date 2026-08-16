import { ClockCircleOutlined, StopOutlined } from '@ant-design/icons';
import { Alert, Button, Empty, Popconfirm, Progress, Space, Spin, Table, Tag, Timeline, message } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { apiBaseUrl, experimentApi } from '../api/experimentApi';
import { createBackendTrainingAdapter } from '../api/trainingAdapter';
import { Chart, chartGrid, chartText, Panel } from '../components/ChartPanel';
import { MetricCard } from '../components/MetricCard';
import { PageHeader } from '../components/PageHeader';
import { FederationTopology } from '../components/topology/FederationTopology';
import { domains, nodeRows } from '../mock/data';
import { phases, useAppStore } from '../store/useAppStore';
import type { ExperimentMetricPoint, ExperimentRecord, TrainingConnectionState, TrainingEvent, TrainingSnapshot } from '../types';
import { mergeTrainingEvent, recoverTrainingSnapshot } from '../utils/eventReducer';

const statusLabels: Record<string, string> = {
  queued: '排队中', running: '运行中', stopping: '停止中', stopped: '已停止',
  completed: '已完成', failed: '失败', disconnected: '连接断开',
};

const typeLabels = { heterogeneity: '异构协同', privacy: '隐私保护', backdoor: '后门攻防' };
const methodLabels = { fedavg: 'FedAvg', fedprox: 'FedProx', heterogeneous_solution: '异构解决方案' };
const terminal = new Set(['completed', 'failed', 'stopped']);
const eventTypeLabels: Record<string, string> = {
  'experiment.started': '实验启动',
  'experiment.stopping': '正在停止',
  'experiment.stopped': '实验已停止',
  'experiment.completed': '实验完成',
  'experiment.failed': '实验失败',
  'stage.changed': '阶段切换',
  'round.started': '轮次开始',
  'round.completed': '轮次完成',
  'client.status.changed': '节点状态',
  'client.metric.updated': '节点保护统计',
  'metric.updated': '指标更新',
  'defense.decision': '防御判定',
  'warning.raised': '运行告警',
};

function eventDescription(event: TrainingEvent) {
  const payload = event.payload as Record<string, unknown>;
  if (payload.message) return String(payload.message);
  if (event.type === 'stage.changed') return `进入 ${String(payload.stage ?? '新阶段')}`;
  if (event.type === 'round.started') return `第 ${String(payload.round ?? '--')} 轮开始`;
  if (event.type === 'round.completed') return `第 ${String(payload.round ?? '--')} 轮完成，收到 ${String(payload.receivedClientUpdates ?? '--')} 个节点更新`;
  if (event.type === 'client.status.changed') return `${String(payload.clientId ?? '节点')}：${String(payload.status ?? '状态更新')}`;
  if (event.type === 'client.metric.updated') return `${String(payload.clientId ?? '节点')} 已完成上传保护处理`;
  if (event.type === 'defense.decision') return `${payload.stage === 'feature_statistics' ? '特征统计阶段' : '正常训练阶段'}过滤 ${Array.isArray(payload.droppedClientIds) ? payload.droppedClientIds.length : 0} 个节点`;
  if (event.type === 'metric.updated') return `第 ${String(payload.round ?? '--')} 轮指标已更新`;
  return `事件序号 ${event.sequence}`;
}

function percentMetric(value: number | undefined) {
  if (value === undefined) return '--';
  return value <= 1 ? (value * 100).toFixed(2) : value.toFixed(2);
}

export function LiveMonitorPage() {
  const { id = '' } = useParams();
  const navigate = useNavigate();
  const [record, setRecord] = useState<ExperimentRecord>();
  const [snapshot, setSnapshot] = useState<TrainingSnapshot>();
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [connectionState, setConnectionState] = useState<TrainingConnectionState>('connecting');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [stopping, setStopping] = useState(false);
  const setPhaseIndex = useAppStore((state) => state.setPhaseIndex);
  const setActiveExperiment = useAppStore((state) => state.setActiveExperiment);

  useEffect(() => {
    if (!id || id === 'current') {
      setError('当前没有可打开的实验，请先创建实验或从实验记录中选择。');
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    const adapter = createBackendTrainingAdapter(apiBaseUrl());
    let subscription: ReturnType<typeof adapter.subscribe> | undefined;
    Promise.all([
      experimentApi.get(id, controller.signal),
      adapter.getSnapshot(id, controller.signal),
    ]).then(([experiment, initial]) => {
      setRecord(experiment);
      setSnapshot(initial);
      setEvents([...(initial.recentEvents ?? [])].reverse());
      setPhaseIndex(initial.phaseIndex);
      setActiveExperiment(id);
      setError('');
      if (!terminal.has(initial.status)) {
        subscription = adapter.subscribe(
          id,
          initial.sequence,
          (event) => {
            setEvents((current) => [event, ...current].slice(0, 100));
            setSnapshot((current) => {
              const next = current ? mergeTrainingEvent(current, event) : current;
              if (next) setPhaseIndex(next.phaseIndex);
              return next;
            });
          },
          setConnectionState,
        );
      } else {
        setConnectionState('connected');
      }
    }).catch((reason: Error) => setError(reason.message)).finally(() => setLoading(false));
    return () => {
      controller.abort();
      subscription?.close();
    };
  }, [id, setActiveExperiment, setPhaseIndex]);

  useEffect(() => {
    if (!snapshot || terminal.has(snapshot.status) || connectionState !== 'recovering') return;
    const controller = new AbortController();
    createBackendTrainingAdapter(apiBaseUrl()).getSnapshot(id, controller.signal)
      .then((next) => setSnapshot((current) => current ? recoverTrainingSnapshot(current, next) : next))
      .catch(() => undefined);
    return () => controller.abort();
  }, [connectionState, id, snapshot?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const stopExperiment = async () => {
    if (!id) return;
    setStopping(true);
    try {
      const updated = await experimentApi.stop(id);
      setRecord(updated);
      setSnapshot((current) => current ? { ...current, status: updated.status } : current);
      message.success('停止请求已提交');
    } catch (reason) {
      message.error(reason instanceof Error ? reason.message : '停止实验失败');
    } finally {
      setStopping(false);
    }
  };

  const metrics = snapshot?.metrics ?? record?.metrics ?? [];
  const latest: ExperimentMetricPoint = metrics.at(-1) ?? { round: 0 };
  const trendOption = useMemo(() => {
    const series = [
      { key: 'accuracy', name: '全局准确率', color: '#44d8ff' },
      { key: 'loss', name: '损失', color: '#8b7cff' },
      { key: 'domainArt', name: '数字孪生域', color: '#18d7c5' },
      { key: 'domainClipart', name: '战术符号域', color: '#ffbd52' },
      { key: 'domainProduct', name: '装备数据库域', color: '#a68cff' },
      { key: 'domainReal_World', name: '实景侦察域', color: '#5da9ff' },
      ...(record?.type === 'privacy' && record.config.privacy?.attack !== 'reconstruction' ? [{ key: 'privacyRisk', name: '隐私攻击指标', color: '#ffbd52' }] : []),
      ...(record?.type === 'privacy' && record.config.privacy?.attack === 'reconstruction' ? [{ key: 'reconstructionPsnr', name: '重建质量（PSNR）', color: '#f0529d' }, { key: 'reconstructionLoss', name: '重建优化损失', color: '#ff8a52' }] : []),
      ...(record?.type === 'backdoor' ? [{ key: 'attackSuccess', name: '攻击成功率', color: '#f0526d' }] : []),
      ...(record?.type === 'heterogeneity' ? [{ key: 'worstDomain', name: '最弱域准确率', color: '#29e3ae' }] : []),
    ];
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: series.map((item) => item.name), textStyle: { color: chartText } },
      grid: { left: 48, right: 28, top: 42, bottom: 30 },
      xAxis: { type: 'category', data: metrics.map((item) => item.round), axisLabel: { color: chartText }, axisLine: { lineStyle: { color: chartGrid } } },
      yAxis: { type: 'value', axisLabel: { color: chartText }, splitLine: { lineStyle: { color: chartGrid } } },
      series: series.map((item) => ({ name: item.name, type: 'line', smooth: true, connectNulls: true, symbol: 'none', data: metrics.map((metric) => metric[item.key]), lineStyle: { color: item.color, width: 2 } })),
    };
  }, [metrics, record?.type]);

  const privacyConfig = record?.config.type === 'privacy' ? record.config.privacy : null;
  const backdoorConfig = record?.config.type === 'backdoor' ? record.config.backdoor : null;
  const reconstructionMetric = latest.reconstructionPsnr !== undefined
    ? { label: '重建质量（PSNR）', value: latest.reconstructionPsnr.toFixed(2), suffix: 'dB' }
    : { label: '重建优化损失', value: latest.reconstructionLoss?.toFixed(4) ?? '--', suffix: '' };
  const maliciousSet = useMemo(() => new Set(backdoorConfig?.maliciousClients ?? []), [backdoorConfig]);
  const clientRows = nodeRows.map((node) => {
    const runtime = snapshot?.clients[node.id];
    return {
      ...node,
      status: runtime?.status ?? (snapshot?.status === 'running' ? '等待聚合' : '待机'),
      progress: runtime?.progress ?? (snapshot?.status === 'completed' ? 100 : 0),
      sampleCount: runtime?.sampleCount ?? node.sampleCount,
      assessment: runtime?.assessment ?? '通过',
      malicious: maliciousSet.has(node.id),
    };
  });
  const protectedClients = Object.values(snapshot?.clients ?? {}).filter((client) => client.noiseStd !== undefined);
  const averageNoiseStd = protectedClients.length
    ? protectedClients.reduce((sum, client) => sum + (client.noiseStd ?? 0), 0) / protectedClients.length
    : undefined;
  const defenseEvents = events.filter((event) => event.type === 'defense.decision');

  if (loading) return <div className="page route-loading"><Spin size="large" /><span>正在连接实验任务…</span></div>;
  if (error || !record || !snapshot) return <div className="page"><PageHeader eyebrow="LIVE EXPERIMENT MONITOR" title="实验运行监控" description="查看后端单机任务的实时状态。" /><Alert type="error" showIcon message="无法打开实验" description={error || '实验数据不完整'} action={<Space><Button onClick={() => navigate('/experiments/new')}>创建实验</Button><Button onClick={() => navigate('/reports')}>实验记录</Button></Space>} /></div>;

  const phaseIndex = Math.min(phases.length - 1, Math.max(0, snapshot.phaseIndex));
  const status = snapshot.status;
  return <div className="page live-monitor-page">
    <PageHeader eyebrow="LIVE EXPERIMENT MONITOR" title="实验运行监控" description={`${record.name} · ${record.experimentId}`} actions={<Space><Tag color={connectionState === 'connected' ? 'processing' : 'warning'}>{connectionState === 'connected' ? '实时连接' : connectionState === 'recovering' ? '正在恢复' : connectionState}</Tag><Tag color={status === 'failed' ? 'error' : terminal.has(status) ? 'default' : 'success'}>{statusLabels[status] || status}</Tag>{!terminal.has(status) && <Popconfirm title="停止当前实验？" description="停止后不能从当前轮次恢复。" onConfirm={stopExperiment}><Button danger icon={<StopOutlined />} loading={stopping}>停止实验</Button></Popconfirm>}</Space>} />

    {snapshot.error && <Alert type="error" showIcon message="实验运行失败" description={snapshot.error.message} />}
    <div className="stage-strip">{phases.map((phase, index) => <div key={phase} className={index === phaseIndex ? 'active' : index < phaseIndex ? 'done' : ''}><i>{index < phaseIndex ? '✓' : index + 1}</i><span>{phase}</span></div>)}</div>
    <div className="metrics-grid five">
      <MetricCard label="当前轮次" value={snapshot.round} suffix={` / ${snapshot.totalRounds ?? record.totalRounds}`} delta={`${typeLabels[record.type]} · ${methodLabels[record.method]}`} />
      <MetricCard label="任务进度" value={Math.round((snapshot.round / Math.max(1, snapshot.totalRounds ?? record.totalRounds)) * 100)} suffix="%" delta={statusLabels[status] || status} tone="green" />
      <MetricCard label="全局准确率" value={percentMetric(latest.accuracy)} suffix="%" delta={latest.accuracy === undefined ? '等待后端指标' : `第 ${latest.round} 轮`} tone="cyan" />
      {record.type === 'heterogeneity' && <><MetricCard label="最弱域准确率" value={percentMetric(latest.worstDomain)} suffix="%" delta="跨域性能下界" tone="violet" /><MetricCard label="域间性能差距" value={percentMetric(latest.domainGap)} suffix="%" delta="越低越均衡" tone="amber" /></>}
      {record.type === 'privacy' && <><MetricCard label={privacyConfig?.attack === 'reconstruction' ? reconstructionMetric.label : '隐私攻击指标'} value={privacyConfig?.attack === 'reconstruction' ? reconstructionMetric.value : percentMetric(latest.privacyRisk)} suffix={privacyConfig?.attack === 'reconstruction' ? reconstructionMetric.suffix : '%'} delta={privacyConfig?.defenseEnabled ? '本地保护已启用' : '无保护实验'} tone="violet" /><MetricCard label="实际噪声标准差" value={(averageNoiseStd ?? latest.noiseStd ?? latest.noiseMultiplier)?.toFixed(3) ?? '--'} delta={protectedClients.length ? `${protectedClients.length} 个客户端最新均值` : '等待客户端上传统计'} tone="amber" /></>}
      {record.type === 'backdoor' && <><MetricCard label="攻击成功率" value={percentMetric(latest.attackSuccess)} suffix="%" delta={backdoorConfig?.defenseEnabled ? '攻击防御已启用' : '无防御实验'} tone="red" /><MetricCard label="检测 TPR / FPR" value={`${percentMetric(latest.truePositiveRate)} / ${percentMetric(latest.falsePositiveRate)}`} suffix="%" delta="真值仅用于复盘" tone="amber" /></>}
    </div>

    <div className="live-grid">
      <Panel title="三级运行态势" subtitle={`当前：${phases[phaseIndex]} · 第 ${snapshot.round} 轮`} className="live-topology"><FederationTopology compact nodes={clientRows} /></Panel>
      <Panel title="实时事件" subtitle="后端任务事件与运行日志" extra={<ClockCircleOutlined />}>
        {events.length ? <Timeline className="event-timeline" items={events.slice(0, 20).map((event) => ({ color: event.type.includes('failed') ? 'red' : event.type.includes('completed') ? 'green' : 'blue', children: <div className="event-item"><span>{new Date(event.timestamp).toLocaleTimeString()} · {eventTypeLabels[event.type] ?? event.type}</span><p>{eventDescription(event)}</p></div> }))} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="等待后端事件" />}
      </Panel>
    </div>

    {record.type === 'backdoor' && <Panel title="双阶段防御判定" subtitle="恶意真值与防御输出独立展示"><div className="defense-event-grid">{['feature_statistics', 'federated_training'].map((stage) => { const event = defenseEvents.find((item) => (item.payload as { stage?: string }).stage === stage); const payload = event?.payload as { droppedClientIds?: Array<string | number>; keptClientIds?: Array<string | number>; threshold?: number } | undefined; return <div key={stage}><span>{stage === 'feature_statistics' ? '特征统计阶段' : '正常训练阶段'}</span><b>{event ? `${payload?.droppedClientIds?.length ?? 0} 个节点被过滤` : '等待阶段结果'}</b><small>{event ? `保留 ${payload?.keptClientIds?.length ?? 0} · 阈值 ${payload?.threshold?.toFixed(3) ?? '--'}` : '结构化防御事件尚未到达'}</small></div>; })}</div></Panel>}

    <div className="live-bottom-grid">
      <Panel title="运行指标趋势" subtitle="按实验类型展示后端真实指标">{metrics.length ? <Chart option={trendOption} height={310} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="训练产生指标后将在此显示" />}</Panel>
      <Panel title="四域客户端进度" subtitle="三级结构为单机客户端状态投影"><div className="upload-list">{domains.map((domain) => { const rows = clientRows.filter((client) => client.domainId === domain.id); const completed = rows.filter((client) => client.progress >= 100).length; const percent = Math.round(rows.reduce((sum, client) => sum + client.progress, 0) / rows.length); return <div key={domain.id}><div><span><i style={{ background: domain.color }} />{domain.name}</span><em>{completed} / 15 完成</em></div><Progress percent={percent} strokeColor={domain.color} /><small>{rows.filter((client) => client.status === '已过滤').length} 个客户端已过滤</small></div>; })}</div></Panel>
    </div>

    <Panel title="客户端运行明细" subtitle="样本信息来自场景，运行状态来自后端快照">
      <Table rowKey="id" size="small" pagination={{ pageSize: 15 }} dataSource={clientRows} columns={[
        { title: '客户端', dataIndex: 'id' },
        { title: '域', dataIndex: 'domainId', render: (value: string) => domains.find((domain) => domain.id === value)?.name },
        { title: '样本数', dataIndex: 'sampleCount' },
        { title: '状态', dataIndex: 'status', render: (value: string) => <Tag>{value}</Tag> },
        { title: '进度', dataIndex: 'progress', render: (value: number) => <Progress percent={value} size="small" /> },
        ...(record.type === 'backdoor' ? [{ title: '模拟角色', dataIndex: 'malicious', render: (value: boolean) => <Tag color={value ? 'error' : 'default'}>{value ? '恶意' : '正常'}</Tag> }] : []),
        ...(record.type === 'backdoor' ? [{ title: '防御判断', dataIndex: 'assessment', render: (value: string) => <Tag color={value === '过滤' ? 'error' : value === '疑似' ? 'warning' : 'success'}>{value}</Tag> }] : []),
      ]} />
    </Panel>
  </div>;
}
