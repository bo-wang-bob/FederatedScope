import { EyeOutlined, ReloadOutlined } from '@ant-design/icons';
import { Alert, Button, Descriptions, Drawer, Empty, Space, Table, Tag } from 'antd';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { experimentApi } from '../api/experimentApi';
import { Panel } from '../components/ChartPanel';
import { MetricCard } from '../components/MetricCard';
import { PageHeader } from '../components/PageHeader';
import { useAppStore } from '../store/useAppStore';
import type { ExperimentRecord } from '../types';

const typeLabels = { heterogeneity: '异构协同', privacy: '隐私保护', backdoor: '后门攻防' };
const methodLabels = { fedavg: 'FedAvg', fedprox: 'FedProx', heterogeneous_solution: '异构解决方案' };
const statusLabels: Record<string, string> = { queued: '排队中', running: '运行中', stopping: '停止中', stopped: '已停止', completed: '已完成', failed: '失败' };
const statusColors: Record<string, string> = { queued: 'default', running: 'processing', stopping: 'warning', stopped: 'default', completed: 'success', failed: 'error' };

function formatTime(value?: string) {
  return value ? new Date(value).toLocaleString() : '--';
}

function finalMetric(record: ExperimentRecord) {
  const entries = Object.entries(record.finalMetrics || {});
  if (!entries.length) return '暂无';
  return entries.slice(0, 2).map(([key, value]) => `${key}: ${Number(value).toFixed(3)}`).join(' · ');
}

export function ReportsPage() {
  const navigate = useNavigate();
  const setActiveExperiment = useAppStore((state) => state.setActiveExperiment);
  const [records, setRecords] = useState<ExperimentRecord[]>([]);
  const [selected, setSelected] = useState<ExperimentRecord>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = () => {
    setLoading(true);
    experimentApi.list()
      .then((result) => { setRecords(result); setError(''); })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);
  const completed = records.filter((record) => record.status === 'completed').length;
  const running = records.filter((record) => ['queued', 'running', 'stopping'].includes(record.status)).length;
  const security = records.filter((record) => record.type !== 'heterogeneity').length;

  const openMonitor = (record: ExperimentRecord) => {
    setActiveExperiment(record.experimentId);
    navigate(`/experiments/${encodeURIComponent(record.experimentId)}/live`);
  };

  return <div className="page reports-page">
    <PageHeader eyebrow="EXPERIMENT RECORDS" title="实验记录" description="记录每次实验的配置、运行状态和最终指标。" actions={<Button icon={<ReloadOutlined />} loading={loading} onClick={load}>刷新</Button>} />
    {error && <Alert type="error" showIcon message="实验记录加载失败" description={error} action={<Button size="small" onClick={load}>重试</Button>} />}
    <div className="metrics-grid four"><MetricCard label="实验总数" value={records.length} delta="后端持久化记录" /><MetricCard label="已完成" value={completed} delta={`完成率 ${records.length ? ((completed / records.length) * 100).toFixed(1) : '0.0'}%`} tone="green" /><MetricCard label="正在运行" value={running} delta="含排队与停止中" tone="cyan" /><MetricCard label="安全实验" value={security} delta="隐私与后门实验" tone="violet" /></div>
    <Panel title="全部实验" subtitle="按创建时间倒序排列">
      {records.length || loading ? <Table rowKey="experimentId" loading={loading} dataSource={records} pagination={{ pageSize: 12 }} columns={[
        { title: '实验编号', dataIndex: 'experimentId', width: 210, render: (value: string, row: ExperimentRecord) => <button className="link-button" onClick={() => setSelected(row)}>{value}</button> },
        { title: '实验名称', dataIndex: 'name' },
        { title: '类型', dataIndex: 'type', render: (value: ExperimentRecord['type']) => <Tag color={value === 'backdoor' ? 'orange' : value === 'privacy' ? 'purple' : 'cyan'}>{typeLabels[value]}</Tag> },
        { title: '方案', dataIndex: 'method', render: (value: ExperimentRecord['method']) => methodLabels[value] },
        { title: '状态', dataIndex: 'status', render: (value: string) => <Tag color={statusColors[value]}>{statusLabels[value] || value}</Tag> },
        { title: '创建时间', dataIndex: 'createdAt', render: formatTime },
        { title: '轮次', render: (_: unknown, row: ExperimentRecord) => `${row.round} / ${row.totalRounds}` },
        { title: '核心结果', render: (_: unknown, row: ExperimentRecord) => finalMetric(row) },
        { title: '操作', fixed: 'right' as const, width: 110, render: (_: unknown, row: ExperimentRecord) => <Button size="small" icon={<EyeOutlined />} onClick={() => openMonitor(row)}>运行记录</Button> },
      ]} /> : <Empty description="尚无实验记录" />}
    </Panel>

    <Drawer title={`实验配置 · ${selected?.experimentId ?? ''}`} width={660} open={Boolean(selected)} onClose={() => setSelected(undefined)} extra={selected && <Button type="primary" onClick={() => openMonitor(selected)}>查看运行记录</Button>}>
      {selected && <>
        <Descriptions column={2} bordered size="small" items={[
          { key: 'name', label: '实验名称', children: selected.name },
          { key: 'status', label: '状态', children: <Tag color={statusColors[selected.status]}>{statusLabels[selected.status]}</Tag> },
          { key: 'type', label: '类型', children: typeLabels[selected.type] },
          { key: 'method', label: '运行方案', children: methodLabels[selected.method] },
          { key: 'scenario', label: '场景快照', children: selected.scenarioId },
          { key: 'alpha', label: '狄利克雷参数', children: selected.scenarioSummary.alpha },
          { key: 'created', label: '创建时间', children: formatTime(selected.createdAt) },
          { key: 'ended', label: '结束时间', children: formatTime(selected.endedAt) },
        ]} />
        {selected.error && <Alert type="error" showIcon message={selected.error.code} description={selected.error.message} style={{ marginTop: 16 }} />}
        <Panel title="规范化配置" subtitle="任务实际接收的公开配置" className="record-config-panel"><pre>{JSON.stringify(selected.config, null, 2)}</pre></Panel>
        <Panel title="最终指标" subtitle="后端最后一次有效指标"><pre>{JSON.stringify(selected.finalMetrics, null, 2)}</pre></Panel>
      </>}
    </Drawer>
  </div>;
}
