import { DownloadOutlined, EyeOutlined, RedoOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import { Alert, Button, Descriptions, Drawer, Empty, Input, Select, Space, Table, Tag, message } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { experimentApi } from '../api/experimentApi';
import { Panel } from '../components/ChartPanel';
import { MetricCard } from '../components/MetricCard';
import { PageHeader } from '../components/PageHeader';
import { useAppStore } from '../store/useAppStore';
import type { ExperimentConfig, ExperimentRecord, ScenarioRecord } from '../types';

const typeLabels = { heterogeneity: '异构协同', privacy: '隐私保护', backdoor: '后门攻防' };
const methodLabels = { fedavg: 'FedAvg', fedprox: 'FedProx', heterogeneous_solution: '异构解决方案' };
const statusLabels: Record<string, string> = { queued: '排队中', running: '运行中', stopping: '停止中', stopped: '已停止', completed: '已完成', failed: '失败' };
const statusColors: Record<string, string> = { queued: 'default', running: 'processing', stopping: 'warning', stopped: 'default', completed: 'success', failed: 'error' };

function formatTime(value?: string) {
  return value ? new Date(value).toLocaleString() : '--';
}

function formatDuration(record: ExperimentRecord) {
  if (!record.startedAt) return '--';
  const end = record.endedAt ? new Date(record.endedAt).getTime() : Date.now();
  const seconds = Math.max(0, Math.floor((end - new Date(record.startedAt).getTime()) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours ? `${hours}时` : ''}${minutes}分${seconds % 60}秒`;
}

function finalMetric(record: ExperimentRecord) {
  const metrics = record.finalMetrics || {};
  const candidates = record.type === 'backdoor'
    ? [['accuracy', '准确率', '%'], ['attackSuccess', '攻击成功率', '%'], ['truePositiveRate', '检出率', '%']]
    : record.type === 'privacy' && record.config.privacy?.attack === 'reconstruction'
      ? [['reconstructionPsnr', '重建质量', 'dB'], ['reconstructionLoss', '重建损失', '']]
      : record.type === 'privacy'
        ? [['accuracy', '准确率', '%'], ['privacyRisk', '隐私攻击指标', '%']]
        : [['accuracy', '准确率', '%'], ['worstDomain', '最弱域准确率', '%'], ['domainGap', '域间差距', '%']];
  const values = candidates.flatMap(([key, label, unit]) => {
    const value = metrics[key];
    if (value === undefined) return [];
    const displayed = unit === '%' ? (Number(value) * 100).toFixed(2) : Number(value).toFixed(3);
    return [`${label}: ${displayed}${unit}`];
  });
  return values.slice(0, 2).join(' · ') || '暂无';
}

export function ReportsPage() {
  const navigate = useNavigate();
  const setActiveExperiment = useAppStore((state) => state.setActiveExperiment);
  const [records, setRecords] = useState<ExperimentRecord[]>([]);
  const [selected, setSelected] = useState<ExperimentRecord>();
  const [selectedScenario, setSelectedScenario] = useState<ScenarioRecord>();
  const [selectedLogs, setSelectedLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [restarting, setRestarting] = useState('');

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
  const filteredRecords = useMemo(() => records.filter((record) => {
    const search = query.trim().toLowerCase();
    return (typeFilter === 'all' || record.type === typeFilter)
      && (statusFilter === 'all' || record.status === statusFilter)
      && (!search || `${record.experimentId} ${record.name}`.toLowerCase().includes(search));
  }), [query, records, statusFilter, typeFilter]);

  const openMonitor = (record: ExperimentRecord) => {
    setActiveExperiment(record.experimentId);
    navigate(`/experiments/${encodeURIComponent(record.experimentId)}/live`);
  };

  const selectRecord = (record: ExperimentRecord) => {
    setSelected(record);
    setSelectedScenario(undefined);
    setSelectedLogs([]);
    Promise.allSettled([
      experimentApi.getScenario(record.scenarioId),
      experimentApi.logs(record.experimentId, 120),
    ]).then(([scenarioResult, logResult]) => {
      if (scenarioResult.status === 'fulfilled') setSelectedScenario(scenarioResult.value);
      if (logResult.status === 'fulfilled') setSelectedLogs(logResult.value);
    });
  };

  const rerun = async (record: ExperimentRecord) => {
    const config = {
      ...record.config,
      idempotencyKey: typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
      name: `${record.name}-复现`,
    } as ExperimentConfig;
    setRestarting(record.experimentId);
    try {
      await experimentApi.preflight(config);
      const created = await experimentApi.create(config);
      setActiveExperiment(created.experimentId);
      navigate(`/experiments/${encodeURIComponent(created.experimentId)}/live`);
    } catch (reason) {
      message.error(reason instanceof Error ? reason.message : '重新运行失败');
    } finally {
      setRestarting('');
    }
  };

  const exportRecord = (record: ExperimentRecord) => {
    const blob = new Blob([JSON.stringify({
      schemaVersion: '1.0', exportedAt: new Date().toISOString(),
      experiment: record,
    }, null, 2)], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${record.experimentId}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return <div className="page reports-page">
    <PageHeader eyebrow="EXPERIMENT RECORDS" title="实验记录" description="记录每次实验的配置、运行状态和最终指标。" actions={<Button icon={<ReloadOutlined />} loading={loading} onClick={load}>刷新</Button>} />
    {error && <Alert type="error" showIcon message="实验记录加载失败" description={error} action={<Button size="small" onClick={load}>重试</Button>} />}
    <div className="metrics-grid four"><MetricCard label="实验总数" value={records.length} delta="后端持久化记录" /><MetricCard label="已完成" value={completed} delta={`完成率 ${records.length ? ((completed / records.length) * 100).toFixed(1) : '0.0'}%`} tone="green" /><MetricCard label="正在运行" value={running} delta="含排队与停止中" tone="cyan" /><MetricCard label="安全实验" value={security} delta="隐私与后门实验" tone="violet" /></div>
    <Panel title="全部实验" subtitle="按创建时间倒序排列" extra={<Space wrap><Input allowClear prefix={<SearchOutlined />} placeholder="实验编号或名称" value={query} onChange={(event) => setQuery(event.target.value)} /><Select value={typeFilter} onChange={setTypeFilter} options={[{ value: 'all', label: '全部类型' }, ...Object.entries(typeLabels).map(([value, label]) => ({ value, label }))]} /><Select value={statusFilter} onChange={setStatusFilter} options={[{ value: 'all', label: '全部状态' }, ...Object.entries(statusLabels).map(([value, label]) => ({ value, label }))]} /></Space>}>
      {records.length || loading ? <Table rowKey="experimentId" loading={loading} dataSource={filteredRecords} pagination={{ pageSize: 12 }} columns={[
        { title: '实验编号', dataIndex: 'experimentId', width: 210, render: (value: string, row: ExperimentRecord) => <button className="link-button" onClick={() => selectRecord(row)}>{value}</button> },
        { title: '实验名称', dataIndex: 'name' },
        { title: '类型', dataIndex: 'type', render: (value: ExperimentRecord['type']) => <Tag color={value === 'backdoor' ? 'orange' : value === 'privacy' ? 'purple' : 'cyan'}>{typeLabels[value]}</Tag> },
        { title: '方案', dataIndex: 'method', render: (value: ExperimentRecord['method']) => methodLabels[value] },
        { title: '状态', dataIndex: 'status', render: (value: string) => <Tag color={statusColors[value]}>{statusLabels[value] || value}</Tag> },
        { title: '创建时间', dataIndex: 'createdAt', render: formatTime },
        { title: '耗时', render: (_: unknown, row: ExperimentRecord) => formatDuration(row) },
        { title: '轮次', render: (_: unknown, row: ExperimentRecord) => `${row.round} / ${row.totalRounds}` },
        { title: '核心结果', render: (_: unknown, row: ExperimentRecord) => finalMetric(row) },
        { title: '操作', fixed: 'right' as const, width: 210, render: (_: unknown, row: ExperimentRecord) => <Space><Button size="small" icon={<EyeOutlined />} onClick={() => openMonitor(row)}>记录</Button><Button size="small" icon={<RedoOutlined />} loading={restarting === row.experimentId} onClick={() => rerun(row)}>复现</Button></Space> },
      ]} /> : <Empty description="尚无实验记录" />}
    </Panel>

    <Drawer title={`实验配置 · ${selected?.experimentId ?? ''}`} width={660} open={Boolean(selected)} onClose={() => setSelected(undefined)} extra={selected && <Space><Button icon={<DownloadOutlined />} onClick={() => exportRecord(selected)}>导出原始结果</Button><Button type="primary" onClick={() => openMonitor(selected)}>查看运行记录</Button></Space>}>
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
        {selectedScenario && <Panel title="场景快照" subtitle={`数据指纹 ${(selectedScenario.preview.datasetFingerprint || 'simulation').slice(0, 12)}`} className="record-config-panel"><Descriptions size="small" column={2} items={selectedScenario.preview.domains.map((domain) => ({ key: domain.domainKey, label: domain.domainKey, children: `${domain.totalSamples.toLocaleString()} 样本 · ${domain.clients.length} 客户端` }))} /></Panel>}
        <Panel title="规范化配置" subtitle="任务实际接收的公开配置" className="record-config-panel"><pre>{JSON.stringify(selected.config, null, 2)}</pre></Panel>
        <Panel title="最终指标" subtitle="后端最后一次有效指标"><pre>{JSON.stringify(selected.finalMetrics, null, 2)}</pre></Panel>
        <Panel title="运行日志尾部" subtitle="最多显示最近 120 行"><pre>{selectedLogs.length ? selectedLogs.join('\n') : '暂无运行日志'}</pre></Panel>
      </>}
    </Drawer>
  </div>;
}
