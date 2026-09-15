import { useEffect, useState } from 'react';
import { Alert, Button, Collapse, Descriptions, Empty, Popconfirm, Progress, Space, Table, Tabs } from 'antd';
import { DownloadOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons';
import { api, bytes, methodLabel, percent, terminal, type Client, type Job, type Library, type Resource } from './api';
import { Curves, Distribution, LossChart, Topology } from './charts';
import { EvaluationResults } from './evaluation';
import { PredictionPanel } from './inference';
import { Panel, State, Stat } from './ui';

export function ResourceCards({ resource }: { resource: Resource }) {
  return <div className="platform-resource-cards">{resource.gpus.map(g => <div key={g.index}><div><strong>GPU {g.index}</strong><span>{g.utilization}% · {g.temperature}°C</span></div><Progress percent={Math.round(g.memoryUsedMiB / g.memoryTotalMiB * 100)} showInfo={false} size="small" /><small>{(g.memoryUsedMiB / 1024).toFixed(1)} / {(g.memoryTotalMiB / 1024).toFixed(1)} GB 显存</small></div>)}
    {resource.gpuError && <Alert type="warning" title="GPU 指标不可用" description={resource.gpuError} />}
    <p>内存 {bytes(resource.memoryUsed)} / {bytes(resource.memoryTotal)} · 磁盘剩余 {bytes(resource.diskFree)}</p></div>;
}

export function JobTable({ jobs, open }: { jobs: Job[]; open: (id: string) => void }) {
  return <Table size="middle" rowKey="id" dataSource={jobs} scroll={{ x: 650 }} pagination={{ pageSize: 8, hideOnSinglePage: true }} columns={[
    { title: '任务', key: 'name', render: (_, j) => <Button className="platform-job-link" type="link" onClick={() => open(j.id)}>{j.request.name || j.id.slice(0, 8)}<small>{j.request.group} · {methodLabel(j.request.method)}</small></Button> },
    { title: '类型', dataIndex: 'action', render: a => ({ train: '联邦训练', inspect: '缓存预检', evaluate: '独立评测', predict: '单图预测' })[a as string] },
    { title: '状态', dataIndex: 'status', render: value => <State value={value} /> },
    { title: '创建时间', dataIndex: 'createdAt', render: value => new Date(value).toLocaleString() },
  ]} />;
}

export function JobDetail({ job, resources, library, stop, rerun }: { job: Job; resources?: Resource; library: Library; stop: (id: string) => void; rerun: () => void }) {
  const [tab, setTab] = useState(['evaluate', 'predict'].includes(job.action) ? 'advanced' : 'monitor');
  const [logs, setLogs] = useState('');
  useEffect(() => { if (tab !== 'advanced') return; let active = true; const update = async () => { try { const value = await api<string>(`jobs/${job.id}/logs`); if (active) setLogs(value); }
    catch (e) { if (active) setLogs(`日志读取失败：${(e as Error).message}`); } }; void update(); const timer = setInterval(update, 2500);
    return () => { active = false; clearInterval(timer); }; }, [job.id, tab]);
  const clients = Object.values(job.clients || {}), points = job.metrics || [];
  const final = points.at(-1);
  return <><div className="platform-job-heading"><div><h2>{job.request.name || job.id.slice(0, 8)} <State value={job.status} /></h2><p>{job.stage} · {job.request.group} / {methodLabel(job.request.method)}</p></div><Space wrap>
    {!terminal(job.status) && <Popconfirm title="停止当前任务？" description="仅回收此任务进程，保留配置、日志及已生成文件。" onConfirm={() => stop(job.id)}><Button danger icon={<StopOutlined />}>停止任务</Button></Popconfirm>}
    {terminal(job.status) && ['train', 'inspect'].includes(job.action) && <Button onClick={rerun} icon={<ReloadOutlined />}>复用配置</Button>}
    <Button href={`/api/platform/jobs/${job.id}/export`} icon={<DownloadOutlined />}>结果 JSON</Button>
    {terminal(job.status) && <Button href={`/api/platform/jobs/${job.id}/bundle`}>完整复现包</Button>}</Space></div>
    {job.error && <Alert type="error" title={job.error} showIcon />}
    {job.data?.augmentation?.warning && <Alert type="warning" title={job.data.augmentation.warning} showIcon />}
    {job.request.method === 'heterogeneous_solution' && <Alert type="info" title={`${job.request.augmentationMode === 'generate' ? '按配置重新生成' : '复用已有增强缓存'} · 已保存增强样本 ${job.data?.augmentation?.cachedSamples ?? '—'}；同轮数不等于同计算量。`} />}
    {terminal(job.status) && !job.cleanup.ok && <Alert type="error" title={job.cleanup.message} action={!job.cleanup.ok && <Button onClick={() => stop(job.id)}>重试本任务清理</Button>} />}
    {job.action === 'predict' ? <PredictionPanel job={job} /> : job.action === 'evaluate' ? <EvaluationResults job={job} library={library} /> : <div className="platform-stats"><Stat label="已评测轮次" value={final ? `${final.round} / ${job.request.rounds}` : '—'} /><Stat label="总体准确率" value={percent(final?.accuracy)} sub="测试样本加权" /><Stat label="分域平均" value={percent(final?.domainMean)} sub="各域等权" /><Stat label="最差域准确率" value={percent(final?.worstDomain)} /></div>}
    <Tabs activeKey={tab} onChange={setTab} items={[{ key: 'monitor', label: '训练监控', children: <><div className="platform-grid"><Panel title="准确率曲线"><Curves points={points} /></Panel><Panel title="客户端拓扑"><Topology clients={clients} /></Panel></div><div className="platform-grid"><Panel title="本地训练损失"><LossChart points={points} /></Panel><Panel title="资源占用">{resources ? <ResourceCards resource={resources} /> : <Empty />}</Panel></div>
      <Panel title="客户端状态"><Table<Client> size="small" rowKey="id" dataSource={clients} pagination={{ pageSize: 12 }} columns={[{ title: '客户端', dataIndex: 'id' }, { title: '域', dataIndex: 'domain' }, { title: '原始样本', dataIndex: 'samples' }, { title: '状态', dataIndex: 'stage' }, { title: '轮次', dataIndex: 'round', render: n => n == null ? '—' : n + 1 }, { title: '训练损失', dataIndex: 'loss', render: n => n?.toFixed(4) ?? '—' }, { title: '本地训练准确率', dataIndex: 'accuracy', render: percent }]} /></Panel></> },
      { key: 'data', label: '数据异构', children: <Panel title="客户端与类别分布"><Distribution clients={clients} classes={job.data?.classes || []} /><p className="platform-muted">此图为原始划分；若配置每客户端抽样，实际训练样本数另保存在客户端日志。</p></Panel> },
      { key: 'advanced', label: '高级详情', children: <><Collapse items={[{ key: 'config', label: '参数与版本', children: <><Descriptions bordered column={2} items={Object.entries(job.request).map(([k, v]) => ({ key: k, label: k, children: JSON.stringify(v) }))} /><Panel title="实际配置与版本"><pre className="platform-code">{JSON.stringify({ config: job.config, provenance: job.provenance, data: job.data }, null, 2)}</pre></Panel></> }, { key: 'logs', label: '运行日志', children: <pre className="platform-log">{logs || '尚无日志'}</pre> }]} /></> }].filter(item => !['evaluate', 'predict'].includes(job.action) || item.key === 'advanced')} />
  </>;
}
