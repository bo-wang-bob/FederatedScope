import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { useLocation, useSearchParams } from 'react-router-dom';
import { App as AntApp, Alert, Badge, Button, Card, Collapse, ConfigProvider, Descriptions, Empty,
  Checkbox, Form, Input, InputNumber, Popconfirm, Progress, Select, Space, Spin, Table, Tabs, Tag, theme } from 'antd';
import { DatabaseOutlined, DownloadOutlined,
  PlayCircleOutlined, ReloadOutlined, StopOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { api, bytes, key, methodLabel, percent, statusText, terminal, type Catalog, type Client, type DataInfo,
  type Job, type Library, type RequestConfig, type Resource } from './api';
import { Curves, Distribution, LossChart, ResourcesChart, Topology } from './charts';
import { ComparisonPanel, EvaluationPanel, EvaluationResults } from './evaluation';
import { ModelExperience, PredictionPanel } from './inference';
import { AppShell } from '../components/AppShell';
import { MetricCard } from '../components/MetricCard';
import { Panel as ExistingPanel } from '../components/ChartPanel';
import { SystemHome } from './home';
import { navigationItems, pages, resolveView } from './navigation';
import './platform.css';
import './refinements.css';
import './workspace.css';

export const State = ({ value }: { value: string }) => <Tag color={value === 'completed' ? 'success' : value === 'failed' ? 'error' : value === 'running' ? 'processing' : 'default'}>{statusText[value] || value}</Tag>;
export const Panel = ({ title, extra, children }: { title: string; extra?: ReactNode; children: ReactNode }) => <ExistingPanel title={title} extra={extra} className="platform-panel">{children}</ExistingPanel>;
export const Stat = ({ label, value, sub }: { label: string; value: string | number; sub?: string }) => <MetricCard label={label} value={value} delta={sub} />;

export default function PlatformApp() {
  return <ConfigProvider theme={{ algorithm: theme.darkAlgorithm, token: { colorPrimary: '#44d8ff', colorSuccess: '#29e3ae', borderRadius: 10,
    fontSize: 14, fontFamily: 'Inter, "Microsoft YaHei", sans-serif', colorBgBase: '#07101f', colorBgContainer: '#101f31', colorText: '#e5f1fb' },
    components: { Card: { headerFontSize: 16 }, Button: { controlHeight: 38 }, Input: { controlHeight: 38 }, Select: { controlHeight: 38 } } }}>
    <AntApp><Workspace /></AntApp>
  </ConfigProvider>;
}

function Workspace() {
  const [query, setQuery] = useSearchParams();
  const location = useLocation();
  const view = resolveView(query.get('view'), location.pathname), selectedId = view === 'jobs' ? query.get('id') : null;
  const page = pages[view];
  useEffect(() => { document.title = `${page.label} · 全域智汇`; }, [page.label]);
  const [catalog, setCatalog] = useState<Catalog>();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [library, setLibrary] = useState<Library>({ models: [], testsets: [] });
  const [resources, setResources] = useState<Resource[]>([]);
  const [selected, setSelected] = useState<Job>();
  const [overview, setOverview] = useState<Job>();
  const [error, setError] = useState('');
  const [detailError, setDetailError] = useState('');
  const { message } = AntApp.useApp();
  const pending = useRef(new Map<string, string>());
  const open = useCallback((id: string) => setQuery({ view: 'jobs', id }), [setQuery]);
  useEffect(() => {
    let active = true, busy = false;
    const refresh = async () => {
      if (busy) return;
      busy = true;
      try {
        const [c, j, l, r] = await Promise.all([api<Catalog>('catalog'), api<Job[]>('jobs'), api<Library>('library'), api<Resource>('resources')]);
        const training = j.find(job => job.action === 'train' && !terminal(job.status))
          || j.find(job => job.action === 'train' && job.status === 'completed')
          || j.find(job => job.action === 'train');
        const detail = training ? await api<Job>(`jobs/${training.id}`) : undefined;
        if (active) { setCatalog(c); setJobs(j); setLibrary(l); setResources(old => [...old, r].slice(-60)); setOverview(detail); setError(''); }
      } catch (e) { if (active) setError((e as Error).message); }
      finally { busy = false; }
    };
    void refresh(); const timer = setInterval(refresh, 4000);
    return () => { active = false; clearInterval(timer); };
  }, []);
  useEffect(() => {
    let active = true;
    setSelected(undefined); setDetailError('');
    if (!selectedId) return;
    const refresh = async () => { try { const job = await api<Job>(`jobs/${selectedId}`); if (active) { setSelected(job); setDetailError(''); } }
      catch (e) { if (active) setDetailError((e as Error).message); } };
    void refresh(); const timer = setInterval(refresh, 1500);
    return () => { active = false; clearInterval(timer); };
  }, [selectedId]);
  const create = async (action: string, payload: object) => {
    const signature = JSON.stringify([action, payload]);
    if (!pending.current.has(signature)) pending.current.set(signature, key());
    const job = await api<Job>(action, { ...payload, idempotencyKey: pending.current.get(signature) });
    pending.current.delete(signature);
    setJobs(old => [job, ...old.filter(j => j.id !== job.id)]);
    return job;
  };
  const stop = async (id: string) => { try { setSelected(await api<Job>(`jobs/${id}/stop`, {})); message.success('本任务停止请求已处理'); }
    catch (e) { message.error((e as Error).message); } };
  const latest = resources.at(-1), running = jobs.find(j => !terminal(j.status));
  const showJob = selected || undefined;
  return <div className="platform-shell"><AppShell platform={{ menuItems: navigationItems, selectedKey: view,
    pageLabel: page.label, sectionLabel: page.section,
    navigate: key => { if (key === 'demo') window.location.assign('/demo'); else setQuery({ view: key }); }, connected: !error && !!latest,
    clientCount: (selected || overview)?.data?.clientCount, domainCount: (selected || overview)?.data?.domains.length,
    openCurrent: running ? () => open(running.id) : undefined }}>
    <main className="platform-main"><header className="platform-header"><div><span className="platform-eyebrow">{view === 'home' ? 'FEDERATED INTELLIGENCE / WORKSPACE' : 'FEDERATED LEARNING / ACCURACY'}</span><h1>{page.label}</h1><p className="platform-page-description">{page.description}</p></div>
      <Space><span className="platform-muted">{latest ? `更新于 ${new Date(latest.at).toLocaleTimeString()}` : '正在连接服务器'}</span><Badge status={error ? 'error' : latest ? 'success' : 'default'} text={error ? '数据已过期' : latest ? '实时连接' : '连接中'} /></Space></header>
      {error && <Alert type="error" showIcon title="服务器连接异常，正在自动重试；已有数据为上次读取的快照" description={error} />}
      {view === 'home' ? <SystemHome catalog={catalog} jobs={jobs} library={library} resource={latest} stale={!!error} /> : !catalog ? <div className="platform-loading"><Spin /><p>正在读取服务器配置与缓存目录</p></div> : <>
      {view === 'overview' && <>
        <div className="platform-intro"><div><h2>多域协同，一处掌握实验全程</h2><p>复用已缓存特征，保留联邦聚合与数据异构；训练、模型和评测形成完整记录。</p></div><Space wrap><Button type="primary" onClick={() => setQuery({ view: 'experience' })}>体验现有模型</Button><Button icon={<PlayCircleOutlined />} onClick={() => setQuery({ view: 'train' })}>新建准确率实验</Button></Space></div>
        <div className="platform-stats"><Stat label="已发现缓存配置" value={catalog.groups.filter(g => g.cacheFound).length} sub="逐样本完整性以预检为准" /><Stat label="正在执行" value={running ? 1 : 0} sub={running?.stage || '服务器可接收任务'} /><Stat label="已保存模型" value={library.models.length} sub="含 final / best 两种检查点" /><Stat label="成功训练" value={jobs.filter(j => j.action === 'train' && j.status === 'completed').length} sub="独立进程 · 完整产物" /></div>
        <div className="platform-grid platform-overview-grid"><Panel title="全域联邦拓扑" extra={<Space><Tag color="cyan">逻辑拓扑 · 非物理多机</Tag>{overview && <State value={overview.status} />}</Space>}>
          <div className="platform-topology-meta"><span>{overview?.request.group || '等待实验配置'}</span><strong>{overview?.data?.clientCount ?? '—'}<small> 逻辑客户端</small></strong><span>{overview?.data?.domains.length ?? '—'} 个数据域</span></div>
          <Topology clients={Object.values(overview?.clients || {})} />
          <div className="platform-flow-board">{['缓存核验', '联邦训练', '模型保存', '独立评测'].map((label, i) => {
            const done = i === 0 ? !!overview?.data : i < 3 ? overview?.status === 'completed' : jobs.some(j => j.action === 'evaluate' && j.status === 'completed' && j.request.modelId?.startsWith(overview?.id || 'none'));
            return <button key={label} className={done ? 'done' : ''} onClick={() => i === 3 ? setQuery({ view: 'evaluate' }) : overview ? open(overview.id) : setQuery({ view: 'train' })}><i>{String(i + 1).padStart(2, '0')}</i><span>{label}<small>{done ? '已完成' : '待执行'}</small></span></button>;
          })}</div>
        </Panel>
          <Panel title="服务器资源" extra={<Tag>本页实时采样</Tag>}>{latest ? <><ResourcesChart history={resources} /><ResourceCards resource={latest} /></> : <Empty description="等待实际资源数据" />}</Panel></div>
        <div className="platform-grid"><Panel title="最近实验的训练曲线" extra={overview && <Button type="link" onClick={() => open(overview.id)}>查看实验 →</Button>}><Curves points={overview?.metrics || []} /></Panel><Panel title="实验记录" extra={<Button type="link" onClick={() => setQuery({ view: 'jobs' })}>全部任务 →</Button>}><JobTable jobs={jobs.filter(j => j.action !== 'inspect').slice(0, 5)} open={open} /></Panel></div>
        <Alert type="info" showIcon title="本架构可按配置重新生成增强数据，或复用已有增强缓存；原始主干特征不重复提取。" />
      </>}
      {view === 'cache' && <><p className="platform-muted">已通过预检的配置优先显示。历史预检不代替本次校验；缓存版本或参数变化会明确报错。</p><div className="platform-cache-grid">{[...catalog.groups].sort((a, b) => Number(b.lastPreflight?.status === 'completed') - Number(a.lastPreflight?.status === 'completed')).map(g => <Card key={g.id} title={g.dataset} extra={<Tag color={g.lastPreflight?.status === 'completed' ? 'success' : g.cacheFound ? 'blue' : 'error'}>{g.lastPreflight?.status === 'completed' ? '上次预检通过' : g.cacheFound ? '缓存已发现' : '缓存缺失'}</Tag>}><span className="platform-backbone">{g.backbone.toUpperCase()}</span><p>{g.domains} 个域 · {g.cacheFiles} 个文件 · {bytes(g.cacheBytes)}</p><p className="platform-muted">{g.methods.filter(m => m.enabled).map(m => m.label).join(' / ')}</p>
        {g.lastPreflight && <p className="platform-muted">最近预检：{new Date(g.lastPreflight.at).toLocaleString()}<br />{g.lastPreflight.error && <span style={{ color: '#f392a0' }}>{g.lastPreflight.error}</span>}</p>}
        <Space wrap><Button disabled={!g.cacheFound || !!error} onClick={() => setQuery({ view: 'train', group: g.id, ...(g.lastPreflight?.status === 'completed' ? { source: g.lastPreflight.id } : {}) })}>{g.lastPreflight?.status === 'completed' ? '复用已核验配置' : '配置与完整预检'}</Button>{g.lastPreflight && <Button type="link" onClick={() => open(g.lastPreflight!.id)}>预检记录</Button>}</Space></Card>)}</div></>}
      {view === 'train' && <TrainingForm catalog={catalog} initialGroup={query.get('group') || undefined} sourceId={query.get('source') || undefined} resources={latest} running={running} disconnected={!!error} create={create} open={open} />}
      {view === 'jobs' && <>{selectedId ? <><Button type="link" onClick={() => setQuery({ view: 'jobs' })}>← 返回任务列表</Button>{detailError && <Alert type="error" title={detailError} />}{showJob ? <JobDetail job={showJob} resources={latest} library={library} stop={stop} rerun={() => setQuery({ view: 'train', source: showJob.id, group: showJob.request.group })} /> : !detailError && <Spin />}</> : <Panel title="所有任务"><JobTable jobs={jobs} open={open} /></Panel>}</>}
      {view === 'evaluate' && <EvaluationPanel library={library} disabled={!!error || !!running} create={create} open={open} />}
      {view === 'experience' && <>{running && <Alert type="info" showIcon title="平台正在执行其他任务，单图预测需等待当前任务结束。" action={<Button onClick={() => open(running.id)}>查看占用任务</Button>} />}<ModelExperience library={library} disabled={!!error || !!running} create={create} open={open} /></>}
      {view === 'compare' && <ComparisonPanel jobs={jobs} open={open} />}
      </>}
      <footer>FederatedScope · frozen-features-v2<span>未接入隐私与后门扩展</span></footer>
    </main>
  </AppShell></div>;
}

function ResourceCards({ resource }: { resource: Resource }) {
  return <div className="platform-resource-cards">{resource.gpus.map(g => <div key={g.index}><div><strong>GPU {g.index}</strong><span>{g.utilization}% · {g.temperature}°C</span></div><Progress percent={Math.round(g.memoryUsedMiB / g.memoryTotalMiB * 100)} showInfo={false} size="small" /><small>{(g.memoryUsedMiB / 1024).toFixed(1)} / {(g.memoryTotalMiB / 1024).toFixed(1)} GB 显存</small></div>)}
    {resource.gpuError && <Alert type="warning" title="GPU 指标不可用" description={resource.gpuError} />}
    <p>内存 {bytes(resource.memoryUsed)} / {bytes(resource.memoryTotal)} · 磁盘剩余 {bytes(resource.diskFree)}</p></div>;
}

function JobTable({ jobs, open }: { jobs: Job[]; open: (id: string) => void }) {
  return <Table size="middle" rowKey="id" dataSource={jobs} scroll={{ x: 650 }} pagination={{ pageSize: 8, hideOnSinglePage: true }} columns={[
    { title: '任务', key: 'name', render: (_, j) => <Button className="platform-job-link" type="link" onClick={() => open(j.id)}>{j.request.name || j.id.slice(0, 8)}<small>{j.request.group} · {methodLabel(j.request.method)}</small></Button> },
    { title: '类型', dataIndex: 'action', render: a => ({ train: '联邦训练', inspect: '缓存预检', evaluate: '独立评测', predict: '单图预测' })[a as string] },
    { title: '状态', dataIndex: 'status', render: value => <State value={value} /> },
    { title: '创建时间', dataIndex: 'createdAt', render: value => new Date(value).toLocaleString() },
  ]} />;
}

export function TrainingForm({ catalog, initialGroup, sourceId, resources, running, disconnected, create, open }: {
  catalog: Catalog; initialGroup?: string; sourceId?: string; resources?: Resource; running?: Job; disconnected: boolean;
  create: (action: string, payload: object) => Promise<Job>; open: (id: string) => void;
}) {
  const [form] = Form.useForm<RequestConfig>();
  const [groupId, setGroupId] = useState(initialGroup || 'officehome_vit');
  const [preflight, setPreflight] = useState<Job>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const group = catalog.groups.find(g => g.id === groupId) || catalog.groups[0];
  const method = Form.useWatch('method', form);
  const augmentationMode = Form.useWatch('augmentationMode', form);
  const ours = method === 'heterogeneous_solution';
  useEffect(() => { const defaults = group.methods.find(m => m.enabled)?.defaults; if (defaults) form.setFieldsValue(defaults); setPreflight(undefined); }, [groupId, form]);
  useEffect(() => {
    if (!sourceId) return;
    let active = true;
    void api<Job>(`jobs/${sourceId}`).then(job => { if (active) form.setFieldsValue(job.request); }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [sourceId, form]);
  useEffect(() => {
    if (!preflight || terminal(preflight.status)) return;
    let active = true;
    const timer = setInterval(async () => { try { const j = await api<Job>(`jobs/${preflight.id}`); if (active) { setPreflight(j); setError(''); } }
      catch (e) { if (active) setError((e as Error).message); } }, 1000);
    return () => { active = false; clearInterval(timer); };
  }, [preflight?.id, preflight?.status]);
  const run = async (action: 'preflight' | 'train') => {
    setBusy(true); setError('');
    try { const validated = await form.validateFields(); const values = { ...form.getFieldsValue(true), ...validated }; const job = await create(action, { ...values, ...(action === 'train' ? { preflightId: preflight?.id } : {}) });
      if (action === 'preflight') setPreflight(job); else open(job.id);
    } catch (e) { setError(e instanceof Error ? e.message : '请检查参数'); } finally { setBusy(false); }
  };
  const info = preflight?.result as DataInfo | undefined;
  const blocked = !!running || disconnected || busy || (!!preflight && !terminal(preflight.status));
  const number = (name: keyof RequestConfig, label: string, min: number, max: number, extra?: string, disabled = false, step = 1) => <Form.Item name={name} label={label} extra={extra} rules={[{ required: true }]}><InputNumber min={min} max={max} step={step} disabled={disabled} style={{ width: '100%' }} /></Form.Item>;
  return <div className="platform-training-grid"><Panel title="训练配置" extra={<Tag>单机模拟多客户端</Tag>}>
    {running && <Alert type="info" title="已有任务占用平台，请等待完成或在“任务与记录”中停止。" action={<Button onClick={() => open(running.id)}>查看任务</Button>} />}
    {error && <Alert type="error" showIcon title={error} />}
    <Form form={form} layout="vertical" onValuesChange={() => setPreflight(undefined)} initialValues={group.methods.find(m => m.enabled)?.defaults}>
      <Form.Item name="name" label="实验名称"><Input maxLength={120} placeholder="例如：Office-Home · FedAvg 基线" /></Form.Item>
      <div className="platform-form-grid"><Form.Item name="group" label="数据集与特征主干"><Select onChange={setGroupId} options={catalog.groups.map(g => ({ value: g.id, label: `${g.dataset} / ${g.backbone.toUpperCase()}`, disabled: !g.cacheFound }))} /></Form.Item>
        <Form.Item name="method" label="算法"><Select onChange={id => { const defaults = group.methods.find(m => m.id === id)?.defaults; if (defaults) { const values = form.getFieldsValue(true); form.setFieldsValue({ ...defaults, ...Object.fromEntries(['rounds', 'localEpochs', 'learningRate', 'batchSize', 'sampleClients', 'clientCount', 'seed', 'splitSeed', 'alpha', 'gpu', 'samplesPerClient', 'evaluationFrequency', 'name'].filter(k => values[k as keyof RequestConfig] !== undefined).map(k => [k, values[k as keyof RequestConfig]])) }); } setPreflight(undefined); }} options={group.methods.map(m => ({ value: m.id, label: methodLabel(m.id), disabled: !m.enabled }))} /></Form.Item>
        {number('rounds', '通信轮数', 1, 1000, '实际模型更新次数，不含第 0 轮初始化')}{number('localEpochs', '本地训练轮数', 1, 100)}
        {number('learningRate', '学习率', 1e-8, 1, undefined, false, .0001)}{number('batchSize', 'Batch size', 1, 1024)}
        {number('clientCount', '客户端总数', group.domains, 240, `必须为 ${group.domains} 的倍数`, group.partitionLocked, group.domains)}{number('sampleClients', '每轮参与客户端数', 0, 240, '0 表示全部参与')}
        <Form.Item name="gpu" label="训练设备"><Select options={[...(resources?.gpus || []).map(g => ({ value: g.index, label: `GPU ${g.index} · ${g.name} · ${g.utilization}% 占用` })), { value: -1, label: 'CPU' }]} /></Form.Item>
        {number('samplesPerClient', '每客户端训练样本数', 0, 100000, '0 使用全部已有样本；正数启用确定性抽样')}
      </div>
      <Collapse ghost items={[{ key: 'advanced', label: '数据异构与复现参数', children: <div className="platform-form-grid">
        {number('alpha', 'Dirichlet α', .0001, 100, '值越小，标签分布差异通常越大', group.partitionLocked, .1)}
        {number('splitSeed', '数据划分种子', 0, 2147483647, '改变后必须匹配已有测试缓存', group.partitionLocked)}
        {number('seed', '训练随机种子', 0, 2147483647)}{number('evaluationFrequency', '每多少轮评测一次', 1, 1000)}
      </div> }]} />
      {ours && <><Form.Item name="augmentationMode" label="增强方式"><Select onChange={() => form.setFieldsValue({ augmentationSourceId: '', allowLegacyAugmentation: false })} options={[{ value: 'generate', label: '重新生成增强数据（按当前训练划分）' }, { value: 'reuse', label: '复用已有增强缓存', disabled: !group.methods.find(m => m.id === 'heterogeneous_solution')?.augmentedCacheFound && !group.augmentationSources?.length }]} /></Form.Item>
        {augmentationMode === 'reuse' && <Form.Item name="augmentationSourceId" label="增强缓存来源"><Select onChange={id => { const source = group.augmentationSources?.find(s => s.id === id); if (source) form.setFieldsValue({ ...source.request, name: '', augmentationSourceId: id, augmentationMode: 'reuse', allowLegacyAugmentation: false }); }} options={[{ value: '', label: '服务器历史缓存（来源有限制）', disabled: !group.methods.find(m => m.id === 'heterogeneous_solution')?.augmentedCacheFound }, ...(group.augmentationSources || []).map(s => ({ value: s.id, label: `${s.name} · ${s.id.slice(0, 8)}` }))]} /></Form.Item>}
        <div className="platform-form-grid">{number('generatedPerSample', '每个原始样本生成数', 0, 1000)}{number('generatedPerPrototype', '每个原型生成数', 0, 1000)}{number('targetPerClass', '每客户端每类目标样本数', 0, 5000, '0 不限制；正数由原算法选择或补齐')}{number('covarianceScale', '生成协方差缩放', 0, 10, '按源算法缩放生成协方差', false, .1)}</div>
        <Alert type={augmentationMode === 'reuse' ? 'warning' : 'info'} showIcon title={augmentationMode === 'reuse' ? '缓存必须匹配当前增强参数和种子；不匹配即失败，不会自动生成。' : '预检不生成数据；点击启动训练后执行统计交换与增强。新缓存只写入本实验目录。'} />
        {augmentationMode === 'reuse' && <Form.Item name="allowLegacyAugmentation" valuePropName="checked"><Checkbox>允许历史缓存试跑：来源记录不完整，结果不能直接作为严格提升证明</Checkbox></Form.Item>}
        <p className="platform-muted">切换算法保留训练预算，便于配对比较；增强参数只对本架构生效。</p>
      </>}
      <div className="platform-form-actions"><Button icon={<DatabaseOutlined />} loading={busy} disabled={blocked} onClick={() => void run('preflight')}>检查完整缓存</Button><Button type="primary" icon={<PlayCircleOutlined />} disabled={blocked || preflight?.status !== 'completed'} onClick={() => void run('train')}>启动训练</Button></div>
    </Form>
  </Panel><div><Panel title="预检结果" extra={preflight && <State value={preflight.status} />}>
    {!preflight ? <Empty description="先检查缓存，预检通过后才能启动" /> : <><p>{preflight.stage}</p>{preflight.error && <Alert type="error" title={preflight.error} />}
      {info && <><div className="platform-stats compact"><Stat label="客户端" value={info.clientCount} /><Stat label="训练样本" value={info.trainSamples} /><Stat label="测试样本" value={info.testSamples} /></div><Topology clients={Object.values(preflight.clients || {})} /><p className="platform-muted">版本 {info.testFingerprint.slice(0, 16)}</p><Alert type="info" title="旧测试缓存按划分种子、样本数和完整标签顺序核验；未内嵌样本 ID 的来源限制保存在复现包中。" /></>}
      {info?.augmentation?.warning && <Alert type="warning" title={info.augmentation.warning} />}
      {info?.augmentation?.mode === 'generate' && <Alert type="info" title="原始训练/测试特征已核验，启动后重新生成增强数据。" />}
      <Button type="link" onClick={() => open(preflight.id)}>查看预检日志与配置</Button></>}
  </Panel><div className="platform-note"><strong>缓存策略</strong><p>不下载主干、不重复提取原始特征。增强方式由配置决定；失败只回收本任务进程，保留日志和产物。</p></div></div></div>;
}

function JobDetail({ job, resources, library, stop, rerun }: { job: Job; resources?: Resource; library: Library; stop: (id: string) => void; rerun: () => void }) {
  const [logs, setLogs] = useState('');
  useEffect(() => { let active = true; const update = async () => { try { const value = await api<string>(`jobs/${job.id}/logs`); if (active) setLogs(value); }
    catch (e) { if (active) setLogs(`日志读取失败：${(e as Error).message}`); } }; void update(); const timer = setInterval(update, 2500);
    return () => { active = false; clearInterval(timer); }; }, [job.id]);
  const clients = Object.values(job.clients || {}), points = job.metrics || [];
  const final = points.at(-1);
  return <><div className="platform-job-heading"><div><h2>{job.request.name || job.id.slice(0, 8)} <State value={job.status} /></h2><p>{job.stage} · {job.request.group} / {methodLabel(job.request.method)}</p></div><Space wrap>
    {!terminal(job.status) && <Popconfirm title="停止当前任务？" description="仅回收此任务进程，保留配置、日志及已生成文件。" onConfirm={() => stop(job.id)}><Button danger icon={<StopOutlined />}>停止任务</Button></Popconfirm>}
    {terminal(job.status) && ['train', 'inspect'].includes(job.action) && <Button onClick={rerun} icon={<ReloadOutlined />}>复用配置重跑</Button>}
    <Button href={`/api/platform/jobs/${job.id}/export`} icon={<DownloadOutlined />}>结果 JSON</Button>
    {terminal(job.status) && <Button href={`/api/platform/jobs/${job.id}/bundle`}>完整复现包</Button>}</Space></div>
    {job.error && <Alert type="error" title={job.error} showIcon />}
    {job.data?.augmentation?.warning && <Alert type="warning" title={job.data.augmentation.warning} showIcon />}
    {job.request.method === 'heterogeneous_solution' && <Alert type="info" title={`${job.request.augmentationMode === 'generate' ? '按配置重新生成' : '复用已有增强缓存'} · 已保存增强样本 ${job.data?.augmentation?.cachedSamples ?? '—'}；同轮数不代表相同训练样本量。`} />}
    {terminal(job.status) && <Alert type={job.cleanup.ok ? 'success' : 'error'} title={job.cleanup.message} action={!job.cleanup.ok && <Button onClick={() => stop(job.id)}>重试本任务清理</Button>} />}
    {job.action === 'predict' ? <PredictionPanel job={job} /> : job.action === 'evaluate' ? <EvaluationResults job={job} library={library} /> : <div className="platform-stats"><Stat label="已评测轮次" value={final ? `${final.round} / ${job.request.rounds}` : '—'} /><Stat label="总体准确率" value={percent(final?.accuracy)} sub="测试样本加权" /><Stat label="分域平均" value={percent(final?.domainMean)} sub="各域等权" /><Stat label="最差域准确率" value={percent(final?.worstDomain)} /></div>}
    <Tabs items={[{ key: 'monitor', label: '训练监控', children: <><div className="platform-grid"><Panel title="全局模型准确率"><Curves points={points} /></Panel><Panel title="单机联邦拓扑"><Topology clients={clients} /></Panel></div><div className="platform-grid"><Panel title="本地训练损失"><LossChart points={points} /></Panel><Panel title="资源占用">{resources ? <ResourceCards resource={resources} /> : <Empty />}</Panel></div>
      <Panel title="客户端状态"><Table<Client> size="small" rowKey="id" dataSource={clients} pagination={{ pageSize: 12 }} columns={[{ title: '客户端', dataIndex: 'id' }, { title: '域', dataIndex: 'domain' }, { title: '原始样本', dataIndex: 'samples' }, { title: '状态', dataIndex: 'stage' }, { title: '轮次', dataIndex: 'round', render: n => n == null ? '—' : n + 1 }, { title: '训练损失', dataIndex: 'loss', render: n => n?.toFixed(4) ?? '—' }, { title: '本地训练准确率', dataIndex: 'accuracy', render: percent }]} /></Panel></> },
      { key: 'data', label: '数据异构', children: <Panel title="客户端 × 类别：实际训练样本分布"><Distribution clients={clients} classes={job.data?.classes || []} /><p className="platform-muted">此图为原始划分；若配置每客户端抽样，实际训练样本数另保存在客户端日志。</p></Panel> },
      { key: 'config', label: '参数与溯源', children: <><Descriptions bordered column={2} items={Object.entries(job.request).map(([k, v]) => ({ key: k, label: k, children: JSON.stringify(v) }))} /><Panel title="实际配置与版本"><pre className="platform-code">{JSON.stringify({ config: job.config, provenance: job.provenance, data: job.data }, null, 2)}</pre></Panel></> },
      { key: 'logs', label: '完整运行日志', children: <Panel title="最近 128 KiB；全量日志请下载复现包"><pre className="platform-log">{logs || '尚无日志'}</pre></Panel> }].filter(tab => !['evaluate', 'predict'].includes(job.action) || ['config', 'logs'].includes(tab.key))} />
  </>;
}
