import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import { App as AntApp, Alert, Badge, Button, Card, ConfigProvider, Empty, Input, Select, Skeleton, Space, Tag } from 'antd';
import { ArrowLeftOutlined, DatabaseOutlined, PlusOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import { api, bytes, key, methodLabel, terminal, type Catalog, type Job, type Library, type Resource } from './api';
import { ResourcesChart, Topology } from './charts';
import { ComparisonPanel, EvaluationPanel } from './evaluation';
import { ModelExperience } from './inference';
import { SystemHome } from './home';
import { mainView, pages, resolveView, viewHref, type PlatformView } from './navigation';
import { isPlannedView, PlannedModule } from './extensions';
import { StudioShell } from './StudioShell';
import { TrainingForm } from './training';
import { JobDetail, JobTable, ResourceCards } from './jobs';
import { Panel } from './ui';
import './studio.css';
export { TrainingForm } from './training';

export default function PlatformApp() {
  return <ConfigProvider theme={{ token: { colorPrimary: '#7258ed', colorSuccess: '#219b76', colorInfo: '#7258ed', colorWarning: '#bc8222',
    borderRadius: 10, fontSize: 14, fontFamily: 'Inter, "PingFang SC", "Microsoft YaHei", sans-serif',
    colorBgBase: '#ffffff', colorBgLayout: '#f6f7fb', colorBgContainer: '#ffffff', colorText: '#242539', colorTextSecondary: '#7c7e90', colorBorder: '#e3e5ee' },
    components: { Button: { controlHeight: 40, primaryShadow: '0 4px 12px #7258ed24' }, Input: { controlHeight: 40 }, InputNumber: { controlHeight: 40 }, Select: { controlHeight: 40 },
      Card: { headerFontSize: 15, headerHeight: 62 }, Table: { headerBg: '#f8f9fc', headerColor: '#777a8e', rowHoverBg: '#f8f6ff', cellPaddingBlock: 15 } } }}>
    <AntApp><Workspace /></AntApp>
  </ConfigProvider>;
}

function Workspace() {
  const [query, setQuery] = useSearchParams(), location = useLocation();
  const view = resolveView(query.get('view'), location.pathname), selectedId = view === 'jobs' ? query.get('id') : null;
  const page = pages[view], area = mainView(view);
  const [catalog, setCatalog] = useState<Catalog>();
  const [jobs, setJobs] = useState<Job[]>([]), [library, setLibrary] = useState<Library>({ models: [], testsets: [] });
  const [resources, setResources] = useState<Resource[]>([]), [selected, setSelected] = useState<Job>(), [overview, setOverview] = useState<Job>();
  const [error, setError] = useState(''), [detailError, setDetailError] = useState(''), [refreshKey, setRefreshKey] = useState(0);
  const [search, setSearch] = useState(''), [jobFilter, setJobFilter] = useState('all');
  const { message } = AntApp.useApp();
  const pending = useRef(new Map<string, string>());
  const open = useCallback((id: string) => setQuery({ view: 'jobs', id }), [setQuery]);
  useEffect(() => { document.title = `${page.label} · 全域智汇`; }, [page.label]);
  useEffect(() => { window.scrollTo(0, 0); }, [view, selectedId]);
  useEffect(() => {
    let alive = true, busy = false;
    const refresh = async () => {
      if (busy) return; busy = true;
      try {
        const [c, j, l, r] = await Promise.all([api<Catalog>('catalog'), api<Job[]>('jobs'), api<Library>('library'), api<Resource>('resources')]);
        const training = j.find(job => job.action === 'train' && !terminal(job.status)) || j.find(job => job.action === 'train' && job.status === 'completed') || j.find(job => job.action === 'train');
        const detail = training ? await api<Job>(`jobs/${training.id}`) : undefined;
        if (alive) { setCatalog(c); setJobs(j); setLibrary(l); setResources(old => [...old, r].slice(-60)); setOverview(detail); setError(''); }
      } catch (e) { if (alive) setError((e as Error).message); }
      finally { busy = false; }
    };
    void refresh(); const timer = setInterval(refresh, 4000);
    return () => { alive = false; clearInterval(timer); };
  }, [refreshKey]);
  useEffect(() => {
    let alive = true, busy = false;
    setSelected(undefined); setDetailError('');
    if (!selectedId) return;
    const refresh = async () => {
      if (busy) return; busy = true;
      try { const job = await api<Job>(`jobs/${selectedId}`); if (alive) { setSelected(job); setDetailError(''); } }
      catch (e) { if (alive) setDetailError((e as Error).message); }
      finally { busy = false; }
    };
    void refresh(); const timer = setInterval(refresh, 1500);
    return () => { alive = false; clearInterval(timer); };
  }, [selectedId]);
  const create = async (action: string, payload: object) => {
    const signature = JSON.stringify([action, payload]);
    if (!pending.current.has(signature)) pending.current.set(signature, key());
    const job = await api<Job>(action, { ...payload, idempotencyKey: pending.current.get(signature) });
    pending.current.delete(signature);
    setJobs(old => [job, ...old.filter(j => j.id !== job.id)]);
    return job;
  };
  const stop = async (id: string) => {
    try { setSelected(await api<Job>(`jobs/${id}/stop`, {})); message.success('停止请求已处理'); }
    catch (e) { message.error((e as Error).message); }
  };
  const latest = resources.at(-1), running = jobs.find(job => !terminal(job.status));
  const visibleJobs = jobs.filter(job => (jobFilter === 'all' || job.action === jobFilter) &&
    `${job.request.name} ${job.request.group} ${methodLabel(job.request.method)} ${job.id}`.toLowerCase().includes(search.toLowerCase()));
  const routeTabs = area === 'train' ? [
    { view: 'train', label: '新建实验' }, { view: 'jobs', label: '实验记录' }, { view: 'cache', label: '数据缓存' }, { view: 'overview', label: '运行资源' },
  ] : area === 'experience' ? [{ view: 'experience', label: '单图预测' }, { view: 'evaluate', label: '独立评测' }] : [];
  return <StudioShell view={view} connected={!error && !!latest} running={!!running} openCurrent={() => running && open(running.id)}>
    {view !== 'home' && <div className="studio-page-heading"><div><span className="studio-kicker">{isPlannedView(view) ? 'RESEARCH EXTENSIONS' : 'FEDERATED STUDIO'}</span><h1>{page.label}</h1></div><div>{!isPlannedView(view) && <Badge status={error ? 'error' : latest ? 'success' : 'default'} text={error ? '离线快照' : latest ? '已同步' : '连接中'} />}{view === 'jobs' && <Button type="primary" icon={<PlusOutlined />} onClick={() => setQuery({ view: 'train' })}>新建实验</Button>}</div></div>}
    {routeTabs.length > 0 && <nav className="studio-route-tabs" aria-label="模块功能">{routeTabs.map(tab => <Link aria-current={view === tab.view ? 'page' : undefined} className={view === tab.view ? 'active' : ''} key={tab.view} to={viewHref(tab.view as PlatformView)}>{tab.label}</Link>)}</nav>}
    {error && <Alert className="studio-connection-alert" type="error" showIcon title="连接中断，正在重试" description={error} action={<Button icon={<ReloadOutlined />} onClick={() => setRefreshKey(value => value + 1)}>重连</Button>} />}
    {view === 'home' ? <SystemHome catalog={catalog} jobs={jobs} library={library} resource={latest} stale={!!error} overview={overview} /> : isPlannedView(view) ? <PlannedModule moduleId={view} /> : !catalog ? <div className="studio-loading"><Skeleton active paragraph={{ rows: 8 }} /></div> : <>
      {area === 'train' && <div hidden={view !== 'train'}><TrainingForm key={`${query.get('group') || ''}:${query.get('source') || ''}`} catalog={catalog} initialGroup={query.get('group') || undefined} sourceId={query.get('source') || undefined} resources={latest} running={running} disconnected={!!error} create={create} open={open} /></div>}
      {view === 'jobs' && <div className="studio-page-enter">{selectedId ? <><Button className="studio-back" type="text" icon={<ArrowLeftOutlined />} onClick={() => setQuery({ view: 'jobs' })}>返回实验记录</Button>{detailError && <Alert type="error" title={detailError} />}{selected ? <JobDetail key={selected.id} job={selected} resources={latest} library={library} stop={stop} rerun={() => setQuery({ view: 'train', source: selected.id, group: selected.request.group })} /> : !detailError && <Skeleton active />}</> : <Panel title="所有任务" extra={<Space wrap><Input aria-label="搜索实验" prefix={<SearchOutlined />} placeholder="搜索实验" allowClear value={search} onChange={event => setSearch(event.target.value)} /><Select aria-label="任务类型" value={jobFilter} onChange={setJobFilter} options={[{ value: 'all', label: '全部类型' }, { value: 'train', label: '训练' }, { value: 'evaluate', label: '评测' }, { value: 'predict', label: '预测' }, { value: 'inspect', label: '预检' }]} /></Space>}><JobTable jobs={visibleJobs} open={open} /></Panel>}</div>}
      {view === 'cache' && <div className="platform-cache-grid studio-page-enter">{[...catalog.groups].sort((a, b) => Number(b.lastPreflight?.status === 'completed') - Number(a.lastPreflight?.status === 'completed')).map(group => <Card key={group.id} className="studio-cache-card" title={<Space><DatabaseOutlined />{group.dataset}</Space>} extra={<Tag color={group.lastPreflight?.status === 'completed' ? 'success' : group.cacheFound ? 'processing' : 'default'}>{group.lastPreflight?.status === 'completed' ? '曾通过预检' : group.cacheFound ? '已发现' : '缺失'}</Tag>}><h2>{group.backbone.toUpperCase()}</h2><p>{group.domains} 个域 <i>·</i> {group.cacheFiles} 个文件 <i>·</i> {bytes(group.cacheBytes)}</p><div className="studio-cache-methods">{group.methods.filter(method => method.enabled).map(method => <Tag key={method.id}>{methodLabel(method.id)}</Tag>)}</div>{group.lastPreflight?.error && <Alert type="warning" title={group.lastPreflight.error} />}<div className="studio-cache-actions"><Button disabled={!group.cacheFound || !!error} onClick={() => setQuery({ view: 'train', group: group.id, ...(group.lastPreflight?.status === 'completed' ? { source: group.lastPreflight.id } : {}) })}>使用配置</Button>{group.lastPreflight && <Button type="text" onClick={() => open(group.lastPreflight!.id)}>预检记录</Button>}</div></Card>)}</div>}
      {view === 'overview' && <div className="platform-grid studio-page-enter"><Panel title="服务器资源">{latest ? <><ResourcesChart history={resources} /><ResourceCards resource={latest} /></> : <Empty />}</Panel><Panel title="客户端拓扑" extra={<Tag>逻辑客户端</Tag>}><Topology clients={Object.values(overview?.clients || {})} />{overview && <Button type="text" onClick={() => open(overview.id)}>查看关联实验</Button>}</Panel></div>}
      {area === 'experience' && running && <Alert className="studio-busy" type="info" showIcon title="当前有任务运行中" action={<Button onClick={() => open(running.id)}>查看任务</Button>} />}
      {view === 'experience' && <div className="studio-page-enter"><ModelExperience library={library} disabled={!!error || !!running} create={create} open={open} /></div>}
      {view === 'evaluate' && <div className="studio-page-enter"><EvaluationPanel library={library} disabled={!!error || !!running} create={create} open={open} /></div>}
      {view === 'compare' && <div className="studio-page-enter"><ComparisonPanel jobs={jobs} open={open} /></div>}
    </>}
    <footer className="studio-footer"><span>全域智汇 <i>/</i> FederatedScope</span><span>{latest && !error ? `更新于 ${new Date(latest.at).toLocaleTimeString()}` : '等待连接'}</span></footer>
  </StudioShell>;
}
