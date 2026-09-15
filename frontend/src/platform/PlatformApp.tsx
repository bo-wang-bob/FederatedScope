import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import { App as AntApp, Alert, Button, ConfigProvider, Input, Select, Skeleton, Space } from 'antd';
import { ArrowLeftOutlined, ArrowRightOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import { api, key, methodLabel, terminal, type Catalog, type Job, type Library } from './api';
import { ComparisonPanel, EvaluationPanel } from './evaluation';
import { ModelExperience } from './inference';
import { SystemHome } from './home';
import { mainView, modelHref, pages, resolveView, viewHref, type PlatformView } from './navigation';
import { isPlannedView, PlannedModule } from './extensions';
import { StudioShell } from './StudioShell';
import { TrainingForm } from './training';
import { JobDetail, JobTable } from './jobs';
import { useTrainingLaunch } from './launch';
import { researchTheme } from './theme';
import './studio.css';
export { TrainingForm } from './training';

export default function PlatformApp() {
  return <ConfigProvider theme={researchTheme}>
    <AntApp><Workspace /></AntApp>
  </ConfigProvider>;
}
function pendingRequests(): Map<string,string> {
  try { return new Map(JSON.parse(sessionStorage.getItem('federated-studio.requests.v2') || '[]')); } catch { return new Map(); }
}
function Workspace() {
  const [query,setQuery] = useSearchParams(), location = useLocation();
  const view = resolveView(query.get('view'),location.pathname), selectedId = view === 'jobs' ? query.get('id') : null;
  const page=pages[view], area=mainView(view), modelId=query.get('model') || undefined, testsetId=query.get('testset') || undefined;
  const [catalog,setCatalog]=useState<Catalog>(), [jobs,setJobs]=useState<Job[]>([]);
  const [library,setLibrary]=useState<Library>({models:[],testsets:[]}), [libraryError,setLibraryError]=useState('');
  const [selected,setSelected]=useState<Job>(), [error,setError]=useState(''), [detailError,setDetailError]=useState('');
  const [refreshKey,setRefreshKey]=useState(0), [search,setSearch]=useState(''), [jobFilter,setJobFilter]=useState('train');
  const [statusFilter,setStatusFilter]=useState('all');
  const {message}=AntApp.useApp(), pending=useRef(pendingRequests());
  const open=useCallback((id:string) => { setQuery({view:'jobs',id});setRefreshKey(value => value+1); },[setQuery]);
  const launch=useTrainingLaunch(open);
  useEffect(() => { document.title=page.label+' · 跨域联邦学习'; },[page.label]);
  useEffect(() => { window.scrollTo(0,0); },[view,selectedId]);
  useEffect(() => {
    let alive=true,busy=false;
    const refresh=async () => {
      if(busy) return; busy=true;
      const results=await Promise.allSettled([api<Catalog>('catalog'),api<Job[]>('jobs'),api<Library>('library')]);
      if(alive) {
        const [c,j,l]=results;
        if(c.status === 'fulfilled') setCatalog(c.value);
        if(j.status === 'fulfilled') setJobs(j.value);
        if(l.status === 'fulfilled') { setLibrary(l.value);setLibraryError(''); } else setLibraryError(String(l.reason?.message || l.reason));
        const failed=[c,j].find(r => r.status === 'rejected');
        setError(failed?.status === 'rejected' ? String(failed.reason?.message || failed.reason) : '');
      }
      busy=false;
    };
    void refresh(); const timer=setInterval(refresh,5000);
    return () => {alive=false;clearInterval(timer);};
  },[refreshKey]);
  useEffect(() => {
    let alive=true,busy=false;
    setSelected(undefined);setDetailError('');
    if(!selectedId) return;
    const refresh=async () => {
      if(busy) return;busy=true;
      try { const job=await api<Job>('jobs/'+selectedId);if(alive) {setSelected(job);setDetailError('');} }
      catch(e) {if(alive) setDetailError((e as Error).message);}
      finally {busy=false;}
    };
    void refresh();const timer=setInterval(refresh,1800);
    return () => {alive=false;clearInterval(timer);};
  },[selectedId,refreshKey]);
  const create=async (action:string,payload:object) => {
    const signature=JSON.stringify([action,payload]);
    if(!pending.current.has(signature)) pending.current.set(signature,key());
    const save=() => {try {sessionStorage.setItem('federated-studio.requests.v2',JSON.stringify([...pending.current]));}catch{/* Keep pending keys in memory. */}};
    save();
    const job=await api<Job>(action,{...payload,idempotencyKey:pending.current.get(signature)});
    pending.current.delete(signature);save();
    setJobs(old => [job,...old.filter(j => j.id !== job.id)]);
    return job;
  };
  const stop=async (id:string) => {
    try {const job=await api<Job>('jobs/'+id+'/stop',{});setSelected(job);setJobs(old => old.map(j => j.id === id ? job : j));message.success('停止请求已处理');}
    catch(e) {message.error((e as Error).message);}
  };
  const running=jobs.find(job => !terminal(job.status));
  const visibleJobs=jobs.filter(job => (jobFilter === 'all' ? job.action !== 'inspect' : job.action === jobFilter) && (statusFilter === 'all' || job.status === statusFilter) &&
    (job.request.name+' '+job.request.group+' '+methodLabel(job.request.method)+' '+job.id).toLowerCase().includes(search.toLowerCase()));
  const routeTabs = area === 'train' ? [{view:'train',label:'新建训练'},{view:'jobs',label:'实验记录'}] :
    area === 'experience' ? [{view:'experience',label:'单图验证'},{view:'evaluate',label:'独立评测'}] : [];
  return <StudioShell view={view} connected={!!catalog && !error} running={!!running} openCurrent={() => running && open(running.id)}>
    {error && <Alert className="studio-connection-alert" type="error" showIcon title="连接中断，正在重试" description={error} action={<Button icon={<ReloadOutlined />} onClick={() => setRefreshKey(v => v+1)}>重连</Button>} />}
    {launch.intent && <div className={'launch-banner '+launch.intent.phase} role="status"><span className="launch-symbol">{launch.busy ? <i className="live-dot" /> : '!'}</span><div><strong>{launch.intent.phase === 'checking' ? launch.intent.cancelRequested ? '正在取消启动…' : '正在检查数据与配置…' : launch.intent.phase === 'starting' ? '正在启动训练…' : launch.intent.phase === 'blocked' ? '本次启动未通过检查' : '需要确认上次启动状态'}</strong><span>{launch.intent.error || launch.intent.request.name}</span></div>
      {launch.intent.phase === 'checking' && <Button size="small" disabled={launch.intent.cancelRequested} onClick={launch.cancel}>取消启动</Button>}
      {launch.intent.phase === 'recover' && <><Button onClick={launch.resume}>{launch.intent.trainRequested ? '确认任务状态' : launch.intent.cancelRequested ? '继续取消' : '继续启动'}</Button>{!launch.intent.trainRequested && !launch.intent.cancelRequested && <Button onClick={launch.cancel}>取消本次启动</Button>}</>}
      {launch.intent.phase === 'blocked' && <Button onClick={() => {launch.edit();setQuery({view:'train'});}}>返回修改</Button>}
      {view !== 'train' && <Link to="/?view=train">查看配置 <ArrowRightOutlined /></Link>}
    </div>}
    {view !== 'home' && <div className="studio-page-heading"><div><h1>{selectedId ? '实验详情' : area === 'experience' ? '模型验证' : page.label}</h1></div>
      <div>{routeTabs.length > 0 && <nav className="studio-route-tabs" aria-label="模块功能">{routeTabs.map(tab => <Link aria-current={view === tab.view ? 'page' : undefined} className={view === tab.view ? 'active' : ''} key={tab.view} to={modelId && area === 'experience' ? modelHref(modelId,tab.view === 'evaluate',testsetId) : viewHref(tab.view as PlatformView)}>{tab.label}</Link>)}</nav>}
      </div></div>}
    {view === 'home' ? <SystemHome jobs={jobs} loading={!catalog} /> : isPlannedView(view) ? <PlannedModule moduleId={view} /> : !catalog ? <div className="studio-loading"><Skeleton active paragraph={{rows:8}} /></div> : <>
      {view === 'train' && <TrainingForm key={(query.get('group') || '')+':'+(query.get('source') || '')} catalog={catalog} initialGroup={query.get('group') || undefined} sourceId={query.get('source') || undefined} running={running} disconnected={!!error} launch={launch} open={open} />}
      {view === 'jobs' && <div className="studio-page-enter">{selectedId ? <><Button className="studio-back" type="text" icon={<ArrowLeftOutlined />} onClick={() => setQuery({view:'jobs'})}>返回实验记录</Button>{detailError && <Alert type="error" title={detailError} action={<Button onClick={() => setRefreshKey(x => x+1)}>重试</Button>} />}{selected ? <JobDetail key={selected.id} job={selected} library={library} stop={stop} rerun={() => setQuery({view:'train',source:selected.id,group:selected.request.group})} /> : !detailError && <Skeleton active />}</> :
        <section className="studio-surface jobs-collection"><div className="collection-toolbar"><Input aria-label="搜索实验" prefix={<SearchOutlined />} placeholder="搜索实验名称或算法" allowClear value={search} onChange={event => setSearch(event.target.value)} /><Space>
          <Select aria-label="任务类型" value={jobFilter} onChange={setJobFilter} options={[{value:'train',label:'训练实验'},{value:'evaluate',label:'独立评测'},{value:'predict',label:'单图验证'},{value:'all',label:'所有实验'}]} />
          <Select aria-label="实验状态" value={statusFilter} onChange={setStatusFilter} options={[{value:'all',label:'全部状态'},{value:'completed',label:'已完成'},{value:'running',label:'运行中'},{value:'failed',label:'失败'},{value:'stopped',label:'已停止'},{value:'interrupted',label:'已中断'}]} /></Space></div><JobTable jobs={visibleJobs} open={open} /></section>}</div>}
      {area === 'experience' && libraryError && <Alert type="error" title={'模型库读取失败：'+libraryError} action={<Button onClick={() => setRefreshKey(x => x+1)}>重试</Button>} />}
      {area === 'experience' && running && <Alert className="studio-busy" type="info" showIcon title="当前有任务运行中" action={<Button onClick={() => open(running.id)}>查看任务</Button>} />}
      {view === 'experience' && <div className="studio-page-enter"><ModelExperience key={modelId || 'default'} initialModel={modelId} initialTestset={testsetId} onSelectionChange={(model,testset) => setQuery({view,model,...(testset ? {testset} : {})})} library={library} disabled={!!error || !!libraryError || !!running || !!launch.intent} create={create} open={open} /></div>}
      {view === 'evaluate' && <div className="studio-page-enter"><EvaluationPanel key={modelId || 'default'} initialModel={modelId} initialTestset={testsetId} onSelectionChange={(model,testset) => setQuery({view,model,...(testset ? {testset} : {})})} library={library} disabled={!!error || !!libraryError || !!running || !!launch.intent} create={create} open={open} /></div>}
      {view === 'compare' && <div className="studio-page-enter"><ComparisonPanel jobs={jobs} open={open} /></div>}
    </>}
  </StudioShell>;
}
