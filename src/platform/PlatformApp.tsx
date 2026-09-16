import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import { App as AntApp, Alert, Button, ConfigProvider, Input, Select, Skeleton, Space } from 'antd';
import { ArrowLeftOutlined, ArrowRightOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import zhCN from 'antd/locale/zh_CN';
import { api, key, methodLabel, terminal, type Catalog, type Job, type Library } from './api';
import { BackdoorLab } from './backdoor';
import { BackdoorCompare } from './backdoorCompare';
import { ComparisonPanel, EvaluationPanel } from './evaluation';
import { ModelExperience } from './inference';
import { SystemHome } from './home';
import { mainView, modelHref, pages, resolveView, viewHref, backdoorCompareHref, type PlatformView } from './navigation';
import { isPlannedView, PlannedModule } from './extensions';
import { StudioShell } from './StudioShell';
import { TrainingForm } from './training';
import { JobDetail, JobTable } from './jobs';
import { useTrainingLaunch } from './launch';
import { researchTheme } from './theme';
import { aircraftDemoEnabled, DEMO_DRAFT_KEY, DEMO_LAUNCH_KEY, presentationCatalog, presentationJobs, presentationLibrary, requestInPresentation } from './presentationScope';
import './studio.css';
import '../design/design.css';
import './live.css';
export { TrainingForm } from './training';

export default function PlatformApp() {
  return <ConfigProvider theme={researchTheme} locale={zhCN}>
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
  const [allCatalog,setCatalog]=useState<Catalog>(), [jobs,setJobs]=useState<Job[]>([]);
  const [allLibrary,setLibrary]=useState<Library>({models:[],testsets:[]}), [libraryError,setLibraryError]=useState('');
  const restricted=aircraftDemoEnabled(), catalog=allCatalog && presentationCatalog(allCatalog);
  const library=presentationLibrary(allLibrary,allCatalog), scopedJobs=presentationJobs(jobs,allCatalog);
  const [selected,setSelected]=useState<Job>(), [error,setError]=useState(''), [detailError,setDetailError]=useState('');
  const [refreshKey,setRefreshKey]=useState(0), [search,setSearch]=useState(''), [jobFilter,setJobFilter]=useState('train');
  const [statusFilter,setStatusFilter]=useState('all');
  const {message}=AntApp.useApp(), pending=useRef(pendingRequests());
  const open=useCallback((id:string) => { setQuery({view:'jobs',id});setRefreshKey(value => value+1); },[setQuery]);
  const launch=useTrainingLaunch(open,restricted ? DEMO_LAUNCH_KEY : undefined);
  useEffect(() => { document.title=page.label+' · 跨域协同训练'; },[page.label]);
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
    if (restricted && ['predict','evaluate'].includes(action)) {
      const request=payload as {modelId?:string;testsetId?:string};
      if (!library.models.some(model=>model.id===request.modelId) || !library.testsets.some(test=>test.id===request.testsetId))
        throw new Error('模型或测试集不在当前演示范围');
    }
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
  const visibleJobs=scopedJobs.filter(job => (jobFilter === 'all' ? job.action !== 'inspect' : job.action === jobFilter) && (statusFilter === 'all' || job.status === statusFilter) &&
    (job.request.name+' '+job.request.group+' '+methodLabel(job.request.method)+' '+job.id).toLowerCase().includes(search.toLowerCase()));
  const jobIdFromQuery = query.get('job');
  const routeTabs: { view: PlatformView; label: string; href?: string }[] = area === 'train' ? [{view:'train',label:'新建训练'},{view:'jobs',label:'实验记录'}] :
    area === 'experience' ? [{view:'experience',label:'单图验证'},{view:'evaluate',label:'独立评测'}] :
    area === 'backdoor' && jobIdFromQuery ? [{view:'backdoorCompare',label:'逐样本对照',href:backdoorCompareHref(jobIdFromQuery)}] : [];
  return <StudioShell view={view} connected={!!catalog && !error} running={!!running} openCurrent={() => running && open(running.id)} showDatasetImages={!restricted}>
    {error && <Alert className="studio-connection-alert" type="error" showIcon title="连接中断，正在重试" description={error} action={<Button icon={<ReloadOutlined />} onClick={() => setRefreshKey(v => v+1)}>重连</Button>} />}
    {launch.intent && <div className={'launch-banner '+launch.intent.phase} role="status"><span className="launch-symbol">{launch.busy ? <i className="live-dot" /> : '!'}</span><div><strong>{launch.intent.phase === 'checking' ? launch.intent.cancelRequested ? '正在取消启动…' : '正在检查数据与配置…' : launch.intent.phase === 'starting' ? '正在启动训练…' : launch.intent.phase === 'blocked' ? '本次启动未通过检查' : '需要确认上次启动状态'}</strong><span>{launch.intent.error || launch.intent.request.name}</span></div>
      {launch.intent.phase === 'checking' && <Button size="small" disabled={launch.intent.cancelRequested} onClick={launch.cancel}>取消启动</Button>}
      {launch.intent.phase === 'recover' && <><Button onClick={launch.resume}>{launch.intent.trainRequested ? '确认任务状态' : launch.intent.cancelRequested ? '继续取消' : '继续启动'}</Button>{!launch.intent.trainRequested && !launch.intent.cancelRequested && <Button onClick={launch.cancel}>取消本次启动</Button>}</>}
      {launch.intent.phase === 'blocked' && <Button onClick={() => {launch.edit();setQuery({view:'train'});}}>返回修改</Button>}
      {view !== 'train' && <Link to="/?view=train">查看配置 <ArrowRightOutlined /></Link>}
    </div>}
    {view !== 'home' && <div className="studio-page-heading"><div><h1>{selectedId ? '实验详情' : area === 'experience' ? '模型验证' : page.label}</h1></div>
      <div>{routeTabs.length > 0 && <nav className="studio-route-tabs" aria-label="模块功能">{routeTabs.map(tab => <Link aria-current={view === tab.view ? 'page' : undefined} className={view === tab.view ? 'active' : ''} key={tab.view} to={tab.href ?? (modelId && area === 'experience' ? modelHref(modelId,tab.view === 'evaluate',testsetId) : viewHref(tab.view))}>{tab.label}</Link>)}</nav>}
      </div></div>}
    {view === 'home' ? <SystemHome jobs={scopedJobs} loading={!catalog} showDatasetImages={!restricted} /> : isPlannedView(view) ? <PlannedModule moduleId={view} /> : view === 'backdoor' || view === 'backdoorCompare' ? <div className="studio-page-enter">{view === 'backdoor' ? <BackdoorLab /> : <BackdoorCompare />}</div> : !catalog ? <div className="studio-loading"><Skeleton active paragraph={{rows:8}} /></div> : <>
      {view === 'train' && (catalog.groups.length ? <TrainingForm key={(query.get('group') || '')+':'+(query.get('source') || '')} catalog={catalog} draftKey={restricted ? DEMO_DRAFT_KEY : undefined} initialGroup={query.get('group') || undefined} sourceId={query.get('source') || undefined} running={running} disconnected={!!error} launch={launch} open={open} /> : <div className="studio-empty-state"><h2>训练配置尚未接入</h2></div>)}
      {view === 'jobs' && <div className="studio-page-enter">{selectedId ? <><Button className="studio-back" type="text" icon={<ArrowLeftOutlined />} onClick={() => setQuery({view:'jobs'})}>返回实验记录</Button>{detailError && <Alert type="error" title={detailError} action={<Button onClick={() => setRefreshKey(x => x+1)}>重试</Button>} />}{selected ? requestInPresentation(selected.request,allCatalog) || !terminal(selected.status) ? <JobDetail key={selected.id} job={selected} library={library} stop={stop} rerun={() => setQuery({view:'train',source:selected.id,group:selected.request.group})} /> : <Alert type="info" title="该记录不在当前演示范围" /> : !detailError && <Skeleton active />}</> :
        <section className="studio-surface jobs-collection"><div className="collection-toolbar"><Input aria-label="搜索实验" prefix={<SearchOutlined />} placeholder="搜索实验名称或算法" allowClear value={search} onChange={event => setSearch(event.target.value)} /><Space>
          <Select aria-label="任务类型" value={jobFilter} onChange={setJobFilter} options={[{value:'train',label:'训练实验'},{value:'evaluate',label:'独立评测'},{value:'predict',label:'单图验证'},{value:'all',label:'所有实验'}]} />
          <Select aria-label="实验状态" value={statusFilter} onChange={setStatusFilter} options={[{value:'all',label:'全部状态'},{value:'completed',label:'已完成'},{value:'running',label:'运行中'},{value:'failed',label:'失败'},{value:'stopped',label:'已停止'},{value:'interrupted',label:'已中断'}]} /></Space></div><JobTable jobs={visibleJobs} open={open} /></section>}</div>}
      {area === 'experience' && libraryError && <Alert type="error" title={'模型库读取失败：'+libraryError} action={<Button onClick={() => setRefreshKey(x => x+1)}>重试</Button>} />}
      {area === 'experience' && running && <Alert className="studio-busy" type="info" showIcon title="当前有任务运行中" action={<Button onClick={() => open(running.id)}>查看任务</Button>} />}
      {view === 'experience' && <div className="studio-page-enter"><ModelExperience key={modelId || 'default'} imageGroups={restricted ? catalog.groups.map(group=>group.id) : undefined} initialModel={modelId} initialTestset={testsetId} onSelectionChange={(model,testset) => setQuery({view,model,...(testset ? {testset} : {})})} library={library} disabled={!!error || !!libraryError || !!running || !!launch.intent} create={create} open={open} /></div>}
      {view === 'evaluate' && <div className="studio-page-enter"><EvaluationPanel key={modelId || 'default'} initialModel={modelId} initialTestset={testsetId} onSelectionChange={(model,testset) => setQuery({view,model,...(testset ? {testset} : {})})} library={library} disabled={!!error || !!libraryError || !!running || !!launch.intent} create={create} open={open} /></div>}
      {view === 'compare' && <div className="studio-page-enter"><ComparisonPanel jobs={scopedJobs} open={open} /></div>}
    </>}
  </StudioShell>;
}
