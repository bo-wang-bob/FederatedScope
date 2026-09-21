import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom';
import { StudioShell } from '../platform/StudioShell';
import PlatformApp from '../platform/PlatformApp';
import { SystemHome } from '../platform/home';
import { mainPages, pages, resolveView, viewHref } from '../platform/navigation';
import type { Catalog, Job, Library, LibraryItem, RequestConfig, Resource } from '../platform/api';

// Ant Design accessibility queries are expensive under Windows/JSDOM.
vi.setConfig({ testTimeout: 30000 });

vi.mock('../platform/charts', () => ({ Curves: () => null, Distribution: () => null, LossChart: () => null,
  ResourcesChart: () => null, Topology: () => null }));
vi.mock('../platform/inference', () => ({ ModelExperience: () => <h2>单图验证工作区</h2>, PredictionPanel: () => null }));
vi.mock('../platform/evaluation', () => ({ ComparisonPanel: () => <h2>实验结果工作区</h2>,
  EvaluationPanel: () => <h2>独立评测工作区</h2>, EvaluationResults: () => null }));

beforeAll(() => {
  window.scrollTo = vi.fn();
  window.matchMedia = vi.fn().mockImplementation(query => ({ matches: false, media: query,
    addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn() }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
beforeEach(() => { vi.clearAllMocks(); localStorage.clear(); sessionStorage.clear(); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const defaults: RequestConfig = { group: 'military_vit', method: 'fedavg', name: '正在训练的实验', rounds: 8,
  clientCount: 60, sampleClients: 0, batchSize: 32, localEpochs: 1, learningRate: .0001,
  seed: 42, splitSeed: 42, alpha: .1, gpu: 1, evaluationFrequency: 1, samplesPerClient: 0 };
const catalog: Catalog = { host: 'ct-2x4090-lziy', address: '10.112.81.135', protocol: 'test', evaluationPolicy: 'test', groups: [
  { id: defaults.group, dataset: 'MilitaryAircraft-3D', backbone: 'vit', cacheFound: true, cacheFiles: 8,
    cacheBytes: 10, domains: 4, partitionLocked: false, methods: [{ id: 'fedavg', label: 'FedAvg', enabled: true, reason: null, defaults }] },
] };
const running: Job = { id: 'a'.repeat(32), action: 'train', status: 'running', stage: '协同训练 第 3 轮',
  request: defaults, createdAt: '2026-09-10T04:00:00Z', updatedAt: '2026-09-10T04:03:00Z',
  error: null, cleanup: { ok: true, message: '' }, clients: {}, metrics: [], config: {}, provenance: {} };
const completed: Job = { ...running, id: 'b'.repeat(32), status: 'completed', stage: '训练完成',
  request: { ...defaults, name: '已完成的基线实验' }, createdAt: '2026-09-09T04:00:00Z' };
const model: LibraryItem = { id: `${completed.id}:final`, jobId: completed.id, name: completed.request.name,
  group: defaults.group, method: 'fedavg', kind: 'final', domains: [], classes: ['class-a'], featureSpace: 'vit-768', sha256: 'model-hash' };
const library: Library = { models: [model, { ...model, id: `${completed.id}:best`, kind: 'best' },
  { ...model, id: 'text-model', group: 'mdsent_lstm', featureSpace: 'lstm-128' }],
  testsets: [{ ...model, id: completed.id, samples: 40 }] };
const resource: Resource = { at: running.updatedAt, hostname: catalog.host, cpuPercent: 16, memoryPercent: 25,
  memoryUsed: 1024, memoryTotal: 4096, diskFree: 5000, diskTotal: 10000, gpuError: null,
  gpus: [{ index: 1, name: 'RTX 4090', utilization: 32, memoryUsedMiB: 2048, memoryTotalMiB: 24576, temperature: 45 }] };

function LocationProbe() {
  const location = useLocation();
  const navigate = useNavigate();
  return <><output aria-label="测试当前路径">{location.pathname}{location.search}</output><button onClick={() => navigate(-1)}>测试返回上一页</button></>;
}

function mockApi() {
  const fetch = vi.fn().mockImplementation(async (input: string) => {
    const data = ({ '/api/platform/catalog': catalog, '/api/platform/jobs': [running, completed],
      '/api/platform/library': library, '/api/platform/resources': resource,
      [`/api/platform/jobs/${running.id}`]: running, [`/api/platform/jobs/${completed.id}`]: completed,
    } as Record<string, unknown>)[String(input)];
    if (data === undefined) throw new Error(`Unexpected API request: ${input}`);
    return { ok: true, json: async () => ({ data }) };
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

describe('desktop workspace navigation',()=>{
  it('keeps four primary areas, redirects retired operations pages and preserves job URLs',()=>{
    expect(Object.keys(mainPages)).toEqual(['home','train','experience','compare']);
    for(const view of Object.keys(pages))expect(resolveView(view,'/')).toBe(view);
    expect(resolveView('cache','/')).toBe('train');expect(resolveView('overview','/')).toBe('home');
    expect(resolveView(null,'/overview')).toBe('home');expect(resolveView(null,'/scenario-analysis')).toBe('train');
    expect(resolveView(null,'/experiments/new')).toBe('train');expect(resolveView(null,'/reports')).toBe('jobs');
    expect(resolveView('toString','/')).toBe('home');expect(viewHref('home')).toBe('/');
  });
  it('provides a fixed sidebar, current task access and reserved extensions',async()=>{
    const openCurrent=vi.fn();
    render(<MemoryRouter><StudioShell view="home" connected running openCurrent={openCurrent}>页面内容</StudioShell></MemoryRouter>);
    const nav=screen.getByRole('navigation',{name:'主要功能'});
    expect(within(nav).getAllByRole('link')).toHaveLength(4);
    expect(within(nav).getByRole('link',{name:'系统首页'})).toHaveAttribute('aria-current','page');
    expect(screen.getByRole('complementary')).toBeInTheDocument();
    expect(screen.getByText('服务已连接')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:/当前任务/}));expect(openCurrent).toHaveBeenCalledOnce();
    expect(await screen.findByRole('link',{name:/隐私保护/})).toHaveAttribute('href','/?view=privacy');
    expect(screen.getByRole('link',{name:/后门防御/})).toHaveAttribute('href','/?view=backdoor');
    expect(screen.queryByRole('link',{name:/地图仿真/})).not.toBeInTheDocument();
  });
  it('navigates home → model verification → experiment without any writes or resource requests',async()=>{
    const fetch=mockApi();render(<MemoryRouter><PlatformApp/><LocationProbe/></MemoryRouter>);
    expect(screen.getByRole('heading',{level:1,name:'跨域协同训练'})).toBeInTheDocument();
    await screen.findByRole('link',{name:/当前任务/});
    fireEvent.click(within(screen.getByRole('region',{name:'功能导航'})).getByRole('link',{name:'模型验证'}));
    expect(await screen.findByRole('heading',{name:'单图验证工作区'})).toBeInTheDocument();
    expect(screen.getByText('当前有任务运行中')).toBeInTheDocument();
    expect(screen.getByLabelText('测试当前路径')).toHaveTextContent('/?view=experience');
    fireEvent.click(screen.getByRole('button',{name:'测试返回上一页'}));
    fireEvent.click(within(screen.getByRole('navigation',{name:'主要功能'})).getByRole('link',{name:'训练实验'}));
    fireEvent.click(screen.getByRole('link',{name:'实验记录'}));
    fireEvent.click(screen.getByRole('button',{name:/^已完成的基线实验/}));
    expect(screen.getByLabelText('测试当前路径')).toHaveTextContent('/?view=jobs&id='+completed.id);
    expect(fetch.mock.calls.every(([url,init])=>init.method==='GET'&&!url.includes('/resources'))).toBe(true);
    expect(screen.queryByText(/GPU|数据缓存|运行资源/)).not.toBeInTheDocument();
  });
  it('preserves direct experiment links and ignores IDs on unknown pages',async()=>{
    const fetch=mockApi();const page=render(<MemoryRouter initialEntries={['/?view=jobs&id='+running.id]}><PlatformApp/></MemoryRouter>);
    expect(screen.getByRole('heading',{level:1,name:'实验详情'})).toBeInTheDocument();
    await screen.findByRole('button',{name:/停止任务/});
    expect(fetch).toHaveBeenCalledWith('/api/platform/jobs/'+running.id,expect.anything());
    page.unmount();render(<MemoryRouter initialEntries={['/?view=missing&id=must-not-fetch']}><PlatformApp/></MemoryRouter>);
    expect(screen.getByRole('heading',{level:1,name:'跨域协同训练'})).toBeInTheDocument();
    expect(fetch.mock.calls.some(([url])=>url.includes('must-not-fetch'))).toBe(false);
  });
  it('keeps drafts across module navigation, not only the training tabs',async()=>{
    const fetch=mockApi();render(<MemoryRouter initialEntries={['/?view=train']}><PlatformApp/></MemoryRouter>);
    fireEvent.change(await screen.findByLabelText(/实验名称/),{target:{value:'未提交草稿'}});
    fireEvent.click(screen.getByRole('button',{name:/下一步/}));
    fireEvent.change(screen.getByRole('spinbutton',{name:'通信轮数'}),{target:{value:'15'}});
    fireEvent.click(within(screen.getByRole('navigation',{name:'主要功能'})).getByRole('link',{name:'系统首页'}));
    fireEvent.click(within(screen.getByRole('navigation',{name:'主要功能'})).getByRole('link',{name:'训练实验'}));
    expect(screen.getByLabelText(/实验名称/)).toHaveValue('未提交草稿');
    fireEvent.click(screen.getByRole('button',{name:/下一步/}));
    expect(screen.getByRole('spinbutton',{name:'通信轮数'})).toHaveValue('15');
    expect(fetch.mock.calls.every(([,init])=>init.method==='GET')).toBe(true);
  });
  it('keeps navigation and reconnect available when the backend is offline',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('server offline')));
    render(<MemoryRouter><PlatformApp/></MemoryRouter>);
    await screen.findByText('server offline');
    expect(screen.getByRole('button',{name:/重连/})).toBeInTheDocument();
    expect(within(screen.getByRole('region',{name:'功能导航'})).getByRole('link',{name:'模型验证'})).toBeInTheDocument();
    expect(within(screen.getByRole('navigation',{name:'研究扩展'})).getByRole('link',{name:/隐私保护/})).toHaveAttribute('href','/?view=privacy');
    expect(screen.queryByText(/GPU|资源占用|缓存配置/)).not.toBeInTheDocument();
  });
  it('keeps data-cache legacy bookmarks useful without exposing a cache page',async()=>{
    const fetch=mockApi();render(<MemoryRouter initialEntries={['/?view=cache']}><PlatformApp/></MemoryRouter>);
    await screen.findByLabelText(/实验名称/);
    expect(screen.getByRole('heading',{name:'数据与算法'})).toBeInTheDocument();
    expect(screen.queryByRole('link',{name:'数据缓存'})).not.toBeInTheDocument();
    expect(fetch.mock.calls.every(([,init])=>init.method==='GET')).toBe(true);
  });
  it('reports unavailable backdoor API without launching a task',async()=>{
    const fetch=vi.fn().mockRejectedValue(new Error('server offline'));vi.stubGlobal('fetch',fetch);
    render(<MemoryRouter initialEntries={['/?view=backdoor&id=must-not-fetch']}><PlatformApp/></MemoryRouter>);
    await screen.findByText('后门防御接口不可用');
    expect(screen.queryByText('三连对比')).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'生成对比'})).not.toBeInTheDocument();
    expect(fetch.mock.calls.every(([url,init])=>init.method==='GET'&&!url.includes('must-not-fetch'))).toBe(true);
  });
  it('shows a membership loading failure without launching or fetching a task', async()=>{
    const fetch=vi.fn().mockRejectedValue(new Error('server offline'));vi.stubGlobal('fetch',fetch);
    render(<MemoryRouter initialEntries={['/?view=privacy&id=must-not-fetch']}><PlatformApp/></MemoryRouter>);
    expect((await screen.findAllByText('server offline')).length).toBeGreaterThan(0);
    expect(screen.queryByLabelText('演示范围')).not.toBeInTheDocument();
    expect(fetch.mock.calls.some(([url])=>url.includes('privacy/experiments/catalog'))).toBe(true);
    expect(fetch.mock.calls.every(([url,init])=>init.method==='GET'&&!url.includes('must-not-fetch'))).toBe(true);
  });
});
describe('action-first homepage',()=>{
  it('shows the actual current task and three approved image-led navigation entries',()=>{
    render(<MemoryRouter><SystemHome jobs={[running,completed]} loading={false}/></MemoryRouter>);
    expect(screen.getByRole('link',{name:/当前任务/})).toHaveAttribute('href','/?view=jobs&id='+running.id);
    expect(screen.queryByRole('link',{name:/已完成的基线实验/})).not.toBeInTheDocument();
    expect(within(screen.getByRole('region',{name:'功能导航'})).getAllByRole('link')).toHaveLength(3);
    expect(screen.getByRole('link',{name:/新建训练/})).toHaveAttribute('href','/?view=train');
    expect(screen.queryByText(/GPU|缓存|4090|服务器|资源/)).not.toBeInTheDocument();
  });
  it('keeps core navigation usable while loading without made-up metrics or inventory',()=>{
    const page=render(<MemoryRouter><SystemHome jobs={[]} loading/></MemoryRouter>);
    expect(screen.getByRole('link',{name:/新建训练/})).toBeInTheDocument();
    page.rerender(<MemoryRouter><SystemHome jobs={[]} loading={false}/></MemoryRouter>);
    expect(screen.getByRole('link',{name:/新建训练/})).toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });
});
