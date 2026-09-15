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
vi.mock('../platform/inference', () => ({ ModelExperience: () => <h2>单图体验工作区</h2>, PredictionPanel: () => null }));
vi.mock('../platform/evaluation', () => ({ ComparisonPanel: () => <h2>实验结果工作区</h2>,
  EvaluationPanel: () => <h2>独立评测工作区</h2>, EvaluationResults: () => null }));

beforeAll(() => {
  window.scrollTo = vi.fn();
  window.matchMedia = vi.fn().mockImplementation(query => ({ matches: false, media: query,
    addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn() }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
beforeEach(() => vi.clearAllMocks());
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const defaults: RequestConfig = { group: 'officehome_vit', method: 'fedavg', name: '正在训练的实验', rounds: 8,
  clientCount: 60, sampleClients: 0, batchSize: 32, localEpochs: 1, learningRate: .0001,
  seed: 42, splitSeed: 42, alpha: .1, gpu: 1, evaluationFrequency: 1, samplesPerClient: 0 };
const catalog: Catalog = { host: 'ct-2x4090-lziy', address: '10.112.81.135', protocol: 'test', evaluationPolicy: 'test', groups: [
  { id: defaults.group, dataset: 'Office-Home', backbone: 'vit', cacheFound: true, cacheFiles: 8,
    cacheBytes: 10, domains: 4, partitionLocked: false, methods: [{ id: 'fedavg', label: 'FedAvg', enabled: true, reason: null, defaults }] },
] };
const running: Job = { id: 'a'.repeat(32), action: 'train', status: 'running', stage: '联邦训练 第 3 轮',
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

describe('workspace navigation', () => {
  it('keeps four primary entries, legacy routes and exact job deep links', () => {
    expect(Object.keys(mainPages)).toEqual(['home', 'train', 'experience', 'compare']);
    expect(resolveView(null, '/')).toBe('home');
    for (const view of Object.keys(pages)) expect(resolveView(view, '/')).toBe(view);
    expect(resolveView(null, '/overview')).toBe('overview');
    expect(resolveView(null, '/scenario-analysis')).toBe('cache');
    expect(resolveView(null, '/experiments/new')).toBe('train');
    expect(resolveView(null, '/reports')).toBe('jobs');
    expect(resolveView('toString', '/')).toBe('home');
    expect(resolveView(null, '/unknown')).toBe('home');
    expect(viewHref('home')).toBe('/');
  });

  it('collapses the sidebar, closes the mobile drawer on navigation and preserves current task access', async () => {
    const openCurrent = vi.fn();
    render(<MemoryRouter><StudioShell view="home" connected running openCurrent={openCurrent}><div>页面内容</div></StudioShell><LocationProbe /></MemoryRouter>);
    const nav = screen.getByRole('navigation', { name: '主要功能' });
    expect(within(nav).getAllByRole('link')).toHaveLength(6);
    expect(within(nav).getByRole('link', { name: '首页' })).toHaveAttribute('aria-current', 'page');
    fireEvent.click(screen.getByRole('button', { name: /当前任务/ }));
    expect(openCurrent).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole('button', { name: '收起侧栏' }));
    expect(screen.getByRole('button', { name: '展开侧栏' }).closest('.studio-shell')).toHaveClass('studio-collapsed');
    const trigger = screen.getByRole('button', { name: '打开功能导航' });
    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    const drawer = await screen.findByRole('dialog');
    fireEvent.click(within(drawer).getByRole('link', { name: '训练实验' }));
    expect(screen.getByLabelText('测试当前路径')).toHaveTextContent('/?view=train');
    await waitFor(() => expect(trigger).toHaveAttribute('aria-expanded', 'false'));
  });

  it('routes from home to model validation and back without submitting a task', async () => {
    const fetch = mockApi();
    render(<MemoryRouter><PlatformApp /><LocationProbe /></MemoryRouter>);
    expect(screen.getByRole('heading', { level: 1, name: /开始一次新的探索/ })).toBeInTheDocument();
    await waitFor(() => expect(within(screen.getByLabelText('服务器资产概览')).getByRole('link', { name: /已保存模型/ })).toHaveTextContent('3'));
    fireEvent.click(screen.getByRole('link', { name: /单图预测/ }));
    expect(await screen.findByRole('heading', { name: '单图体验工作区' })).toBeInTheDocument();
    expect(screen.getByText('当前有任务运行中')).toBeInTheDocument();
    expect(screen.getByLabelText('测试当前路径')).toHaveTextContent('/?view=experience');
    expect(document.title).toBe('模型验证 · 全域智汇');
    fireEvent.click(screen.getByRole('button', { name: '测试返回上一页' }));
    fireEvent.click(screen.getByRole('link', { name: /已完成的基线实验/ }));
    expect(screen.getByLabelText('测试当前路径')).toHaveTextContent('/?view=jobs&id=' + completed.id);
    expect(fetch.mock.calls.every(([, init]) => init.method === 'GET')).toBe(true);
  }, 20000);

  it('preserves direct task links and ignores stray IDs outside the task view', async () => {
    const fetch = mockApi();
    const page = render(<MemoryRouter initialEntries={['/?view=jobs&id=' + running.id]}><PlatformApp /></MemoryRouter>);
    expect(screen.getByRole('heading', { level: 1, name: '训练实验' })).toBeInTheDocument();
    await screen.findByRole('button', { name: /停止任务/ });
    expect(fetch).toHaveBeenCalledWith('/api/platform/jobs/' + running.id, expect.anything());
    page.unmount();
    render(<MemoryRouter initialEntries={['/?view=missing&id=must-not-fetch']}><PlatformApp /></MemoryRouter>);
    expect(screen.getByRole('heading', { level: 1, name: /开始一次新的探索/ })).toBeInTheDocument();
    expect(fetch.mock.calls.some(([url]) => url.includes('must-not-fetch'))).toBe(false);
  });

  it('preserves edited training fields while switching training subpages', async () => {
    const fetch = mockApi();
    render(<MemoryRouter initialEntries={['/?view=train']}><PlatformApp /></MemoryRouter>);
    fireEvent.change(await screen.findByLabelText('通信轮数'), { target: { value: '15' } });
    fireEvent.change(screen.getByLabelText('实验名称'), { target: { value: '未提交的草稿' } });
    const tabs = screen.getByRole('navigation', { name: '模块功能' });
    fireEvent.click(within(tabs).getByRole('link', { name: '实验记录' }));
    expect(screen.queryByRole('spinbutton', { name: /通信轮数/ })).not.toBeInTheDocument();
    fireEvent.click(within(tabs).getByRole('link', { name: '新建实验' }));
    expect(screen.getByLabelText('通信轮数')).toHaveValue('15');
    expect(screen.getByLabelText('实验名称')).toHaveValue('未提交的草稿');
    expect(fetch.mock.calls.every(([, init]) => init.method === 'GET')).toBe(true);
  }, 20000);

  it('keeps navigation, retry and the separate map demo usable while offline', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('server offline')));
    render(<MemoryRouter><PlatformApp /></MemoryRouter>);
    await screen.findByText('server offline');
    expect(screen.getByRole('button', { name: /重连/ })).toBeInTheDocument();
    expect(screen.getByText('服务器暂时离线')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /单图预测/ })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '地图模拟演示' })).toHaveAttribute('href', '/demo');
  });

  it.each([['privacy', '隐私保护'], ['backdoor', '后门攻防']])('reserves %s without starting tasks, even offline', async (view, label) => {
    const fetch = vi.fn().mockRejectedValue(new Error('server offline'));
    vi.stubGlobal('fetch', fetch);
    render(<MemoryRouter initialEntries={['/?view=' + view + '&id=must-not-fetch']}><PlatformApp /></MemoryRouter>);
    expect(screen.getByRole('heading', { level: 1, name: label })).toBeInTheDocument();
    const module = screen.getByRole('region', { name: label + '规划说明' });
    expect(within(module).getByText('功能预留，尚未接入真实任务。')).toBeInTheDocument();
    expect(within(module).getAllByText('待接入')).toHaveLength(3);
    expect(within(module).queryByRole('button')).not.toBeInTheDocument();
    await screen.findByText('server offline');
    expect(fetch.mock.calls.every(([url, init]) => init.method === 'GET' && !url.includes('must-not-fetch'))).toBe(true);
    fireEvent.click(within(module).getByRole('link', { name: /返回首页/ }));
    expect(screen.getByRole('heading', { level: 1, name: /开始一次新的探索/ })).toBeInTheDocument();
  });
});

describe('homepage live data', () => {
  it('renders actual inventory, actual stages and precise job links', () => {
    render(<MemoryRouter><SystemHome catalog={catalog} jobs={[running, completed]} library={library} resource={resource} stale={false} overview={running} /></MemoryRouter>);
    const inventory = screen.getByLabelText('服务器资产概览');
    expect(within(inventory).getByRole('link', { name: /已保存模型/ })).toHaveTextContent('3');
    expect(within(inventory).getByRole('link', { name: /已完成训练/ })).toHaveTextContent('1');
    expect(screen.getByText(running.stage)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /查看详情/ })).toHaveAttribute('href', '/?view=jobs&id=' + running.id);
    expect(screen.getByRole('link', { name: /查看服务器资源/ })).toHaveTextContent('32%');
    for (const view of ['train', 'jobs', 'cache', 'experience', 'evaluate', 'compare', 'privacy', 'backdoor']) {
      expect(screen.getAllByRole('link').some(link => link.getAttribute('href') === '/?view=' + view)).toBe(true);
    }
  });

  it('distinguishes loading from empty without inventing metrics', () => {
    const page = render(<MemoryRouter><SystemHome jobs={[]} library={{ models: [], testsets: [] }} stale={false} /></MemoryRouter>);
    expect(within(screen.getByLabelText('服务器资产概览')).getAllByText('—')).toHaveLength(4);
    expect(screen.getAllByText('正在读取实验').length).toBeGreaterThan(0);
    page.rerender(<MemoryRouter><SystemHome catalog={catalog} jobs={[]} library={{ models: [], testsets: [] }} stale={false} /></MemoryRouter>);
    expect(screen.getByText('还没有训练记录')).toBeInTheDocument();
    expect(screen.queryByText(/GPU 1/)).not.toBeInTheDocument();
    expect(within(screen.getByLabelText('服务器资产概览')).getByRole('link', { name: /已保存模型/ })).toHaveTextContent('0');
  });

  it('marks a stale snapshot and hides stale resource percentages', () => {
    render(<MemoryRouter><SystemHome catalog={catalog} jobs={[running, completed]} library={library} resource={resource} stale overview={running} /></MemoryRouter>);
    expect(screen.getByText('离线快照')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /查看服务器资源/ })).toHaveTextContent('资源数据待刷新');
    expect(screen.queryByText('32%')).not.toBeInTheDocument();
  });
});
