import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom';
import { AppShell } from '../components/AppShell';
import PlatformApp from '../platform/PlatformApp';
import { SystemHome } from '../platform/home';
import { navigationItems, pages, resolveView, viewHref } from '../platform/navigation';
import type { Catalog, Job, Library, LibraryItem, RequestConfig, Resource } from '../platform/api';

vi.mock('../platform/charts', () => ({ Curves: () => null, Distribution: () => null, LossChart: () => null,
  ResourcesChart: () => null, Topology: () => null }));
vi.mock('../platform/inference', () => ({ ModelExperience: () => <h2>单图体验工作区</h2>, PredictionPanel: () => null }));
vi.mock('../platform/evaluation', () => ({ ComparisonPanel: () => <h2>实验结果工作区</h2>,
  EvaluationPanel: () => <h2>独立评测工作区</h2>, EvaluationResults: () => null }));

beforeAll(() => {
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
  it('defaults to home while preserving the old routes and every existing query view', () => {
    expect(resolveView(null, '/')).toBe('home');
    for (const view of Object.keys(pages)) expect(resolveView(view, '/')).toBe(view);
    expect(resolveView(null, '/overview')).toBe('overview');
    expect(resolveView(null, '/scenario-analysis')).toBe('cache');
    expect(resolveView(null, '/experiments/new')).toBe('train');
    expect(resolveView(null, '/reports')).toBe('jobs');
    expect(resolveView('invalid', '/')).toBe('home');
    expect(resolveView('toString', '/')).toBe('home');
    expect(resolveView(null, '/unknown')).toBe('home');
    expect(viewHref('home')).toBe('/');
  });

  it('groups the sidebar, closes the mobile drawer on selection and keeps the current task shortcut', async () => {
    const navigate = vi.fn(), openCurrent = vi.fn();
    render(<MemoryRouter><AppShell platform={{ menuItems: navigationItems, selectedKey: 'home', navigate,
      connected: true, openCurrent, pageLabel: '系统首页', sectionLabel: '工作台' }}><div>页面内容</div></AppShell></MemoryRouter>);
    for (const group of ['工作台', '模型验证', '训练实验', '模拟演示']) expect(screen.getByText(group)).toBeInTheDocument();
    expect(screen.queryByText('准确率实验 · 缓存只读')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /查看当前任务/ }));
    expect(openCurrent).toHaveBeenCalledOnce();
    const trigger = screen.getByRole('button', { name: '打开功能导航' });
    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    const drawer = await screen.findByRole('dialog');
    fireEvent.click(within(drawer).getByRole('menuitem', { name: /独立评测/ }));
    expect(navigate).toHaveBeenCalledWith('evaluate');
    await waitFor(() => expect(trigger).toHaveAttribute('aria-expanded', 'false'));
    fireEvent.click(screen.getByRole('button', { name: '全域智汇 · 返回系统首页' }));
    expect(navigate).toHaveBeenCalledWith('home');
  });

  it('opens home by default, routes to a feature and returns without submitting a task', async () => {
    const fetch = mockApi();
    render(<MemoryRouter><PlatformApp /><LocationProbe /></MemoryRouter>);
    expect(screen.getByRole('heading', { level: 1, name: '系统首页' })).toBeInTheDocument();
    await screen.findByText('2 个图像模型检查点匹配到测试集');
    fireEvent.click(screen.getByRole('link', { name: /进入模型体验台/ }));
    expect(await screen.findByRole('heading', { name: '单图体验工作区' })).toBeInTheDocument();
    expect(screen.getByText('平台正在执行其他任务，单图预测需等待当前任务结束。')).toBeInTheDocument();
    expect(screen.getByLabelText('测试当前路径')).toHaveTextContent('/?view=experience');
    expect(document.title).toBe('模型体验台 · 全域智汇');
    fireEvent.click(screen.getByRole('button', { name: '测试返回上一页' }));
    expect(screen.getByRole('heading', { level: 1, name: '系统首页' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('link', { name: /已完成的基线实验/ }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(`/api/platform/jobs/${completed.id}`, expect.anything()));
    expect(screen.getByLabelText('测试当前路径')).toHaveTextContent(`/?view=jobs&id=${completed.id}`);
    expect(fetch.mock.calls.every(([, init]) => init.method === 'GET')).toBe(true);
  }, 20000);

  it('preserves direct task links and falls back to a usable home for an unknown view', async () => {
    const fetch = mockApi();
    const page = render(<MemoryRouter initialEntries={[`/?view=jobs&id=${running.id}`]}><PlatformApp /></MemoryRouter>);
    expect(screen.getByRole('heading', { level: 1, name: '任务与记录' })).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(`/api/platform/jobs/${running.id}`, expect.anything()));
    page.unmount();
    render(<MemoryRouter initialEntries={['/?view=missing&id=must-not-fetch']}><PlatformApp /></MemoryRouter>);
    expect(screen.getByRole('heading', { level: 1, name: '系统首页' })).toBeInTheDocument();
    await screen.findByText('2 个图像模型检查点匹配到测试集');
    expect(fetch.mock.calls.some(([url]) => url.includes('must-not-fetch'))).toBe(false);
  });

  it('keeps homepage navigation available when the backend cannot be reached', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('server offline')));
    render(<MemoryRouter><PlatformApp /></MemoryRouter>);
    await screen.findByText('server offline');
    expect(screen.getByText('暂时无法读取任务')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /进入模型体验台/ })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /地图模拟演示/ })).toHaveAttribute('href', '/demo');
    expect(screen.queryByText('当前没有进行中的任务')).not.toBeInTheDocument();
  });
});

describe('homepage real data and feature links', () => {
  it('uses backend counts, compatible image models, real stages and exact task links', () => {
    render(<MemoryRouter><SystemHome catalog={catalog} jobs={[running, completed]} library={library} resource={resource} stale={false} /></MemoryRouter>);
    const inventory = screen.getByLabelText('服务器资产概览');
    expect(within(inventory).getByRole('link', { name: /已保存模型/ })).toHaveTextContent('3');
    expect(within(inventory).getByRole('link', { name: /已完成训练/ })).toHaveTextContent('1');
    expect(screen.getByText('2 个图像模型检查点匹配到测试集')).toBeInTheDocument();
    expect(screen.getByText(running.stage)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /查看任务与进度/ })).toHaveAttribute('href', `/?view=jobs&id=${running.id}`);
    expect(screen.getByRole('link', { name: /查看服务器资源/ })).toHaveTextContent('32%');
    for (const view of Object.keys(pages).filter(key => key !== 'home')) {
      expect(screen.getAllByRole('link').some(link => link.getAttribute('href') === `/?view=${view}`)).toBe(true);
    }
    expect(screen.getByRole('link', { name: /地图模拟演示/ })).toHaveAttribute('href', '/demo');
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('does not invent counts, activity or model readiness while loading or empty', () => {
    const page = render(<MemoryRouter><SystemHome jobs={[]} library={{ models: [], testsets: [] }} stale={false} /></MemoryRouter>);
    expect(within(screen.getByLabelText('服务器资产概览')).getAllByText('—')).toHaveLength(4);
    expect(screen.getByText('正在读取任务')).toBeInTheDocument();
    page.rerender(<MemoryRouter><SystemHome catalog={catalog} jobs={[]} library={{ models: [], testsets: [] }} stale={false} /></MemoryRouter>);
    expect(screen.getByText('当前没有进行中的任务')).toBeInTheDocument();
    expect(screen.getByText('尚无匹配测试集的图像模型')).toBeInTheDocument();
    expect(screen.queryByText(/GPU 1/)).not.toBeInTheDocument();
  });

  it('marks stale tasks and resources without presenting them as live', () => {
    render(<MemoryRouter><SystemHome catalog={catalog} jobs={[running, completed]} library={library} resource={resource} stale /></MemoryRouter>);
    expect(screen.getByText('状态待刷新')).toBeInTheDocument();
    expect(screen.getByText('连接异常，下方为上次读取的记录。')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /查看服务器资源/ })).toHaveTextContent('资源数据待刷新');
    expect(screen.queryByText('32%')).not.toBeInTheDocument();
  });
});
