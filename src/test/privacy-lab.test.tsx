import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { PrivacyLab } from '../platform/privacyLab';

const defaults = { group: 'military_cnn', method: 'ggeur', name: '隐私测试', rounds: 200, clientCount: 9, defense: false };
const catalog = { groups: [{ id: 'military_cnn', dataset: 'MilitaryAircraft3D', backbone: 'ConvNeXt-Base', datasetFound: false, domains: 3,
  methods: [{ id: 'ggeur', label: 'GGEUR + FedMIA', enabled: true, defaults, defenseDefaults: { ...defaults, clientCount: 12, defense: true } }] }] };
const response = (data: unknown) => Promise.resolve({ ok: true, json: async () => ({ data }) } as Response);
beforeEach(() => {
  window.matchMedia = vi.fn().mockImplementation(query => ({ matches: false, media: query,
    addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn() }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it('shows training status without accuracy charts or backbone names', async () => {
  const job = { id: 'c'.repeat(32), request: defaults, action: 'train', status: 'running',
    stage: '模型训练', clients: {}, metrics: [{ round: 40, accuracy: .9876 }],
    featureStorage: { files: 9, rounds: [40], directory: 'features', resultsReady: false } };
  vi.spyOn(globalThis, 'fetch').mockImplementation(url => response(
    String(url).endsWith('/catalog') ? catalog : String(url).endsWith('/jobs') ? [job] : job));
  render(<PrivacyLab />);
  expect(await screen.findByText('模型训练中')).toBeVisible();
  expect(screen.getByText('40 / 200')).toBeVisible();
  expect(screen.queryByText('当前准确率')).not.toBeInTheDocument();
  expect(screen.queryByText('98.76%')).not.toBeInTheDocument();
  expect(screen.queryByRole('tab', { name: '训练曲线' })).not.toBeInTheDocument();
  expect(screen.queryByRole('tab', { name: '客户端状态' })).not.toBeInTheDocument();
  expect(screen.queryByRole('tab', { name: '执行日志' })).not.toBeInTheDocument();
  expect(screen.queryByText(/CNN|ConvNeXt/i)).not.toBeInTheDocument();
  expect(screen.getAllByText('MilitaryAircraft3D').length).toBeGreaterThan(0);
}, 15000);

it('shows only FedMIA-II metrics, distribution and sample predictions', async () => {
  const job = { id: 'b'.repeat(32), request: { ...defaults, name: 'II 展示测试' }, action: 'train',
    status: 'completed', stage: '完成', metrics: [], clients: {},
    featureStorage: { files: 6, rounds: [199], directory: 'features', resultsReady: true } };
  const metric = { auc: .92, tprAt1Fpr: .75, actualFpr: .01, threshold: .65, members: 1, nonmembers: 1 };
  const result = { dataset: 'military_cnn', defense: false, clientId: 1, clientIds: [1, 2],
    thresholdPolicy: 'mix 校准，经验 FPR≤1%；真实图片使用 test 分数', rounds: [199], indexedWarning: 'indexed 截断至影子客户端最短有效长度',
    metrics: { fedmia_i: { ...metric, threshold: .1234 }, fedmia_ii: metric },
    samples: {
      member: [{ domain: 'aerial', className: 'C-17', scores: { fedmia_i: .1, fedmia_ii: .83 },
        predictions: { fedmia_i: 'nonmember', fedmia_ii: 'member' } }],
      nonmember: [{ domain: 'aerial', className: 'F-16', scores: { fedmia_i: .9, fedmia_ii: .2 },
        predictions: { fedmia_i: 'member', fedmia_ii: 'nonmember' } }],
    } };
  vi.spyOn(globalThis, 'fetch').mockImplementation(url => {
    const path = String(url);
    return response(path.endsWith('/catalog') ? catalog : path.includes('/results?') ? result : path.endsWith('/jobs') ? [job] : job);
  });
  render(<PrivacyLab />);
  await screen.findByText('尚未找到数据集');
  fireEvent.click(screen.getByRole('tab', { name: /结果展示/ }));
  fireEvent.click(await screen.findByRole('button', { name: 'II 展示测试' }));
  expect(await screen.findByText('FedMIA-II', {}, { timeout: 5000 })).toBeVisible();
  expect(screen.queryByText('FedMIA-I', { exact: true })).not.toBeInTheDocument();
  expect(screen.queryByRole('combobox', { name: '攻击变体' })).not.toBeInTheDocument();
  expect(screen.getByRole('combobox', { name: '攻击评测客户端' })).toBeVisible();
  expect(screen.getByText('0.6500')).toBeVisible();
  expect(screen.queryByText('0.1234')).not.toBeInTheDocument();
  const samples = within(screen.getByRole('region', { name: '训练样本攻击结果' }));
  expect(samples.getByText('攻击分数：0.8300')).toBeVisible();
  expect(samples.getByText('成员', { exact: true })).toBeVisible();
  expect(screen.getByText(/均值差 0.6300/)).toBeVisible();
  expect(screen.queryByText('military_cnn')).not.toBeInTheDocument();
  expect(screen.queryByText(/mix 校准|indexed 截断|使用 1 个保存轮次|展示分布与 Mix|越大越容易/)).not.toBeInTheDocument();
}, 15000);

it.each(['实验名称', 'MilitaryAircraft3D', '无防御', '45 个文件', '已完成', 'Enter', ' '])('opens experiment from row target %s', async target => {
  const job = { id: 'c'.repeat(32), request: { ...defaults, name: '实验名称' }, action: 'train',
    status: 'completed', stage: '完成', metrics: [], clients: {}, featureStorage: { files: 45, rounds: [1], resultsReady: false } };
  const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation(url => {
    const path = String(url);
    return response(path.endsWith('/catalog') ? catalog : path.endsWith('/jobs') ? [job] : job);
  });
  render(<PrivacyLab />);
  await screen.findByText('尚未找到数据集');
  fireEvent.click(screen.getByRole('tab', { name: /结果展示/ }));
  const row = await screen.findByRole('row', { name: '查看实验 实验名称' });
  if (target === 'Enter' || target === ' ') fireEvent.keyDown(row, { key: target });
  else fireEvent.click(within(row).getByText(target));
  await waitFor(() => expect(row).toHaveAttribute('aria-selected', 'true'));
  expect(fetcher.mock.calls.filter(([url]) => String(url).endsWith('/jobs/' + job.id))).toHaveLength(1);
}, 15000);

it('opens local experiment results directly without the historical replay entry', async () => {
  const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation(url => response(String(url).endsWith('catalog') ? catalog : []));
  render(<PrivacyLab />);
  await screen.findByText('尚未找到数据集');
  fireEvent.click(screen.getByRole('tab', { name: /结果展示/ }));
  expect(screen.getByText('实验结果')).toBeVisible();
  expect(screen.queryByText('已有攻击结果')).not.toBeInTheDocument();
  expect(fetcher.mock.calls.some(([url]) => String(url).includes('/privacy/membership'))).toBe(false);
});

it('loads only private catalog and jobs and shows genuine missing resource warning', async () => {
  const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation(url => response(String(url).endsWith('catalog') ? catalog : []));
  render(<PrivacyLab />);
  expect(await screen.findByText('尚未找到数据集')).toBeVisible();
  expect(screen.queryByText(/云服务器|fedmia_local|已验证的实验配置|无需下载/)).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: /开始训练与攻击/ })).toBeDisabled();
  expect(fetcher.mock.calls.every(([url]) => String(url).startsWith('/api/platform/privacy/experiments/'))).toBe(true);
  expect(screen.queryByText('已有攻击结果')).not.toBeInTheDocument();
});

it('launches a private preflight with attack and training settings, not ordinary training', async () => {
  const task = { id: 'a'.repeat(32), request: defaults, action: 'inspect', status: 'queued',
    stage: '等待启动', clients: {}, metrics: [], events: [], cleanup: { ok: true } };
  const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation((url, options) => {
    if (options?.method === 'POST' || String(url).endsWith('/jobs/' + task.id)) return response(task);
    return response(String(url).endsWith('catalog') ? catalog : []);
  });
  render(<PrivacyLab />);
  await screen.findByText('尚未找到数据集');
  fireEvent.click(screen.getByRole('button', { name: /预检资源与划分/ }));
  await waitFor(() => expect(fetcher.mock.calls.some(([url, init]) => String(url).endsWith('/preflight') && init?.method === 'POST')).toBe(true));
  const call = fetcher.mock.calls.find(([, init]) => init?.method === 'POST')!;
  expect(call[0]).toBe('/api/platform/privacy/experiments/preflight');
  const submitted = JSON.parse(call[1]!.body as string);
  expect(submitted).toMatchObject({ group: 'military_cnn', method: 'ggeur', defense: false, rounds: 200, clientCount: 9 });
  expect(submitted.idempotencyKey).toBeTruthy();
  expect(Object.keys(submitted).sort()).toEqual(['clientCount', 'defense', 'group', 'idempotencyKey', 'method', 'name', 'rounds']);
});
