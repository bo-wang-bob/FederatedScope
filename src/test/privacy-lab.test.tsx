import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
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

it('opens local experiment results directly without the historical replay entry', async () => {
  const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation(url => response(String(url).endsWith('catalog') ? catalog : []));
  render(<PrivacyLab />);
  await screen.findByText('尚未找到本地数据集');
  fireEvent.click(screen.getByRole('tab', { name: /结果展示/ }));
  expect(screen.getByText('本地特征与攻击结果')).toBeVisible();
  expect(screen.queryByText('已有攻击结果')).not.toBeInTheDocument();
  expect(fetcher.mock.calls.some(([url]) => String(url).includes('/privacy/membership'))).toBe(false);
});

it('loads only private catalog and jobs and shows genuine missing resource warning', async () => {
  const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation(url => response(String(url).endsWith('catalog') ? catalog : []));
  render(<PrivacyLab />);
  expect(await screen.findByText('尚未找到本地数据集')).toBeVisible();
  expect(screen.getByRole('button', { name: /开始训练与攻击/ })).toBeDisabled();
  expect(fetcher.mock.calls.every(([url]) => String(url).startsWith('/api/platform/privacy/experiments/'))).toBe(true);
  expect(screen.queryByText('已有攻击结果')).not.toBeInTheDocument();
});

it('launches a private preflight with attack and training settings, not ordinary training', async () => {
  const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation((url, options) => {
    if (options?.method === 'POST') return response({ id: 'a'.repeat(32), request: defaults, action: 'inspect', status: 'queued',
      stage: '等待启动', clients: {}, metrics: [], events: [], cleanup: { ok: true } });
    return response(String(url).endsWith('catalog') ? catalog : []);
  });
  render(<PrivacyLab />);
  await screen.findByText('尚未找到本地数据集');
  fireEvent.click(screen.getByRole('button', { name: /预检资源与划分/ }));
  await waitFor(() => expect(fetcher.mock.calls.some(([url, init]) => String(url).endsWith('/preflight') && init?.method === 'POST')).toBe(true));
  const call = fetcher.mock.calls.find(([, init]) => init?.method === 'POST')!;
  expect(call[0]).toBe('/api/platform/privacy/experiments/preflight');
  const submitted = JSON.parse(call[1]!.body as string);
  expect(submitted).toMatchObject({ group: 'military_cnn', method: 'ggeur', defense: false, rounds: 200, clientCount: 9 });
  expect(submitted.idempotencyKey).toBeTruthy();
  expect(Object.keys(submitted).sort()).toEqual(['clientCount', 'defense', 'group', 'idempotencyKey', 'method', 'name', 'rounds']);
});
