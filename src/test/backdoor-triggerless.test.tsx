import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, expect, it, vi } from 'vitest';
import { BackdoorCompare } from '../platform/backdoorCompare';
import { api, type BackdoorJob } from '../platform/api';

vi.mock('../platform/api', async importOriginal => ({
  ...await importOriginal<typeof import('../platform/api')>(), api: vi.fn(),
}));
window.matchMedia = vi.fn().mockReturnValue({ matches: false, addListener: vi.fn(), removeListener: vi.fn() });
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as never;
afterEach(() => { cleanup(); vi.clearAllMocks(); });

const id = 'a'.repeat(32);

// label_flip (数据投毒 / 模型投毒) 的后端结果: triggerless, 没有 triggered 那一格。
const triggerlessJob = {
  id, action: 'backdoor', ids: ['uploaded_00001', 'uploaded_00002'], name: '',
  base: 'runs', device: 'cpu', runs: {}, status: 'completed', stage: '已完成',
  error: null, createdAt: '', updatedAt: '',
  images: { clean: '/clean.png', defenseClean: '/defense_clean.png' },
  result: {
    ids: ['uploaded_00001', 'uploaded_00002'], classNames: ['A', 'B'],
    targetLabel: 0, targetName: 'A', attackName: 'label_flip', triggerless: true,
    runs: {},
    images: [
      { id: 'uploaded_00001', label: 1, labelName: 'B',
        clean: { label: 0, name: 'A' }, defense: { label: 1, name: 'B', hit: false } },
      { id: 'uploaded_00002', label: 0, labelName: 'A',
        clean: { label: 0, name: 'A' }, defense: { label: 0, name: 'A', hit: false } },
    ],
    stats: {}, paths: { clean: 'clean.png', defenseClean: 'defense_clean.png' },
  },
} as unknown as BackdoorJob;

const mount = () => render(
  <MemoryRouter initialEntries={[`/?job=${id}`]}><BackdoorCompare /></MemoryRouter>);

it('renders a triggerless comparison without crashing on the missing trigger column', async () => {
  vi.mocked(api).mockImplementation(async (path: string) =>
    (path === 'backdoor/jobs' ? [triggerlessJob] : triggerlessJob) as never);
  mount();
  expect(await screen.findByText('label_flip 属于无触发器攻击')).toBeVisible();
  expect(screen.getByText('注入触发器（不适用）')).toBeVisible();
  // 两行都没有触发器预测 -> 该列显示占位符
  expect(screen.getAllByText('—')).toHaveLength(2);
});

it('still labels the trigger column normally for a trigger-based attack', async () => {
  const job = structuredClone(triggerlessJob) as BackdoorJob;
  job.result!.triggerless = false;
  job.result!.images = job.result!.images.map(row => ({
    ...row, triggered: { label: 0, name: 'A', hit: row.label !== 0 },
  }));
  vi.mocked(api).mockImplementation(async (path: string) =>
    (path === 'backdoor/jobs' ? [job] : job) as never);
  mount();
  expect(await screen.findByText('逐样本预测对照')).toBeVisible();
  expect(screen.getByText('注入触发器')).toBeVisible();
  expect(screen.queryByText('label_flip 属于无触发器攻击')).not.toBeInTheDocument();
  // 第一行真实标签 B 被判成目标类 A -> 劫持
  expect(screen.getByText('劫持')).toBeVisible();
});
