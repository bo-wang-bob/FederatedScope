import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, expect, it, vi } from 'vitest';
import { BackdoorLab } from '../platform/backdoor';
import { api } from '../platform/api';

vi.mock('../platform/api', async importOriginal => ({
  ...await importOriginal<typeof import('../platform/api')>(), api: vi.fn(),
}));
vi.mock('../platform/datasetUpload', () => ({ DatasetUpload: () => null }));
// antd 的下拉浮层/尺寸监听依赖这两个浏览器 API, jsdom 没有实现。
window.matchMedia = vi.fn().mockReturnValue({
  matches: false, addListener: vi.fn(), removeListener: vi.fn(),
  addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
});
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as never;
afterEach(() => { cleanup(); vi.clearAllMocks(); });

// 后门页拆成「训练实验 / 对比测试」两个标签: 训练区常驻挂载, 对比区放原来的挑图与三连对比。
const trainingStatus = {
  runnable: true, base: 'runs', missing: [], group: null, job: null,
  datasets: [{ id: 'd1', name: 'uploaded', classes: 4, count: 40, layout: 'classes' }],
  templates: [{ key: 'baseline', file: 'vit_newdataset.yaml', label: '干净基线（无攻击）', exists: true }],
  attacks: [
    { key: 'data_poisoning', name: '数据投毒', attack: 'a.yaml', defense: 'a_d.yaml' },
    { key: 'model_poisoning', name: '模型投毒', attack: 'b.yaml', defense: 'b_d.yaml' },
    { key: 'common_backdoor', name: '常见后门', attack: 'c.yaml', defense: 'c_d.yaml' },
    { key: 'new_backdoor', name: '新型后门', attack: 'd.yaml', defense: 'd_d.yaml' },
  ],
  attack: 'new_backdoor', attackName: '新型后门', defaultAttack: 'new_backdoor',
};
const testset = {
  exported: true, total: 1, classNames: ['Alarm_Clock', 'Laptop'], runs: { attack: 'x' },
  domains: [{ name: 'Art', count: 1 }], labels: [{ index: 1, name: 'Laptop', count: 1 }],
};

const mount = (entries = ['/']) => render(
  <MemoryRouter initialEntries={entries}><BackdoorLab /></MemoryRouter>);

const respond = (path: string) => {
  if (path === 'backdoor/training') return trainingStatus;
  if (path === 'backdoor/testset') return testset;
  if (path.startsWith('backdoor/jobs/')) {
    return { id: 'b'.repeat(32), action: 'backdoor', ids: [], name: '', base: 'runs',
      device: 'cpu', runs: {}, status: 'completed', stage: '已完成', error: null,
      createdAt: '', updatedAt: '' };
  }
  if (path === 'backdoor/jobs') return [];
  return { ids: ['Art_00001'], labels: [1] };
};

it('splits the page into 训练实验 / 对比测试 and starts on training', async () => {
  vi.mocked(api).mockImplementation(async (path: string) => respond(path) as never);
  mount();
  const train = await screen.findByRole('tab', { name: /训练实验/ });
  expect(train).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByRole('tab', { name: /对比测试/ })).toHaveAttribute('aria-selected', 'false');
  // 训练实验区含「实验配置」与「训练进度」两块
  expect(screen.getByText('实验配置')).toBeVisible();
  expect(screen.getByText('训练进度')).toBeVisible();
});

it('reveals the picker and comparison once 对比测试 is selected', async () => {
  vi.mocked(api).mockImplementation(async (path: string) => respond(path) as never);
  mount();
  fireEvent.click(await screen.findByRole('tab', { name: /对比测试/ }));
  expect(await screen.findByText('选择测试图片')).toBeVisible();
  expect(screen.getByRole('button', { name: '样本 Art_00001 · Laptop' })).toBeVisible();
});

it('lands on 对比测试 directly for a shared ?job= link', async () => {
  vi.mocked(api).mockImplementation(async (path: string) => respond(path) as never);
  mount(['/?job=' + 'b'.repeat(32)]);
  expect(await screen.findByRole('tab', { name: /对比测试/ })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByRole('tab', { name: /训练实验/ })).toHaveAttribute('aria-selected', 'false');
});
