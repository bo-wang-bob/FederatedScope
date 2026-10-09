import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { BackdoorTrainingPanel } from '../platform/backdoorTraining';
import { api } from '../platform/api';

vi.mock('../platform/api', async importOriginal => ({
  ...await importOriginal<typeof import('../platform/api')>(), api: vi.fn(),
}));
vi.mock('../platform/datasetUpload', () => ({ DatasetUpload: () => null }));
// antd 的下拉浮层依赖这两个浏览器 API, jsdom 没有实现。
window.matchMedia = vi.fn().mockImplementation(query => ({
  matches: false, media: query, addListener: vi.fn(), removeListener: vi.fn(),
  addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
}));
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as never;
afterEach(() => { cleanup(); vi.clearAllMocks(); });

const ATTACKS = [
  { key: 'data_poisoning', name: '数据投毒', attack: 'label_flip_stats.yaml', defense: 'label_flip_stats_defense.yaml' },
  { key: 'model_poisoning', name: '模型投毒', attack: 'label_flip_train.yaml', defense: 'label_flip_train_defense.yaml' },
  { key: 'common_backdoor', name: '常见后门', attack: 'a3fl.yaml', defense: 'a3fl_defense.yaml' },
  { key: 'new_backdoor', name: '新型后门', attack: 'sabre.yaml', defense: 'sabre_defense.yaml' },
];

const ready = (attack = 'new_backdoor') => ({
  datasets: [{ id: 'd1', name: 'uploaded', classes: 4, count: 40, layout: 'classes' }],
  templates: [{ key: 'baseline', file: 'vit_newdataset.yaml', label: '干净基线（无攻击）', exists: true }],
  missing: [], attacks: ATTACKS, attack, attackName: ATTACKS.find(a => a.key === attack)!.name,
  defaultAttack: 'new_backdoor', job: null, group: null, base: 'runs',
});

it.each(['cpu', 'cuda'])('keeps training progress without displaying device %s', async device => {
  vi.mocked(api).mockResolvedValue({ datasets: [], templates: [], missing: [],
    job: { id: 'a'.repeat(32), status: 'running', stage: '训练 2/3: SABRE 后门攻击',
      stageIndex: 1, datasetName: 'test', device },
  });
  render(<BackdoorTrainingPanel />);
  expect(await screen.findByText('训练 2/3: SABRE 后门攻击 · test')).toBeVisible();
  expect(screen.getByRole('button', { name: /停止训练/ })).toBeVisible();
  expect(screen.queryByText(/CPU|CUDA/)).not.toBeInTheDocument();
});

// antd v6: 选中值渲染在 .ant-select-content, 选项列表挂在 body 的浮层里。
const optionNodes = () => Array.from(document.querySelectorAll('.ant-select-item-option'));
const openAttackSelect = async () => {
  const input = await screen.findByLabelText('后门攻击类型');
  const root = input.closest('.ant-select')!;
  fireEvent.mouseDown(root);
  await waitFor(() => expect(optionNodes().length).toBe(ATTACKS.length));
  return root;
};

it('offers the four attacks with 新型后门 preselected', async () => {
  vi.mocked(api).mockResolvedValue(ready());
  render(<BackdoorTrainingPanel />);
  const root = await openAttackSelect();
  expect(root.querySelector('.ant-select-content')).toHaveTextContent('新型后门');
  expect(optionNodes().map(node => node.textContent))
    .toEqual(['数据投毒', '模型投毒', '常见后门', '新型后门']);
});

it('submits the selected attack key', async () => {
  const apiMock = vi.mocked(api);
  apiMock.mockResolvedValue(ready());
  render(<BackdoorTrainingPanel />);
  await openAttackSelect();
  fireEvent.click(optionNodes().find(node => node.textContent === '数据投毒')!);
  fireEvent.click(await screen.findByRole('button', { name: /启动训练/ }));
  await waitFor(() => expect(apiMock).toHaveBeenCalledWith('backdoor/training',
    expect.objectContaining({ datasetId: 'd1', attack: 'data_poisoning' })));
});

it('omits the attack field when the backend exposes no catalog', async () => {
  const apiMock = vi.mocked(api);
  const legacy = ready();
  delete (legacy as { attacks?: unknown }).attacks;
  apiMock.mockResolvedValue(legacy);
  render(<BackdoorTrainingPanel />);
  await screen.findByText('数据集');
  expect(screen.queryByLabelText('后门攻击类型')).not.toBeInTheDocument();
  fireEvent.click(await screen.findByRole('button', { name: /启动训练/ }));
  await waitFor(() => expect(apiMock).toHaveBeenCalledWith('backdoor/training', { datasetId: 'd1' }));
});

// 训练进度面板 (参考隐私页): 三个实验位 -> Steps + Progress + Stat。
const progressStatus = (status: string, stageIndex: number) => ({
  ...ready('model_poisoning'),
  templates: [
    { key: 'baseline', file: 'vit_newdataset.yaml', label: '干净基线（无攻击）', exists: true },
    { key: 'attack', file: 'label_flip_train.yaml', label: '模型投毒 后门攻击', exists: true },
    { key: 'defense', file: 'label_flip_train_defense.yaml', label: '模型投毒 攻击 + 防御', exists: true },
  ],
  job: { id: 'b'.repeat(32), status, stage: '训练 2/3: 模型投毒 后门攻击',
    stageIndex, total: 3, datasetName: 'uploaded', attackName: '模型投毒' },
});

it('shows the three experiments as steps with a progress bar and stats', async () => {
  vi.mocked(api).mockResolvedValue(progressStatus('running', 1));
  render(<BackdoorTrainingPanel />);
  expect(await screen.findByText('实验进度')).toBeVisible();
  expect(screen.getByText('1 / 3')).toBeVisible();
  expect(screen.getByRole('progressbar')).toBeInTheDocument();
  expect(screen.getByText('干净基线（无攻击）')).toBeVisible();
  // 未完成时没有跳转按钮, 且仍在运行 -> 可以停止
  expect(screen.queryByRole('button', { name: /前往对比测试/ })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: /停止训练/ })).toBeVisible();
});

it('offers to jump to the comparison tab once training finishes', async () => {
  const onOpenCompare = vi.fn();
  vi.mocked(api).mockResolvedValue(progressStatus('completed', 3));
  render(<BackdoorTrainingPanel onOpenCompare={onOpenCompare} />);
  expect(await screen.findByText('3 / 3')).toBeVisible();
  // 终态任务不再显示"停止训练"
  expect(screen.queryByRole('button', { name: /停止训练/ })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /前往对比测试/ }));
  expect(onOpenCompare).toHaveBeenCalledTimes(1);
});
