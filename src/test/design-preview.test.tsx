import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import DesignPreview from '../design/DesignPreview';
import { samples } from '../design/assets';
import { DRAFT_KEY } from '../platform/draft';

vi.setConfig({ testTimeout: 30000 });
const fetchGuard = vi.fn(() => { throw new Error('A design preview must never request the backend'); });
const xhrGuard = vi.fn();
beforeAll(() => {
  window.scrollTo = vi.fn();
  window.matchMedia = vi.fn().mockImplementation(query => ({ matches: false, media: query,
    addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn() }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
beforeEach(() => {
  localStorage.clear(); sessionStorage.clear(); fetchGuard.mockClear(); xhrGuard.mockClear();
  vi.stubGlobal('fetch', fetchGuard);
  vi.spyOn(XMLHttpRequest.prototype, 'open').mockImplementation(xhrGuard);
});
afterEach(() => {
  expect(fetchGuard).not.toHaveBeenCalled();
  expect(xhrGuard).not.toHaveBeenCalled();
  cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals();
});
const mount = (url = '/') => render(<MemoryRouter initialEntries={[url]}><DesignPreview /></MemoryRouter>);
const navigate = (label: string) => fireEvent.click(within(screen.getByRole('navigation', { name: '主要功能' })).getByRole('link', { name: label }));
async function selectOption(label: string, value: string) {
  fireEvent.mouseDown(screen.getByRole('combobox', { name: label }));
  fireEvent.click(await screen.findByText(value, { selector: '.ant-select-item-option-content' }));
}

describe('frontend design review — isolated from live execution', () => {
  it('keeps only three core homepage actions and locally bundled real imagery', () => {
    mount();
    const home = screen.getByRole('region', { name: '功能导航' });
    expect(within(home).getAllByRole('link')).toHaveLength(3);
    expect(screen.getByRole('status')).toHaveTextContent('设计预览');
    expect(screen.queryByText(/服务已连接|GPU|本地草稿|科研仿真实验平台/)).not.toBeInTheDocument();
    expect(home.querySelectorAll('p,small')).toHaveLength(0);
    expect(within(home).getAllByRole('img').filter(image => image.tagName === 'IMG')).toHaveLength(4);
    expect(samples).toHaveLength(4);
    expect(samples.every(sample => !sample.src.includes('/api/') && /^[a-f0-9]{64}$/.test(sample.sha256))).toBe(true);
  });

  it('omits the removed image-source entry and modal from the console', () => {
    mount();
    expect(screen.queryByRole('button', { name: '图片来源' })).not.toBeInTheDocument();
    expect(screen.queryByText(/Joshua Stevens/)).not.toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('设计预览');
  });

  it('validates training parameters, retains a preview draft and never overwrites a live draft', async () => {
    localStorage.setItem(DRAFT_KEY, 'live-draft-must-not-change');
    mount('/?view=train&source=must-not-fetch');
    fireEvent.change(screen.getByLabelText(/实验名称/), { target: { value: '前端配置验收' } });
    fireEvent.click(screen.getByRole('button', { name: /下一步/ }));
    fireEvent.change(screen.getByRole('spinbutton', { name: '通信轮数' }), { target: { value: '18' } });
    fireEvent.mouseDown(screen.getByRole('combobox', { name: '每轮参与客户端' }));
    expect(screen.queryByText('61', { selector: '.ant-select-item-option-content' })).not.toBeInTheDocument();
    fireEvent.click(await screen.findByTitle('3'));
    navigate('系统首页'); navigate('训练实验');
    expect(screen.getByLabelText(/实验名称/)).toHaveValue('前端配置验收');
    fireEvent.click(screen.getByRole('button', { name: /下一步/ }));
    expect(screen.getByRole('spinbutton', { name: '通信轮数' })).toHaveValue('18');
    fireEvent.click(screen.getByRole('button', { name: /下一步/ }));
    fireEvent.click(screen.getByRole('button', { name: /启动训练/ }));
    expect(await screen.findByText('当前为设计预览，未预检、未提交训练。')).toBeInTheDocument();
    expect(localStorage.getItem(DRAFT_KEY)).toBe('live-draft-must-not-change');
  });

  it('selects actual images, blocks unloaded/broken images, and never fabricates a prediction', async () => {
    mount('/?view=experience');
    const stage = screen.getByRole('region', { name: '当前样本' });
    expect(screen.getByRole('button', { name: 'scan 预测' })).toBeDisabled();
    fireEvent.load(within(stage).getByRole('img', { name: /Real_World · Radio/ }));
    fireEvent.click(screen.getByRole('button', { name: 'scan 预测' }));
    expect(await screen.findByText(/尚未加载模型或执行推理/)).toBeInTheDocument();
    expect(screen.queryByText(/置信度|正确率|\d+\.\d+%/)).not.toBeInTheDocument();
    fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: /确.*定/ }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: '下一张' }));
    const nextImage = within(stage).getByRole('img', { name: /Real_World · Laptop/ });
    expect(screen.getByRole('button', { name: 'scan 预测' })).toBeDisabled();
    fireEvent.error(nextImage);
    expect(within(stage).getByText('图像无法载入')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'scan 预测' })).toBeDisabled();
  });

  it('filters domains/classes together and handles an empty selection', async () => {
    mount('/?view=experience');
    await selectOption('样本域', 'Art');
    expect(screen.getAllByRole('button', { name: /^选择 / })).toHaveLength(1);
    await selectOption('样本类别', 'Laptop');
    expect(screen.getByText('没有匹配的样本')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'scan 预测' })).toBeDisabled();
  });

  it('supports evaluation selection without sending a task or inventing metrics', async () => {
    mount('/?view=evaluate');
    expect(screen.getByText('尚无评测结果')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Art' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Real World' }));
    expect(screen.getByRole('button', { name: /开始评测/ })).toBeDisabled();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Art' }));
    fireEvent.click(screen.getByRole('button', { name: /开始评测/ }));
    expect(await screen.findByText(/未提交任务或生成指标/)).toBeInTheDocument();
  });

  it('preserves comparison and research extensions without the removed map', async () => {
    mount('/?view=compare');
    expect(screen.getByText('尚无可对比结果')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '导出结果' })).toBeDisabled();
    expect(screen.queryByRole('link', { name: /地图仿真/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('link', { name: /隐私保护/ }));
    expect(screen.getByRole('heading', { level: 1, name: '隐私保护' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: '成员推理攻击结果' })).toBeInTheDocument();
    expect(screen.getByText('无防御')).toBeInTheDocument();
    expect(screen.getByText('有防御')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('link', { name: /后门防御/ }));
    expect(screen.getByRole('heading', { level: 1, name: '后门防御' })).toBeInTheDocument();
    expect(screen.getAllByText('未接入')).toHaveLength(1);
  });
});
