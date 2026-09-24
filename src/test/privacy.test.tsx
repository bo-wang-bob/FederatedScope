import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeAll, expect, it, vi } from 'vitest';
import { PrivacyMembershipPanel } from '../platform/extensions';

// Historical replay remains available as a component/API, while the route now
// opens native privacy experiments. Test replay here without mocking that route.
function PlannedModule(_props: { moduleId: string }) { return <PrivacyMembershipPanel />; }

beforeAll(() => {
  window.matchMedia = vi.fn().mockImplementation(query => ({ matches: false, media: query,
    addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn() }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

// Explicit HTTP fixtures; these values are never shipped as production results.
const distribution = { bins: [.25, .75], member: [0, 1], nonmember: [1, 0],
  memberMean: .9, nonmemberMean: .1, meanGap: .8, memberSamples: 1, nonmemberSamples: 1 };
const payload = (clientId = 1, group = 'member') => ({
  configured: true, source: 'fixture', clientId, clients: [1, 2], group,
  metrics: { noDefense: { auc: .92, tprAt1Fpr: .63, fprAtThreshold: .01 },
    defense: { auc: .54, tprAt1Fpr: .02, fprAtThreshold: .01 } },
  distributions: { noDefense: distribution, defense: distribution },
  items: [{ id: `${clientId}-${group}`, truth: group, domain: 'Art', className: 'Radio',
    filename: 'radio.jpg', imageUrl: `/api/platform/privacy/membership/images/${clientId}/${group}/0`,
    noDefenseScore: .9, defenseScore: .4, noDefensePrediction: 'member', defensePrediction: 'nonmember' }],
});

it('shows missing resources without fabricated results or training writes', async () => {
  const fetch = vi.fn(async (_path: string, _init: RequestInit) => ({ ok: true, json: async () => ({ data: {
    configured: false, source: null, items: [], message: '隐私模块资源未就绪，训练与模型评测不受影响。',
    expectedPath: '/fixture/fedmia_local' } }) }));
  vi.stubGlobal('fetch', fetch);
  render(<PlannedModule moduleId="privacy" />);
  expect(await screen.findByText(/隐私模块资源未就绪/)).toBeInTheDocument();
  expect(screen.queryByText('AUC')).not.toBeInTheDocument();
  expect(screen.queryByText(/membership_examples.json/)).not.toBeInTheDocument();
  expect(screen.getByText('资源配置').closest('details')).not.toHaveAttribute('open');
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch.mock.calls[0][1]).toMatchObject({ method: 'GET' });
});

it('uses API metrics and paired predictions, switches groups and clients, reuses loaded results', async () => {
  const fetch = vi.fn(async (path: string) => {
    const query = new URL(path, 'http://test').searchParams;
    return { ok: true, json: async () => ({ data: payload(Number(query.get('clientId') || 1), query.get('group') || 'member') }) };
  });
  vi.stubGlobal('fetch', fetch);
  render(<PlannedModule moduleId="privacy" />);
  expect(await screen.findByText('92.00%')).toBeInTheDocument();
  expect(screen.getByText('54.00%')).toBeInTheDocument();
  const result = screen.getByRole('region', { name: '成员推理攻击结果' });
  expect(within(result).getByRole('img', { name: 'Radio' })).toHaveAttribute('src', '/api/platform/privacy/membership/images/1/member/0');
  expect(within(result).getByText('成员')).toBeInTheDocument();
  expect(within(result).getByText('非成员')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '非训练样本' }));
  await waitFor(() => expect(screen.getByRole('region', { name: '成员推理攻击结果' }).querySelector('img')).toHaveAttribute('src', '/api/platform/privacy/membership/images/1/nonmember/0'));
  fireEvent.click(screen.getByRole('button', { name: '客户端训练样本' }));
  await screen.findByText('92.00%');
  expect(fetch).toHaveBeenCalledTimes(2);
  fireEvent.mouseDown(screen.getByRole('combobox'));
  fireEvent.click(await screen.findByText('Client 2', { selector: '.ant-select-item-option-content' }));
  await waitFor(() => expect(screen.getByRole('region', { name: '成员推理攻击结果' }).querySelector('img')).toHaveAttribute('src', '/api/platform/privacy/membership/images/2/member/0'));
  expect(fetch).toHaveBeenCalledTimes(3);
});

it('renders API errors without fallback demo metrics', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 409,
    json: async () => ({ error: { message: '成员推理分数与样本数量不一致' } }) })));
  render(<PlannedModule moduleId="privacy" />);
  expect(await screen.findByText('成员推理结果读取失败')).toBeInTheDocument();
  expect(screen.getByText('成员推理分数与样本数量不一致')).toBeInTheDocument();
  expect(screen.queryByText('AUC')).not.toBeInTheDocument();
});

it('uses the selected dataset label without interpretation notices', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ data: {
    ...payload(), dataset: 'MilitaryAircraft3D',
    alignment: { memberSamples: 7, nonmemberSamples: 7, metricsMode: 'mix',
      message: '整体指标使用 mix 校准样本，逐图预测使用真实图片分数。' },
  } }) })));
  render(<PlannedModule moduleId="privacy" />);
  expect(await screen.findByText('MilitaryAircraft3D · 1 个')).toBeInTheDocument();
  expect(screen.queryByText('整体指标与当前样本预测')).not.toBeInTheDocument();
  expect(screen.queryByText('共同样本指标与当前样本预测')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: '查看结果口径' })).not.toBeInTheDocument();
});

it('keeps result controls without warning banners or interpretation popovers', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ data: {
    ...payload(), alignment: { memberSamples: 27, nonmemberSamples: 27,
      message: '共同样本：成员 27 个，非成员 27 个。不作为严格防御增益结论。' },
  } }) })));
  render(<PlannedModule moduleId="privacy" />);
  await screen.findByText('92.00%');
  expect(screen.queryByRole('button', { name: '查看结果口径' })).not.toBeInTheDocument();
  expect(screen.queryByText(/不作为严格防御增益结论/)).not.toBeInTheDocument();
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(screen.queryByText('共同样本指标与当前样本预测')).not.toBeInTheDocument();
  expect(screen.queryByText('整体指标与当前样本预测')).not.toBeInTheDocument();
  expect(screen.getByRole('region', { name: '攻击分数分布图表' })).toHaveAttribute('tabindex', '0');
  expect(screen.getByRole('region', { name: '攻防指标对照' })).toHaveAttribute('tabindex', '0');
});
