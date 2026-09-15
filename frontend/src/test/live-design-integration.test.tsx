import { cleanup, configure, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import PlatformApp from '../platform/PlatformApp';
import type { Catalog, Job, Library, RequestConfig, SamplePage } from '../platform/api';

vi.setConfig({ testTimeout: 30000 });
configure({ asyncUtilTimeout: 10000 });
vi.mock('../platform/charts', () => ({ CompareCurves: () => null, Confusion: () => null, DomainBars: () => null,
  Curves: () => null, Distribution: () => null, LossChart: () => null, Topology: () => null }));
beforeAll(() => {
  window.scrollTo = vi.fn();
  window.matchMedia = vi.fn().mockImplementation(query => ({ matches: false, media: query,
    addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn() }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
beforeEach(() => { localStorage.clear(); sessionStorage.clear(); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const request: RequestConfig = { group: 'officehome_vit', method: 'fedavg', name: '', rounds: 1, clientCount: 4,
  sampleClients: 4, batchSize: 8, localEpochs: 1, learningRate: .0001, seed: 42, splitSeed: 42,
  alpha: .1, gpu: 0, evaluationFrequency: 1, samplesPerClient: 4 };
const catalog: Catalog = { host: 'fixture', address: '', protocol: '', evaluationPolicy: '', groups: [{
  id: request.group, dataset: 'Office-Home', backbone: 'vit', domains: 4, cacheFound: true, cacheFiles: 4,
  cacheBytes: 0, partitionLocked: false, methods: [{ id: 'fedavg', label: 'FedAvg', enabled: true, reason: null, defaults: request }],
}] };
const model = { id: 'a'.repeat(32)+':final', jobId: 'a'.repeat(32), name: '后端模型', group: request.group,
  method: 'fedavg', kind: 'final', featureSpace: 'fixture', sha256: 'saved-checkpoint', classes: ['Radio', 'Computer'], domains: [{ name: 'Art', testSamples: 1 }] };
const library: Library = { models: [model], testsets: [{ ...model, id: model.jobId, samples: 1 }] };
const page: SamplePage = { total: 1, offset: 0, limit: 12, testFingerprint: 'saved-testset', provenance: { Art: 'legacy' },
  items: [{ id: 'sample-from-backend', index: 0, domain: 'Art', label: 0, className: 'Radio', filename: 'real.jpg', imageUrl: '/fixture/real.jpg', imageAvailable: true, imageSha256: 'image-version' }] };
const prediction = { id: 'b'.repeat(32), action: 'predict', status: 'completed', request, clients: {}, metrics: [], config: {}, provenance: {}, cleanup: { ok: true, message: '' }, error: null,
  createdAt: '2026-09-15T08:00:00Z', updatedAt: '2026-09-15T08:00:01Z', stage: '完成', result: {
    sampleId: page.items[0].id, predictedClass: 1, predictedName: 'Computer', label: 0, labelName: 'Radio', correct: false, confidence: .6,
    topK: [{ classIndex: 1, className: 'Computer', score: .6 }, { classIndex: 0, className: 'Radio', score: .4 }], inferenceMs: 2, elapsedSeconds: 1,
    checkpointSha256: model.sha256, testBundleSha256: 'saved-testset', imageSha256: 'image-version', manifestSha256: 'manifest', testProvenance: 'legacy', inferenceContract: 'frozen-features', domain: 'Art', filename: 'real.jpg',
  } } as Job;

it('connects the approved live screen to catalog, samples and prediction, not preview fixtures', async () => {
  const writes: { path: string; body: Record<string, unknown> }[] = [];
  vi.stubGlobal('fetch', vi.fn(async (path: string, init: RequestInit) => {
    if (init.method === 'POST') { writes.push({ path, body: JSON.parse(init.body as string) }); return { ok: true, json: async () => ({ data: prediction }) }; }
    const data = path.endsWith('/catalog') ? catalog : path.endsWith('/library') ? library : path.includes('/samples?') ? page : [];
    return { ok: true, json: async () => ({ data }) };
  }));
  render(<MemoryRouter initialEntries={['/?view=experience']}><PlatformApp /></MemoryRouter>);
  const image = await screen.findByAltText('测试原图 real.jpg');
  expect(screen.queryByText('设计预览')).not.toBeInTheDocument();
  expect(screen.getByText('服务已连接')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: '运行单图预测' })).toBeDisabled();
  fireEvent.load(image);
  fireEvent.click(screen.getByRole('button', { name: '运行单图预测' }));
  await screen.findByText('预测不一致');
  expect(writes).toHaveLength(1);
  expect(writes[0].path).toBe('/api/platform/predict');
  expect(writes[0].body).toEqual(expect.objectContaining({ modelId: model.id, testsetId: model.jobId, sampleId: page.items[0].id, imageSha256: 'image-version', idempotencyKey: expect.any(String) }));
});

it('uses the live preflight/train controller from the redesigned training screen', async () => {
  const writes: { path: string; body: Record<string, unknown> }[] = [];
  const running = { ...prediction, id: 'c'.repeat(32), action: 'train', result: undefined, status: 'running', stage: '联邦训练', request } as Job;
  vi.stubGlobal('fetch', vi.fn(async (path: string, init: RequestInit) => {
    if (init.method === 'POST') {
      writes.push({ path, body: JSON.parse(init.body as string) });
      return { ok: true, json: async () => ({ data: path.endsWith('/preflight') ? { ...running, id: 'd'.repeat(32), action: 'inspect', status: 'completed' } : running }) };
    }
    const data = path.endsWith('/catalog') ? catalog : path.endsWith('/library') ? library : path.includes('/jobs/') ? running : [];
    return { ok: true, json: async () => ({ data }) };
  }));
  render(<MemoryRouter initialEntries={['/?view=train']}><PlatformApp /></MemoryRouter>);
  await screen.findByLabelText(/实验名称/);
  fireEvent.click(screen.getByRole('button', { name: /下一步/ }));
  fireEvent.change(screen.getByRole('spinbutton', { name: '通信轮数' }), { target: { value: '7' } });
  fireEvent.click(screen.getByRole('button', { name: /下一步/ }));
  fireEvent.click(screen.getByRole('button', { name: /启动训练/ }));
  await waitFor(() => expect(writes).toHaveLength(2));
  expect(writes.map(w => w.path)).toEqual(['/api/platform/preflight', '/api/platform/train']);
  expect(writes[0].body.rounds).toBe(7);
  expect(writes[1].body.rounds).toBe(7);
  expect(writes[1].body.preflightId).toBe('d'.repeat(32));
  expect(writes[0].body.idempotencyKey).not.toBe(writes[1].body.idempotencyKey);
  expect(await screen.findByRole('button', { name: /停止任务/ })).toBeInTheDocument();
});
