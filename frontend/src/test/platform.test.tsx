import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { TrainingForm } from '../platform/PlatformApp';
import type { Catalog, Job, RequestConfig } from '../platform/api';

vi.mock('../components/ChartPanel', () => ({
  Panel: ({ children, title }: { children: React.ReactNode; title: string }) => <section><h2>{title}</h2>{children}</section>,
  Chart: () => <div aria-label="数据图表" />, chartText: '#abc',
}));
beforeAll(() => {
  window.matchMedia = vi.fn().mockImplementation(query => ({ matches: false, media: query,
    addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn() }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
afterEach(cleanup);

const defaults: RequestConfig = { group: 'officehome_vit', method: 'fedavg', name: '', rounds: 3,
  clientCount: 60, sampleClients: 0, batchSize: 32, localEpochs: 1, learningRate: .0001,
  seed: 42, splitSeed: 42, alpha: .1, gpu: 1, evaluationFrequency: 1, samplesPerClient: 0 };
const catalog: Catalog = { host: 'test', address: 'test', protocol: 'test', evaluationPolicy: 'test',
  groups: [{ id: defaults.group, dataset: 'Office-Home', backbone: 'vit', cacheFound: true,
    cacheFiles: 8, cacheBytes: 10, domains: 4, partitionLocked: false,
    methods: [{ id: 'fedavg', label: 'FedAvg', enabled: true, reason: null, defaults }] }] };
const preflight = { id: 'a'.repeat(32), action: 'inspect', status: 'completed', stage: '预检通过',
  clients: {}, request: defaults, error: null, result: { clientCount: 60, trainSamples: 100,
    testSamples: 40, testFingerprint: 'test-version', classes: ['0'], domains: [], testProvenance: {} } } as unknown as Job;

describe('platform parameter workflow', () => {
  it('exposes generation without an augmented cache and submits its parameters', async () => {
    const own = { ...defaults, method: 'heterogeneous_solution', augmentationMode: 'generate' as const,
      generatedPerSample: 50, generatedPerPrototype: 50, targetPerClass: 50, covarianceScale: 1,
      augmentationSourceId: '', allowLegacyAugmentation: false };
    const ownCatalog = { ...catalog, groups: [{ ...catalog.groups[0], methods: [
      { id: own.method, label: '本架构', enabled: true, reason: null, augmentedCacheFound: false, defaults: own }] }] };
    const create = vi.fn().mockResolvedValue(preflight);
    render(<TrainingForm catalog={ownCatalog} disconnected={false} create={create} open={vi.fn()} />);
    await screen.findByLabelText('每个原始样本生成数');
    fireEvent.change(screen.getByLabelText('每个原始样本生成数'), { target: { value: '2' } });
    fireEvent.change(screen.getByLabelText('每客户端每类目标样本数'), { target: { value: '20' } });
    fireEvent.click(screen.getByRole('button', { name: /检查完整缓存/ }));
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1));
    expect(create.mock.calls[0][1]).toMatchObject({ method: 'heterogeneous_solution',
      augmentationMode: 'generate', generatedPerSample: 2, targetPerClass: 20, allowLegacyAugmentation: false });
  }, 30000);

  it('submits the visible parameters and binds training to the successful preflight', async () => {
    const create = vi.fn().mockResolvedValueOnce(preflight).mockResolvedValueOnce({ ...preflight, id: 'b'.repeat(32), action: 'train' });
    const open = vi.fn();
    render(<TrainingForm catalog={catalog} disconnected={false} create={create} open={open} />);
    expect(screen.getByRole('button', { name: /启动训练/ })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('通信轮数'), { target: { value: '8' } });
    fireEvent.change(screen.getByLabelText('学习率'), { target: { value: '0.002' } });
    fireEvent.change(screen.getByLabelText('Batch size'), { target: { value: '17' } });
    fireEvent.click(screen.getByRole('button', { name: /检查完整缓存/ }));
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1));
    expect(create.mock.calls[0][0]).toBe('preflight');
    expect(create.mock.calls[0][1]).toMatchObject({ rounds: 8, learningRate: .002, batchSize: 17, gpu: 1, clientCount: 60 });
    await waitFor(() => expect(screen.getByRole('button', { name: /启动训练/ })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: /启动训练/ }));
    await waitFor(() => expect(create).toHaveBeenCalledTimes(2));
    expect(create.mock.calls[1]).toEqual(['train', expect.objectContaining({ rounds: 8, learningRate: .002, preflightId: preflight.id })]);
    expect(open).toHaveBeenCalledWith('b'.repeat(32));
  }, 30000);

  it('invalidates a preflight after an input changes', async () => {
    const create = vi.fn().mockResolvedValue(preflight);
    render(<TrainingForm catalog={catalog} disconnected={false} create={create} open={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /检查完整缓存/ }));
    await waitFor(() => expect(screen.getByRole('button', { name: /启动训练/ })).toBeEnabled());
    fireEvent.change(screen.getByLabelText('Batch size'), { target: { value: '64' } });
    expect(screen.getByRole('button', { name: /启动训练/ })).toBeDisabled();
  }, 30000);
});
