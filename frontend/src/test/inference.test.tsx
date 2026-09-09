import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { ModelExperience } from '../platform/inference';
import type { Job, Library, SamplePage } from '../platform/api';

beforeAll(() => {
  window.matchMedia = vi.fn().mockImplementation(query => ({ matches: false, media: query,
    addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn() }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const model = { id: 'a'.repeat(32) + ':final', jobId: 'a'.repeat(32), name: '测试模型', group: 'officehome_vit',
  method: 'fedavg', kind: 'final', classes: ['Class_A', 'Class_B'], domains: [{ name: 'Art', testSamples: 2 }],
  featureSpace: 'fixture', sha256: 'model-version', trainingRounds: 2 };
const library: Library = { models: [model], testsets: [{ ...model, id: model.jobId, samples: 2 }] };
const page: SamplePage = { total: 2, offset: 0, limit: 12, testFingerprint: 'test-version', provenance: {},
  items: [0, 1].map(i => ({ id: String(i).repeat(24), index: i, filename: `${i}.jpg`, domain: 'Art',
    label: i, className: model.classes[i], imageAvailable: true, imageSha256: String(i).repeat(64), imageUrl: `/fixture/${i}.jpg` })) };
const prediction = { id: 'b'.repeat(32), action: 'predict', status: 'completed', request: {}, result: {
  sampleId: page.items[1].id, predictedClass: 0, predictedName: 'Class_A', label: 1, labelName: 'Class_B',
  correct: false, confidence: .8, topK: [{ classIndex: 0, className: 'Class_A', score: .8 }, { classIndex: 1, className: 'Class_B', score: .2 }],
  inferenceMs: 2, elapsedSeconds: 1, checkpointSha256: 'model-version', imageSha256: page.items[1].imageSha256,
} } as unknown as Job;

describe('single-image model experience', () => {
  it('sends the selected real sample and version, renders actual wrong predictions, and clears stale results', async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ data: page }) });
    vi.stubGlobal('fetch', fetch);
    const create = vi.fn().mockResolvedValue(prediction);
    render(<ModelExperience library={library} disabled={false} create={create} open={vi.fn()} />);
    fireEvent.click(await screen.findByRole('button', { name: '选择样本 1.jpg' }));
    expect(screen.getByAltText('测试原图 1.jpg')).toHaveAttribute('src', '/fixture/1.jpg');
    fireEvent.click(screen.getByRole('button', { name: /运行单图预测/ }));
    await waitFor(() => expect(create).toHaveBeenCalledWith('predict', expect.objectContaining({
      modelId: model.id, testsetId: model.jobId, sampleId: page.items[1].id, imageSha256: page.items[1].imageSha256,
    })));
    expect(await screen.findByText('预测不一致')).toBeInTheDocument();
    expect(screen.getByText('实际推理结果')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '选择样本 0.jpg' }));
    expect(screen.queryByText('预测不一致')).not.toBeInTheDocument();
    expect(create).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls.every(([path]) => String(path).includes('/samples?'))).toBe(true);
  }, 30000);
  it('does not allow inference when no original image is available', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ data: { ...page,
      items: page.items.map(item => ({ ...item, imageAvailable: false, imageSha256: undefined, imageError: '缺少原图' })) } }) }));
    const create = vi.fn();
    render(<ModelExperience library={library} disabled={false} create={create} open={vi.fn()} />);
    await screen.findByRole('button', { name: '选择样本 0.jpg' });
    expect(screen.getByRole('button', { name: /运行单图预测/ })).toBeDisabled();
    expect(create).not.toHaveBeenCalled();
  }, 30000);
});
