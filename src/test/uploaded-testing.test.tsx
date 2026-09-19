import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { UploadedResults, UploadedTesting } from '../platform/uploadedTesting';
import type { Job, Library } from '../platform/api';

beforeEach(() => {
  window.matchMedia = vi.fn().mockImplementation(media => ({ matches: false, media, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {}, dispatchEvent() {} }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });
const model = { id: 'a:final', jobId: 'a', name: '上传模型', group: 'uploaded_' + 'a'.repeat(32), method: 'fedavg', kind: 'final', classes: ['飞机', '舰船'], domains: [], featureSpace: 'uploaded-cnn', sha256: 'model' };
const library: Library = { models: [model], testsets: [] };
const result = { samples: 1, labelled: false, metrics: null, checkpointSha256: 'model', items: [{ index: 0, filename: 'photo.png', predictedName: '飞机', labelName: null, confidence: .75, correct: null, imageUrl: '/api/image' }] };
const completed = { id: 'job1', action: 'test-upload', status: 'completed', result } as Job;

it('renders unlabelled predictions without fabricated accuracy or ground truth', () => {
  render(<UploadedResults job={completed} />);
  expect(screen.getByText('飞机')).toBeVisible();
  expect(screen.getByText('75.00%')).toBeVisible();
  expect(screen.queryByText(/准确率/)).not.toBeInTheDocument();
  expect(screen.queryByText('真实类别')).not.toBeInTheDocument();
  expect(screen.getByRole('link', { name: /导出结果/ })).toHaveAttribute('href', '/api/platform/jobs/job1/export');
});

it('uploads a single image, selects it, and sends the saved dataset and model to the backend', async () => {
  const dataset = { id: 'b'.repeat(32), name: 'photo.png', kind: 'test', count: 1, classes: [] };
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, options) => ({ ok: true, json: async () => ({ data: options?.method === 'POST' ? dataset : [] }) }) as Response);
  const create = vi.fn().mockResolvedValue(completed);
  render(<UploadedTesting library={library} disabled={false} create={create} open={vi.fn()} />);
  fireEvent.click(screen.getByRole('button', { name: '上传单张图片' }));
  fireEvent.change(screen.getByLabelText('选择测试图片'), { target: { files: [new File(['image'], 'photo.png')] } });
  fireEvent.click(screen.getByRole('button', { name: '上传并添加' }));
  await screen.findByText('上传完成');
  fireEvent.click(await screen.findByRole('button', { name: '完成' }));
  await waitFor(() => expect(screen.getByRole('button', { name: /开始测试/ })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: /开始测试/ }));
  await waitFor(() => expect(create).toHaveBeenCalledWith('test-upload', expect.objectContaining({ modelId: model.id, uploadId: dataset.id })));
  expect(await screen.findByText('飞机')).toBeVisible();
});

it('keeps errors visible and does not offer incompatible cached models for raw-image testing', async () => {
  vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('连接失败'));
  render(<UploadedTesting library={{ models: [{ ...model, group: 'military_vit' }], testsets: [] }} disabled={false} create={vi.fn()} open={vi.fn()} />);
  expect(await screen.findByText('连接失败')).toBeVisible();
  expect(screen.getByRole('button', { name: /开始测试/ })).toBeDisabled();
});
