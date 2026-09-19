import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { DatasetUpload } from '../platform/datasetUpload';

beforeEach(() => {
  window.matchMedia = vi.fn().mockImplementation(media => ({ matches: false, media,
    addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {}, dispatchEvent() {} }));
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

function file(path: string) {
  const value = new File(['image'], path.split('/').at(-1)!, { type: 'image/png' });
  Object.defineProperty(value, 'webkitRelativePath', { value: path });
  return value;
}

it('requires class folders rather than loose training images', async () => {
  render(<DatasetUpload />);
  fireEvent.click(screen.getByRole('button', { name: '上传训练集' }));
  fireEvent.change(screen.getByLabelText('选择训练数据集文件夹'), { target: { files: [file('Train/a.png')] } });
  await waitFor(() => expect(screen.getByText(/请选择训练集最外层文件夹/)).toBeVisible());
  expect(screen.getByRole('button', { name: '上传并添加' })).toBeDisabled();
});

it('uploads class-relative files and publishes the completed dataset', async () => {
  const id = 'a'.repeat(32), group = 'uploaded_' + id;
  const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation(async url => ({ ok: true,
    json: async () => ({ data: String(url).endsWith('/finish') ? { id, group, count: 6, classes: ['A', 'B'] } : { id } }),
  }) as Response);
  const complete = vi.fn();
  render(<DatasetUpload onUploaded={complete} />);
  fireEvent.click(screen.getByRole('button', { name: '上传训练集' }));
  const files = ['A', 'B'].flatMap(c => [1, 2, 3].map(i => file(`Train/${c}/${i}.png`)));
  fireEvent.change(screen.getByLabelText('选择训练数据集文件夹'), { target: { files } });
  fireEvent.click(screen.getByRole('button', { name: '上传并添加' }));
  await waitFor(() => expect(complete).toHaveBeenCalledWith(group));
  const uploads = fetcher.mock.calls.filter(([url]) => String(url).includes('/files?'));
  expect(uploads).toHaveLength(6);
  expect(uploads[0][0]).toContain('path=A%2F1.png');
  await waitFor(() => expect(screen.getByText('上传完成')).toBeVisible());
  expect(screen.queryByText('本地保存，独立缓存')).not.toBeInTheDocument();
  expect(screen.queryByText(/自动识别训练集/)).not.toBeInTheDocument();
});

it.each([false, true])('uploads flat test images independently (single=%s)', async single => {
  const saved = { id: 'c'.repeat(32), kind: 'test', name: '测试', count: single ? 1 : 2, classes: [] };
  const fetcher = vi.spyOn(globalThis, 'fetch').mockResolvedValue({ ok: true, json: async () => ({ data: saved }) } as Response);
  const complete = vi.fn();
  render(<DatasetUpload kind="test" single={single} onSaved={complete} />);
  fireEvent.click(screen.getByRole('button', { name: single ? '上传单张图片' : '上传测试集' }));
  const input = screen.getByLabelText(single ? '选择测试图片' : '选择测试数据集文件夹');
  expect(input.hasAttribute('webkitdirectory')).toBe(!single);
  const files = single ? [new File(['x'], 'a.png')] : [file('测试/a.png'), file('测试/b.png')];
  fireEvent.change(input, { target: { files } });
  fireEvent.click(screen.getByRole('button', { name: '上传并添加' }));
  await waitFor(() => expect(complete).toHaveBeenCalledWith(saved));
  expect(JSON.parse(fetcher.mock.calls[0][1]!.body as string).kind).toBe('test');
  const uploads = fetcher.mock.calls.filter(([url]) => String(url).includes('/files?'));
  expect(uploads).toHaveLength(files.length);
  expect(uploads[0][0]).toContain('path=a.png');
  expect(uploads[0][1]?.signal).toBeDefined();
});

it('rejects mixing unlabelled and labelled test images', async () => {
  render(<DatasetUpload kind="test" />);
  fireEvent.click(screen.getByRole('button', { name: '上传测试集' }));
  fireEvent.change(screen.getByLabelText('选择测试数据集文件夹'), { target: { files: [file('Test/a.png'), file('Test/A/b.png')] } });
  await waitFor(() => expect(screen.getByText(/不能混合/)).toBeVisible());
  expect(screen.getByRole('button', { name: '上传并添加' })).toBeDisabled();
});

it('retries a lost finish response without duplicating the dataset or files', async () => {
  let finishes = 0;
  const saved = { id: 'd'.repeat(32), kind: 'test', count: 1, classes: [] };
  const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation(async url => {
    if (String(url).endsWith('/finish') && finishes++ === 0) throw new Error('连接中断');
    return { ok: true, json: async () => ({ data: saved }) } as Response;
  });
  render(<DatasetUpload kind="test" single />);
  fireEvent.click(screen.getByRole('button', { name: '上传单张图片' }));
  fireEvent.change(screen.getByLabelText('选择测试图片'), { target: { files: [new File(['x'], 'a.png')] } });
  fireEvent.click(screen.getByRole('button', { name: '上传并添加' }));
  await screen.findByText('连接中断');
  fireEvent.click(screen.getByRole('button', { name: '上传并添加' }));
  await screen.findByText('上传完成');
  expect(fetcher.mock.calls.filter(([url]) => String(url).endsWith('/datasets'))).toHaveLength(1);
  expect(fetcher.mock.calls.filter(([url]) => String(url).includes('/files?'))).toHaveLength(1);
});
