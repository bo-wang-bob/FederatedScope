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
  fireEvent.click(screen.getByText('添加数据集'));
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
  fireEvent.click(screen.getByText('添加数据集'));
  const files = ['A', 'B'].flatMap(c => [1, 2, 3].map(i => file(`Train/${c}/${i}.png`)));
  fireEvent.change(screen.getByLabelText('选择训练数据集文件夹'), { target: { files } });
  fireEvent.click(screen.getByRole('button', { name: '上传并添加' }));
  await waitFor(() => expect(complete).toHaveBeenCalledWith(group));
  const uploads = fetcher.mock.calls.filter(([url]) => String(url).includes('/files?'));
  expect(uploads).toHaveLength(6);
  expect(uploads[0][0]).toContain('path=A%2F1.png');
  await waitFor(() => expect(screen.getByText('数据集已添加')).toBeVisible());
});
