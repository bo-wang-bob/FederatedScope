import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ComparisonPanel } from '../platform/evaluation';
import type { Job } from '../platform/api';

vi.mock('../platform/charts', () => ({ CompareCurves: ({ jobs }: { jobs: Job[] }) => <div data-testid="curves">{jobs.length} curves</div>, Confusion: () => null, DomainBars: () => null }));
vi.mock('antd', async importOriginal => {
  const actual = await importOriginal<typeof import('antd')>();
  return { ...actual, Select: ({ options, value, onChange, mode, ...props }: any) =>
    <select aria-label={props['aria-label']} multiple={mode === 'multiple'} value={value} onChange={e => onChange(mode === 'multiple' ? [...e.target.selectedOptions].map(o => o.value) : e.target.value)}>
      {options.map((o: any) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select> };
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it('keeps curves and truthful export metadata without comparison notices', async () => {
  window.matchMedia = vi.fn().mockReturnValue({ matches: false, addListener: vi.fn(), removeListener: vi.fn() });
  const jobs = ['a', 'b'].map((id, i) => ({ id, action: 'train', status: 'completed',
    request: { name: id, group: 'uploaded_test', method: i ? 'fedavg' : 'heterogeneous_solution', alpha: i ? .1 : .01 },
    data: { testFingerprint: 'same-test', partitionFingerprint: id, augmentation: { warning: '历史增强来源提醒' } },
    metrics: [{ round: 1, accuracy: i ? .6 : .9 }],
  })) as unknown as Job[];
  vi.stubGlobal('fetch', vi.fn(async (url: string) => ({ ok: true, json: async () => ({ data: jobs.find(j => url.endsWith('/' + j.id)) }) })));
  vi.stubGlobal('Blob', class { parts: string[]; constructor(parts: string[]) { this.parts = parts; } });
  const createURL = vi.fn().mockReturnValue('blob:comparison');
  vi.stubGlobal('URL', class extends URL { static createObjectURL = createURL; static revokeObjectURL = vi.fn(); });
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  render(<ComparisonPanel jobs={jobs} open={vi.fn()} />);
  const select = screen.getByLabelText('选择对比实验') as HTMLSelectElement;
  for (const option of select.options) option.selected = true;
  fireEvent.change(select);
  await waitFor(() => expect(screen.getByTestId('curves')).toHaveTextContent('2 curves'));
  expect(screen.queryByText(/对比条件|不能直接比较|历史增强|仅展示观察值/)).not.toBeInTheDocument();
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /导出对比/ }));
  const exported = JSON.parse(createURL.mock.calls[0][0].parts.join(''));
  expect(exported.comparable).toBe(false);
  expect(exported.jobs).toEqual(jobs);
}, 30000);
