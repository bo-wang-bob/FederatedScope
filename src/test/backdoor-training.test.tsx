import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { BackdoorTrainingPanel } from '../platform/backdoorTraining';
import { api } from '../platform/api';

vi.mock('../platform/api', async importOriginal => ({
  ...await importOriginal<typeof import('../platform/api')>(), api: vi.fn(),
}));
vi.mock('../platform/datasetUpload', () => ({ DatasetUpload: () => null }));
afterEach(() => { cleanup(); vi.clearAllMocks(); });

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
