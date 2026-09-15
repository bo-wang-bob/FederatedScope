import type { ScenarioPartitionPreview, ScenarioPreviewRequest, ScenarioRecord } from '../types';
import { generateScenarioPartition } from '../utils/partition';

export interface ScenarioDataAdapter {
  readonly source: 'backend' | 'frontend_simulation';
  preview(request: ScenarioPreviewRequest, signal?: AbortSignal): Promise<ScenarioPartitionPreview>;
  create(request: ScenarioPreviewRequest, signal?: AbortSignal): Promise<ScenarioRecord>;
}

export const demoScenarioAdapter: ScenarioDataAdapter = {
  source: 'frontend_simulation',
  async preview(request, signal) {
    if (signal?.aborted) throw new DOMException('请求已取消', 'AbortError');
    return generateScenarioPartition(request.partition.alpha, request.partition.seed);
  },
  async create(request, signal) {
    if (signal?.aborted) throw new DOMException('请求已取消', 'AbortError');
    const preview = generateScenarioPartition(request.partition.alpha, request.partition.seed);
    return {
      scenarioId: `LOCAL-${preview.partitionVersion}`,
      createdAt: new Date().toISOString(),
      request,
      preview,
    };
  },
};

export function createBackendScenarioAdapter(apiBaseUrl: string): ScenarioDataAdapter {
  return {
    source: 'backend',
    async preview(request, signal) {
      const response = await fetch(`${apiBaseUrl.replace(/\/$/, '')}/api/scenarios/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal,
      });
      if (!response.ok) throw new Error(`场景预览失败：HTTP ${response.status}`);
      const payload = await response.json() as { data: ScenarioPartitionPreview };
      return { ...payload.data, source: 'backend' };
    },
    async create(request, signal) {
      const response = await fetch(`${apiBaseUrl.replace(/\/$/, '')}/api/scenarios`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal,
      });
      const payload = await response.json() as {
        data?: ScenarioRecord;
        error?: { message?: string };
      };
      if (!response.ok || !payload.data) {
        throw new Error(payload.error?.message || `场景应用失败：HTTP ${response.status}`);
      }
      return { ...payload.data, preview: { ...payload.data.preview, source: 'backend' } };
    },
  };
}

export function resolveScenarioDataAdapter(): ScenarioDataAdapter {
  const requestedSource = import.meta.env.VITE_DATA_SOURCE;
  if (requestedSource === 'frontend_simulation') return demoScenarioAdapter;
  return createBackendScenarioAdapter(import.meta.env.VITE_API_BASE_URL || '');
}
