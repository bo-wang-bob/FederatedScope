import type { ScenarioPartitionPreview, ScenarioPreviewRequest } from '../types';
import { generateScenarioPartition } from '../utils/partition';

export interface ScenarioDataAdapter {
  readonly source: 'backend' | 'frontend_simulation';
  preview(request: ScenarioPreviewRequest, signal?: AbortSignal): Promise<ScenarioPartitionPreview>;
}

export const demoScenarioAdapter: ScenarioDataAdapter = {
  source: 'frontend_simulation',
  async preview(request, signal) {
    if (signal?.aborted) throw new DOMException('请求已取消', 'AbortError');
    return generateScenarioPartition(request.partition.alpha, request.partition.seed);
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
  };
}

export function resolveScenarioDataAdapter(): ScenarioDataAdapter {
  const requestedSource = import.meta.env.VITE_DATA_SOURCE;
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
  return requestedSource === 'backend' && apiBaseUrl
    ? createBackendScenarioAdapter(apiBaseUrl)
    : demoScenarioAdapter;
}
