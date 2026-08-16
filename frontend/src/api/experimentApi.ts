import type {
  Capabilities,
  ExperimentConfig,
  ExperimentRecord,
  PreflightResult,
  ScenarioRecord,
  ScenarioPreviewRequest,
} from '../types';

export class ApiError extends Error {
  code: string;
  fieldErrors: Record<string, string>;

  constructor(message: string, code = 'REQUEST_FAILED', fieldErrors: Record<string, string> = {}) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.fieldErrors = fieldErrors;
  }
}

export function apiBaseUrl() {
  return (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl()}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  const payload = await response.json().catch(() => ({})) as {
    data?: T;
    error?: { code?: string; message?: string; fieldErrors?: Record<string, string> };
  };
  if (!response.ok || payload.data === undefined) {
    throw new ApiError(
      payload.error?.message || `请求失败：HTTP ${response.status}`,
      payload.error?.code,
      payload.error?.fieldErrors,
    );
  }
  return payload.data;
}

export const experimentApi = {
  capabilities(signal?: AbortSignal) {
    return request<Capabilities>('/api/capabilities', { signal });
  },
  createScenario(value: ScenarioPreviewRequest, signal?: AbortSignal) {
    return request<ScenarioRecord>('/api/scenarios', {
      method: 'POST', body: JSON.stringify(value), signal,
    });
  },
  getScenario(scenarioId: string, signal?: AbortSignal) {
    return request<ScenarioRecord>(`/api/scenarios/${encodeURIComponent(scenarioId)}`, { signal });
  },
  preflight(value: ExperimentConfig, signal?: AbortSignal) {
    return request<PreflightResult>('/api/experiments/preflight', {
      method: 'POST', body: JSON.stringify(value), signal,
    });
  },
  create(value: ExperimentConfig, signal?: AbortSignal) {
    return request<ExperimentRecord>('/api/experiments', {
      method: 'POST', body: JSON.stringify(value), signal,
    });
  },
  list(signal?: AbortSignal) {
    return request<ExperimentRecord[]>('/api/experiments', { signal });
  },
  get(experimentId: string, signal?: AbortSignal) {
    return request<ExperimentRecord>(`/api/experiments/${encodeURIComponent(experimentId)}`, { signal });
  },
  stop(experimentId: string, signal?: AbortSignal) {
    return request<ExperimentRecord>(`/api/experiments/${encodeURIComponent(experimentId)}/stop`, {
      method: 'POST', body: '{}', signal,
    });
  },
  logs(experimentId: string, limit = 200, signal?: AbortSignal) {
    return request<string[]>(`/api/experiments/${encodeURIComponent(experimentId)}/logs?limit=${limit}`, { signal });
  },
};
