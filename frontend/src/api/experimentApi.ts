import type {
  Capabilities,
  ExperimentConfig,
  ExperimentRecord,
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
  preflight(value: ExperimentConfig, signal?: AbortSignal) {
    return request<Record<string, string>>('/api/experiments/preflight', {
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
};
