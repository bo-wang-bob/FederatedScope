export interface RequestConfig {
  group: string; method: string; name: string; rounds: number; clientCount: number;
  sampleClients: number; batchSize: number; localEpochs: number; learningRate: number;
  seed: number; splitSeed: number; alpha: number; gpu: number; evaluationFrequency: number; samplesPerClient: number;
}
export interface Group {
  id: string; dataset: string; backbone: string; domains: number; cacheFound: boolean;
  cacheFiles: number; cacheBytes: number; partitionLocked: boolean;
  methods: { id: string; label: string; enabled: boolean; reason: string | null; defaults: RequestConfig }[];
}
export interface Catalog { groups: Group[]; host: string; address: string; protocol: string; evaluationPolicy: string }
export interface Resource {
  at: string; hostname: string; cpuPercent: number; memoryPercent: number; memoryUsed: number;
  memoryTotal: number; diskFree: number; diskTotal: number; gpuError: string | null;
  gpus: { index: number; name: string; utilization: number; memoryUsedMiB: number; memoryTotalMiB: number; temperature: number }[];
}
export interface Client { id: number; domain: string; samples: number; stage: string; round?: number; loss?: number; accuracy?: number; histogram: number[] }
export interface Point { round: number; accuracy: number; domainMean: number; domains: Record<string, number>; trainLoss?: number | null; worstDomain: number; domainGap: number }
export interface ClassMetric { classIndex: number; support: number; precision: number; recall: number; f1: number }
export interface Metric { accuracy: number; macroF1: number; precision: number; recall: number; samples: number; perClass: ClassMetric[]; confusionMatrix: number[][] }
export interface DataInfo {
  clientCount: number; trainSamples: number; testSamples: number; classes: string[];
  domains: { name: string; testSamples: number }[]; fingerprint: string; testFingerprint: string;
  testProvenance: Record<string, string>; clients?: Client[];
}
export interface EvaluationResult extends Metric { domains: Record<string, Metric>; domainMean: number; worstDomain: number; domainGap: number; elapsedSeconds: number }
export interface Job {
  id: string; action: 'train' | 'inspect' | 'evaluate'; status: string; stage: string;
  request: RequestConfig & { modelId?: string; testsetId?: string; domains?: string[]; classes?: number[] };
  createdAt: string; updatedAt: string; endedAt?: string; error: string | null;
  cleanup: { ok: boolean; message: string }; clients: Record<string, Client>; metrics: Point[];
  data?: DataInfo; config: Record<string, unknown>; provenance: Record<string, unknown>;
  result?: DataInfo | EvaluationResult;
}
export interface LibraryItem {
  id: string; jobId: string; name: string; group: string; method: string; kind?: string; samples?: number;
  domains: { name: string; testSamples: number }[]; classes: string[]; featureSpace: string; sha256: string;
}
export interface Library { models: LibraryItem[]; testsets: LibraryItem[] }
export const terminal = (status: string) => ['completed', 'failed', 'stopped', 'interrupted'].includes(status);
export const statusText: Record<string, string> = { queued: '排队中', running: '运行中', stopping: '正在停止', completed: '已完成', failed: '失败', stopped: '已停止', interrupted: '重启中断' };
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/platform/${path}`, { method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(15000) });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message || `HTTP ${response.status}`);
  return payload.data;
}
export const key = () => globalThis.crypto?.randomUUID?.() ?? `request-${Date.now()}-${Math.random().toString(16).slice(2)}`;
export const percent = (x?: number) => x == null ? '—' : `${(x * 100).toFixed(2)}%`;
export const bytes = (x: number) => `${(x / 1024 ** 3).toFixed(1)} GB`;
