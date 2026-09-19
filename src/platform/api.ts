export interface RequestConfig {
  group: string; method: string; name: string; rounds: number; clientCount: number;
  sampleClients: number; batchSize: number; localEpochs: number; learningRate: number;
  seed: number; splitSeed: number; alpha: number; gpu: number; evaluationFrequency: number; samplesPerClient: number;
  augmentationMode?: 'none' | 'generate' | 'reuse' | 'auto'; allowLegacyAugmentation?: boolean;
  augmentationSourceId?: string;
  generatedPerSample?: number; generatedPerPrototype?: number; targetPerClass?: number; covarianceScale?: number;
}
export interface Group {
  id: string; dataset: string; backbone: string; domains: number; cacheFound: boolean;
  cacheFiles: number; cacheBytes: number; partitionLocked: boolean;
  augmentationSources?: { id: string; name: string; request: RequestConfig }[];
  lastPreflight?: { id: string; status: string; at: string; error: string | null } | null;
  methods: { id: string; label: string; enabled: boolean; reason: string | null; augmentedCacheFound?: boolean; defaults: RequestConfig }[];
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
  partitionFingerprint?: string;
  augmentation?: { mode: string; provenance: string; warning?: string | null; cachedSamples?: number; generatedClients?: number };
}
export interface EvaluationResult extends Metric { domains: Record<string, Metric>; domainMean: number; worstDomain: number; domainGap: number; elapsedSeconds: number }
export interface Job {
  id: string; action: 'train' | 'inspect' | 'evaluate' | 'predict' | 'test-upload'; status: string; stage: string;
  request: RequestConfig & { modelId?: string; testsetId?: string; domains?: string[]; classes?: number[]; sampleId?: string; imageSha256?: string };
  createdAt: string; updatedAt: string; endedAt?: string; error: string | null;
  cleanup: { ok: boolean; message: string }; clients: Record<string, Client>; metrics: Point[];
  data?: DataInfo; config: Record<string, unknown>; provenance: Record<string, unknown>;
  result?: DataInfo | EvaluationResult | Prediction | UploadedPrediction;
}
export interface LibraryItem {
  id: string; jobId: string; name: string; group: string; method: string; kind?: string; samples?: number;
  domains: { name: string; testSamples: number }[]; classes: string[]; featureSpace: string; sha256: string; trainingRounds?: number;
  augmentationWarning?: string | null;
}
export interface TestSample { id: string; index: number; domain: string; label: number; className: string; filename: string; imageUrl: string; imageAvailable: boolean; imageSha256?: string; imageError?: string }
export interface SamplePage { items: TestSample[]; total: number; offset: number; limit: number; testFingerprint: string; provenance: Record<string, string> }
export interface Prediction {
  sampleId: string; domain: string; filename: string; label: number; labelName: string;
  predictedClass: number; predictedName: string; correct: boolean; confidence: number;
  topK: { classIndex: number; className: string; score: number }[];
  inferenceMs: number; elapsedSeconds: number; checkpointSha256: string; testBundleSha256: string;
  imageSha256: string; manifestSha256: string; testProvenance: string; inferenceContract: string;
}
export interface UploadedPrediction {
  samples: number; labelled: boolean; metrics: Metric | null; checkpointSha256: string;
  items: { index: number; filename: string; labelName: string | null; predictedName: string;
    confidence: number; correct: boolean | null; imageUrl: string }[];
}
export interface Library { models: LibraryItem[]; testsets: LibraryItem[] }
export interface BackdoorTestsetInfo {
  id: string; name: string; count: number; skipped: number; unlabelled: number;
}
export interface BackdoorTestset {
  exported: boolean; total: number; maxIds: number; base: string; message?: string;
  classNames: string[];
  domains: { name: string; count: number }[];
  labels: { index: number; name: string; count: number }[];
  runs: { attack: string | null; defense: string | null };
  /** 当前应用的上传测试集; 未上传时为空 */
  testset?: BackdoorTestsetInfo | null;
}
export interface BackdoorPick { ids: string[]; labels: number[]; total: number; count: number; seed?: number;
  /** 是否按"攻击命中 ∧ 防御拦住"加权抽样（后端有全测试集预测缓存时为 true） */
  filtered?: boolean;
  /** 满足"攻击命中 ∧ 防御拦住"的候选样本数 */
  preferred?: number;
  /** 预筛样本在抽样名额中的目标占比，其余名额随机 */
  preferRatio?: number;
  /** 后门目标类 */
  targetLabel?: number; targetName?: string }
export interface BackdoorStat { correct: number; total: number; accuracy: number; asr: number; asrEligible?: number; asrRate: number | null }
export interface BackdoorResult {
  ids: string[]; classNames: string[]; targetLabel: number; targetName: string; attackName: string;
  runs: Record<string, { name: string; display: string }>;
  images: { id: string; label: number; labelName: string;
    clean: { label: number; name: string };
    triggered: { label: number; name: string; hit: boolean };
    defense?: { label: number; name: string; hit: boolean } }[];
  stats: Record<string, BackdoorStat>;
  paths: Record<string, string>;
}
export interface BackdoorJob {
  id: string; action: string; ids: string[]; name: string; base: string; device: string;
  runs: Record<string, string>; status: string; stage: string; error: string | null;
  createdAt: string; updatedAt: string; endedAt?: string;
  images?: Record<string, string>; result?: BackdoorResult;
}
export interface BackdoorTrainingTemplate { key: string; file: string; label: string; exists: boolean }
export interface BackdoorTrainingDataset { id: string; name: string; classes: number; count: number; layout: string }
export interface BackdoorTrainingGroup {
  token: string; base: string; baseline: string; attack: string; defense: string;
  dataRoot: string; datasetId: string; datasetName: string; classes: string[]; createdAt: string;
  testsetId?: string; testsetName?: string; testsetCount?: number; testsetSkipped?: number;
  testsetUnlabelled?: number; testsetAppliedAt?: string;
}
export interface BackdoorTrainingJob {
  id: string; action: string; token: string; datasetId: string; datasetName: string; base: string;
  device: string; total: number; createdAt: string; updatedAt: string; startedAt?: string; endedAt?: string;
  status: string; stage: string; stageIndex: number; error: string | null; pid?: number;
  results?: { key: string; label: string; expname: string }[]; group?: BackdoorTrainingGroup;
}
export interface BackdoorTrainingStatus {
  runnable: boolean; datasets: BackdoorTrainingDataset[]; templates: BackdoorTrainingTemplate[];
  missing: string[]; base: string; job: BackdoorTrainingJob | null; group: BackdoorTrainingGroup | null;
  testsets?: BackdoorTrainingDataset[];
}
export const backdoorImageUrl = (id: string) => `/api/platform/backdoor/testset/${id}/image`;
export const terminal = (status: string) => ['completed', 'failed', 'stopped', 'interrupted'].includes(status);
export const statusText: Record<string, string> = { queued: '排队中', running: '运行中', stopping: '正在停止', completed: '已完成', failed: '失败', stopped: '已停止', interrupted: '重启中断' };
export class PlatformApiError extends Error {
  constructor(message: string, public status: number, public code?: string) { super(message); this.name = 'PlatformApiError'; }
}
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/platform/${path}`, { method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(90000) });
  const payload = await response.json();
  if (!response.ok) throw new PlatformApiError(payload.error?.message || `HTTP ${response.status}`, response.status, payload.error?.code);
  return payload.data;
}
export const key = () => globalThis.crypto?.randomUUID?.() ?? `request-${Date.now()}-${Math.random().toString(16).slice(2)}`;
export const percent = (x?: number) => x == null ? '—' : `${(x * 100).toFixed(2)}%`;
export const methodLabel = (id?: string) => ({ heterogeneous_solution: '本架构', ggeur: '本架构', GGEUR: '本架构', fedavg: 'FedAvg', fedprox: 'FedProx', fedproto: 'FedProto', fedopt: 'FedOpt', moon: 'MOON' }[id || ''] || id || '—');
export const bytes = (x: number) => `${(x / 1024 ** 3).toFixed(1)} GB`;
const legacyCoordinationTerm = new TextDecoder().decode(Uint8Array.of(0xe8,0x81,0x94,0xe9,0x82,0xa6));
export const terminology = (value?: string | null) => (value || '').replaceAll(`跨域${legacyCoordinationTerm}学习`, '跨域协同训练').replaceAll(legacyCoordinationTerm, '协同');
