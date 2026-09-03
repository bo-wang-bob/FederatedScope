export type DomainKey = 'Art' | 'Clipart' | 'Product' | 'Real_World';
export type ExperimentMode = 'heterogeneity' | 'privacy' | 'backdoor';
export type ExperimentMethod =
  | 'fedavg'
  | 'fedprox'
  | 'fedproto'
  | 'fedopt'
  | 'moon'
  | 'heterogeneous_solution';
export type ExecutionMode = 'standalone' | 'distributed';
export type DataSourceKind = 'backend' | 'frontend_simulation';
export type ExperimentStatus =
  | 'created'
  | 'validating'
  | 'queued'
  | 'running'
  | 'stopping'
  | 'stopped'
  | 'completed'
  | 'failed';
export type NodeStatus =
  | '待机'
  | '接收中'
  | '特征统计'
  | '本地训练'
  | '上传中'
  | '等待聚合'
  | '已完成'
  | '异常'
  | '已过滤'
  | '离线';

export interface MilitaryNode {
  id: string;
  domainId: DomainKey;
  name: string;
  status: NodeStatus;
  progress: number;
  latency: number;
  sampleCount: number;
  quality: number;
  risk: number;
  malicious: boolean;
  assessment: '通过' | '疑似' | '过滤';
  labels: number[];
  classHistogram: number[];
  classProportions: number[];
  coveredClassCount: number;
  missingClassCount: number;
  dominantClassIndex: number;
  dominantClassRatio: number;
  labelEntropy: number;
  source: DataSourceKind;
}

export interface MilitaryDomain {
  id: DomainKey;
  serverId: string;
  name: string;
  shortName: string;
  modality: string;
  featureDimension: number;
  unifiedDimension: number;
  featureShift: number;
  color: string;
  icon: string;
  status: string;
  nodes: MilitaryNode[];
  source: DataSourceKind;
}

export interface ClientPartitionPreview {
  clientId: string;
  domainKey: DomainKey;
  sampleCount: number;
  domainSampleRatio: number;
  classHistogram: number[];
  classProportions: number[];
  coveredClassCount: number;
  missingClassCount: number;
  dominantClassIndex: number;
  dominantClassRatio: number;
  labelEntropy: number;
}

export interface DomainPartitionPreview {
  domainKey: DomainKey;
  totalSamples: number;
  classCount: number;
  clients: ClientPartitionPreview[];
}

export interface ScenarioPartitionPreview {
  datasetKey: 'office-home';
  alpha: number;
  seed: number;
  partitionVersion: string;
  source: DataSourceKind;
  basis?: 'actual_dataset' | 'built_in_simulation';
  datasetFingerprint?: string;
  domains: DomainPartitionPreview[];
}

export interface ScenarioPreviewRequest {
  dataset: 'office-home';
  domains: DomainKey[];
  clientsPerDomain: 15;
  partition: {
    strategy: 'dirichlet';
    alpha: number;
    seed: number;
  };
}

export interface ScenarioRecord {
  scenarioId: string;
  createdAt: string;
  request: ScenarioPreviewRequest;
  preview: ScenarioPartitionPreview;
  artifacts?: Record<string, string>;
}

export interface ExperimentCommonConfig {
  method: ExperimentMethod;
  rounds: number;
  localEpochs: number;
  participationRate: number;
  batchSize: number;
  learningRate: number;
  seed: number;
  device: 'cpu' | 'cuda';
  fedproxMu?: number;
}

export interface HeterogeneityConfig {
  expansionTarget: number;
  featureBatchSize: number;
}

export interface PrivacyConfig {
  attack: 'membership' | 'property' | 'reconstruction';
  defenseEnabled: boolean;
  initialClip?: number;
  targetQuantile?: number;
  noiseMultiplier?: number;
  epsilon?: number;
}

export interface BackdoorConfig {
  attack: 'trigger_injection' | 'label_poisoning' | 'model_update_poisoning';
  defenseEnabled: boolean;
  maliciousRatio: number;
  maliciousClients: string[];
  startRound: number;
  poisonRatio: number;
  targetLabel: number;
  featureStageDefense: boolean;
  trainingStageDefense: boolean;
}

interface ExperimentConfigBase {
  schemaVersion: '1.0';
  idempotencyKey: string;
  name: string;
  scenarioId: string;
  common: ExperimentCommonConfig;
  execution?: {
    mode: ExecutionMode;
    topologyId?: 'lab-three-machine';
    group?: string;
    evaluationFrequency?: number;
    clientsPerSubserver?: number;
    windowsClientCount?: number;
    rootDevice?: number;
    statisticsUploadStaggerSeconds?: number;
    diagonalCovariance?: boolean;
  };
}

export type ExperimentConfig =
  | ExperimentConfigBase & {
      type: 'heterogeneity';
      heterogeneity: HeterogeneityConfig;
      privacy: null;
      backdoor: null;
    }
  | ExperimentConfigBase & {
      type: 'privacy';
      heterogeneity: null;
      privacy: PrivacyConfig;
      backdoor: null;
    }
  | ExperimentConfigBase & {
      type: 'backdoor';
      heterogeneity: null;
      privacy: null;
      backdoor: BackdoorConfig;
    };

export interface ExperimentMetricPoint {
  round: number;
  accuracy?: number;
  loss?: number;
  worstDomain?: number;
  domainGap?: number;
  attackSuccess?: number;
  privacyRisk?: number;
  reconstructionLoss?: number;
  reconstructionPsnr?: number;
  noiseMultiplier?: number;
  noiseStd?: number;
  truePositiveRate?: number;
  falsePositiveRate?: number;
  [key: string]: number | undefined;
}

export interface ExperimentRecord {
  experimentId: string;
  name: string;
  type: ExperimentMode;
  method: ExperimentMethod;
  executionMode?: ExecutionMode;
  topologyId?: string;
  group?: string;
  scenarioId: string;
  scenarioSummary: {
    dataset: string;
    alpha: number;
    seed: number;
    partitionVersion: string;
  };
  config: ExperimentConfig;
  status: ExperimentStatus;
  createdAt: string;
  startedAt?: string;
  endedAt?: string;
  updatedAt: string;
  sequence: number;
  round: number;
  totalRounds: number;
  metrics: ExperimentMetricPoint[];
  finalMetrics: Record<string, number>;
  error?: { code: string; message: string };
}

export interface Capabilities {
  apiVersion: string;
  datasets: Array<{
    key: string;
    name: string;
    domains: DomainKey[];
    clientsPerDomain: number;
    available: boolean;
  }>;
  devices: Array<'cpu' | 'cuda'>;
  methods: ExperimentMethod[];
  experimentTypes: ExperimentMode[];
  runner: {
    ready: boolean;
    dataReady: boolean;
    modelReady: boolean;
    templatesReady: boolean;
  };
  executionModes?: ExecutionMode[];
  distributed?: {
    ready: boolean;
    remoteReadinessCheckedByPreflight: boolean;
    topologyId: 'lab-three-machine';
    nodes: Array<{
      key: 'client' | 'subserver' | 'root';
      label: string;
      operatingSystem: 'windows' | 'linux';
    }>;
    cases: Array<{
      group: string;
      dataset: string;
      model: string;
      methods: ExperimentMethod[];
      clientCount: number;
    }>;
  };
  metrics: Record<string, {
    label: string;
    unit: 'ratio' | 'number' | 'dB';
    modes: string[];
  }>;
  parameters: Record<string, {
    minimum: number;
    maximum: number;
    default: number;
  }>;
}

export interface PreflightCheck {
  name: string;
  ready: boolean;
  message: string;
  details: Record<string, unknown>;
}

export interface PreflightResult {
  ready: boolean;
  checks: PreflightCheck[];
  template: string;
  dataRoot: string;
  modelPath: string;
  partitionManifest: string;
  executionMode?: ExecutionMode;
  topologyId?: string;
  topology?: Array<{
    node: string;
    label: string;
    ready: boolean | null;
    message: string;
  }>;
}

export type TrainingEventType =
  | 'experiment.started'
  | 'experiment.stopping'
  | 'experiment.stopped'
  | 'stage.changed'
  | 'round.started'
  | 'client.status.changed'
  | 'client.metric.updated'
  | 'defense.decision'
  | 'round.completed'
  | 'metric.updated'
  | 'warning.raised'
  | 'log.received'
  | 'topology.status.changed'
  | 'experiment.completed'
  | 'experiment.failed';

export interface TrainingEvent<T = Record<string, unknown>> {
  id: string;
  sequence: number;
  experimentId: string;
  type: TrainingEventType;
  timestamp: string;
  source: DataSourceKind;
  payload: T;
}

export interface ClientTrainingState {
  clientId: string;
  domainKey: DomainKey;
  status: NodeStatus;
  progress: number;
  round: number;
  sampleCount?: number;
  assessment?: '通过' | '疑似' | '过滤';
  clipBound?: number;
  clipFactor?: number;
  noiseStd?: number;
  rawNorm?: number;
  sanitizedNorm?: number;
  source: DataSourceKind;
}

export interface TrainingSnapshot {
  experimentId: string;
  sequence: number;
  phaseIndex: number;
  round: number;
  status: ExperimentStatus | 'disconnected';
  clients: Record<string, ClientTrainingState>;
  topology?: Record<string, {
    node: string;
    label: string;
    status: string;
    ready: boolean;
    source: DataSourceKind;
  }>;
  totalRounds?: number;
  metrics?: ExperimentMetricPoint[];
  recentEvents?: TrainingEvent[];
  type?: ExperimentMode;
  method?: ExperimentMethod;
  name?: string;
  finalMetrics?: Record<string, number>;
  error?: { code: string; message: string };
  source: DataSourceKind;
  updatedAt: string;
}

export type TrainingConnectionState = 'connecting' | 'connected' | 'recovering' | 'disconnected';

export type HierarchyPhase =
  | '中央下发'
  | '域内广播'
  | '节点处理'
  | '节点上传'
  | '域内聚合'
  | '域级上传'
  | '全域聚合';
