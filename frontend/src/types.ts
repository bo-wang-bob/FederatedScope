export type DomainKey = 'Art' | 'Clipart' | 'Product' | 'Real_World';
export type ExperimentMode = 'baseline' | 'privacy' | 'backdoor';
export type DataSourceKind = 'backend' | 'frontend_simulation';
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

export type TrainingEventType =
  | 'experiment.started'
  | 'stage.changed'
  | 'round.started'
  | 'client.status.changed'
  | 'client.metric.updated'
  | 'round.completed'
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
  source: DataSourceKind;
}

export interface TrainingSnapshot {
  experimentId: string;
  sequence: number;
  phaseIndex: number;
  round: number;
  status: 'running' | 'paused' | 'completed' | 'failed' | 'disconnected';
  clients: Record<string, ClientTrainingState>;
  source: DataSourceKind;
  updatedAt: string;
}

export type TrainingConnectionState = 'connecting' | 'connected' | 'recovering' | 'disconnected';

export interface RoundMetric {
  round: number;
  accuracy: number;
  worstDomain: number;
  attackSuccess: number;
  privacyRisk: number;
}

export interface EventItem {
  id: number;
  time: string;
  level: 'info' | 'success' | 'warning' | 'danger';
  source: string;
  message: string;
}

export type HierarchyPhase =
  | '中央下发'
  | '域内广播'
  | '节点处理'
  | '节点上传'
  | '域内聚合'
  | '域级上传'
  | '全域聚合';

export interface PrivacyMetric {
  name: string;
  before: number;
  after: number;
  unit?: string;
  lowerIsBetter?: boolean;
}
