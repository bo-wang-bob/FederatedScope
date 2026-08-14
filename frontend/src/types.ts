export type DomainKey = 'recon' | 'spectrum' | 'unmanned' | 'command';
export type ExperimentMode = 'baseline' | 'privacy' | 'backdoor';
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
}

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
