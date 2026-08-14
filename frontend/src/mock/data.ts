import type {
  EventItem,
  MilitaryDomain,
  MilitaryNode,
  PrivacyMetric,
  RoundMetric,
} from '../types';

const domainDefinitions = [
  {
    id: 'recon' as const,
    serverId: 'DS-01',
    name: '态势感知域',
    shortName: '态势感知',
    modality: '图像 / 目标特征',
    featureDimension: 2048,
    unifiedDimension: 256,
    featureShift: 0.72,
    color: '#43d6ff',
    icon: '◎',
  },
  {
    id: 'spectrum' as const,
    serverId: 'DS-02',
    name: '电磁感知域',
    shortName: '电磁感知',
    modality: '频谱 / 时序特征',
    featureDimension: 1024,
    unifiedDimension: 256,
    featureShift: 0.58,
    color: '#8b7cff',
    icon: '⌁',
  },
  {
    id: 'unmanned' as const,
    serverId: 'DS-03',
    name: '无人平台域',
    shortName: '无人平台',
    modality: '遥测 / 多传感器',
    featureDimension: 768,
    unifiedDimension: 256,
    featureShift: 0.64,
    color: '#27e6a7',
    icon: '◇',
  },
  {
    id: 'command' as const,
    serverId: 'DS-04',
    name: '指挥决策域',
    shortName: '指挥决策',
    modality: '文本 / 结构化特征',
    featureDimension: 1536,
    unifiedDimension: 256,
    featureShift: 0.81,
    color: '#ffbd52',
    icon: '△',
  },
];

const statuses: MilitaryNode['status'][] = [
  '本地训练',
  '上传中',
  '等待聚合',
  '本地训练',
  '接收中',
];

export const domains: MilitaryDomain[] = domainDefinitions.map((domain, domainIndex) => ({
  ...domain,
  status: domainIndex === 1 ? '等待节点' : '域内聚合',
  nodes: Array.from({ length: 5 }, (_, nodeIndex): MilitaryNode => {
    const malicious = (domainIndex === 0 && nodeIndex === 3) ||
      (domainIndex === 2 && nodeIndex === 1);
    const labels = Array.from({ length: 5 }, (_, labelIndex) =>
      10 + ((domainIndex * 17 + nodeIndex * 23 + labelIndex * 13) % 75));
    return {
      id: `D0${domainIndex + 1}-N${String(nodeIndex + 1).padStart(3, '0')}`,
      domainId: domain.id,
      name: `${domain.shortName}节点 ${nodeIndex + 1}`,
      status: malicious && nodeIndex === 3 ? '已过滤' : statuses[(nodeIndex + domainIndex) % statuses.length],
      progress: malicious && nodeIndex === 3 ? 100 : 42 + ((nodeIndex * 11 + domainIndex * 9) % 53),
      latency: 18 + domainIndex * 11 + nodeIndex * 7,
      sampleCount: 860 + domainIndex * 410 + nodeIndex * 337,
      quality: 72 + ((domainIndex * 7 + nodeIndex * 5) % 25),
      risk: malicious ? 0.86 + nodeIndex * 0.02 : 0.08 + ((domainIndex + nodeIndex) % 5) * 0.07,
      malicious,
      assessment: malicious ? (nodeIndex === 1 ? '疑似' : '过滤') :
        (domainIndex === 3 && nodeIndex === 4 ? '疑似' : '通过'),
      labels,
    };
  }),
}));

export const roundMetrics: RoundMetric[] = Array.from({ length: 30 }, (_, index) => {
  const round = index + 1;
  return {
    round,
    accuracy: Number((48 + 39 * (1 - Math.exp(-round / 9)) + Math.sin(round / 2) * 0.8).toFixed(2)),
    worstDomain: Number((40 + 38 * (1 - Math.exp(-round / 10)) + Math.cos(round / 3)).toFixed(2)),
    attackSuccess: Number(Math.max(2, 72 - round * 1.85 + Math.sin(round) * 2).toFixed(2)),
    privacyRisk: Number(Math.max(12, 68 - round * 1.5).toFixed(2)),
  };
});

export const events: EventItem[] = [
  { id: 1, time: '14:32:08', level: 'success', source: '中央服务器', message: '第 18 轮全域状态已下发至 4 个域子服务器' },
  { id: 2, time: '14:32:11', level: 'info', source: 'DS-01', message: '态势感知域已完成域内广播，5 个节点开始处理' },
  { id: 3, time: '14:32:16', level: 'warning', source: 'D03-N002', message: '更新偏移超过动态阈值，进入复核队列' },
  { id: 4, time: '14:32:19', level: 'danger', source: 'D01-N004', message: '训练更新已被防御策略过滤' },
  { id: 5, time: '14:32:23', level: 'success', source: 'DS-04', message: '指挥决策域完成模拟域内聚合' },
  { id: 6, time: '14:32:25', level: 'info', source: '系统', message: '当前有效域覆盖率 100%，满足全域聚合条件' },
];

export const domainAccuracy = [
  { name: '态势感知域', before: 71.4, after: 85.8, samples: 7400 },
  { name: '电磁感知域', before: 65.2, after: 82.6, samples: 10200 },
  { name: '无人平台域', before: 68.8, after: 84.2, samples: 8900 },
  { name: '指挥决策域', before: 59.3, after: 80.7, samples: 13100 },
];

export const featureDistances = [
  [0, 0.72, 0.64, 0.81],
  [0.72, 0, 0.56, 0.69],
  [0.64, 0.56, 0, 0.61],
  [0.81, 0.69, 0.61, 0],
];

export const membershipMetrics: PrivacyMetric[] = [
  { name: '攻击准确率', before: 78.6, after: 54.2, unit: '%', lowerIsBetter: true },
  { name: 'AUC', before: 0.84, after: 0.55, lowerIsBetter: true },
  { name: '攻击优势值', before: 0.53, after: 0.08, lowerIsBetter: true },
  { name: '任务准确率', before: 86.3, after: 83.9, unit: '%' },
];

export const propertyMetrics: PrivacyMetric[] = [
  { name: '属性准确率', before: 74.1, after: 42.8, unit: '%', lowerIsBetter: true },
  { name: '宏平均 F1', before: 0.71, after: 0.39, lowerIsBetter: true },
  { name: '平均置信度', before: 0.82, after: 0.51, lowerIsBetter: true },
  { name: '任务准确率', before: 86.3, after: 83.9, unit: '%' },
];

export const reconstructionMetrics: PrivacyMetric[] = [
  { name: '特征相似度', before: 0.81, after: 0.27, lowerIsBetter: true },
  { name: '结构相似度', before: 0.76, after: 0.21, lowerIsBetter: true },
  { name: '标签恢复率', before: 68.4, after: 17.2, unit: '%', lowerIsBetter: true },
  { name: '任务准确率', before: 86.3, after: 83.9, unit: '%' },
];

export const nodeRows = domains.flatMap((domain) => domain.nodes);
