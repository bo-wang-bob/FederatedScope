import type { MilitaryDomain, MilitaryNode } from '../types';
import { generateScenarioPartition } from '../utils/partition';

const domainDefinitions = [
  {
    id: 'Art' as const,
    serverId: 'OH-DT-S01',
    name: '数字孪生域',
    shortName: '数字孪生',
    modality: '仿真建模 / 三维渲染',
    featureDimension: 2048,
    unifiedDimension: 256,
    featureShift: 0.72,
    color: '#43d6ff',
    icon: '◎',
  },
  {
    id: 'Clipart' as const,
    serverId: 'OH-TS-S02',
    name: '战术符号域',
    shortName: '战术符号',
    modality: '战术图标 / 态势标绘',
    featureDimension: 1024,
    unifiedDimension: 256,
    featureShift: 0.58,
    color: '#8b7cff',
    icon: '⌁',
  },
  {
    id: 'Product' as const,
    serverId: 'OH-ED-S03',
    name: '装备数据库域',
    shortName: '装备数据库',
    modality: '档案图像 / 标准采集',
    featureDimension: 768,
    unifiedDimension: 256,
    featureShift: 0.64,
    color: '#27e6a7',
    icon: '◇',
  },
  {
    id: 'Real_World' as const,
    serverId: 'OH-FR-S04',
    name: '实景侦察域',
    shortName: '实景侦察',
    modality: '无人机 / 地面智能体图像',
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

export const defaultPartitionPreview = generateScenarioPartition(0.3, 20_260_815);

export const domains: MilitaryDomain[] = domainDefinitions.map((domain, domainIndex) => ({
  ...domain,
  status: domainIndex === 1 ? '等待节点' : '域内聚合',
  source: 'frontend_simulation',
  nodes: defaultPartitionPreview.domains[domainIndex].clients.map((partition, nodeIndex): MilitaryNode => {
    const malicious = (domainIndex === 0 && nodeIndex === 3) ||
      (domainIndex === 2 && nodeIndex === 1);
    const labels = partition.classProportions
      .map((value) => Math.round(value * 100))
      .sort((a, b) => b - a)
      .slice(0, 5);
    return {
      id: partition.clientId,
      domainId: domain.id,
      name: `${domain.shortName}节点 ${nodeIndex + 1}`,
      status: malicious && nodeIndex === 3 ? '已过滤' : statuses[(nodeIndex + domainIndex) % statuses.length],
      progress: malicious && nodeIndex === 3 ? 100 : 42 + ((nodeIndex * 11 + domainIndex * 9) % 53),
      latency: 18 + domainIndex * 11 + nodeIndex * 7,
      sampleCount: partition.sampleCount,
      quality: 72 + ((domainIndex * 7 + nodeIndex * 5) % 25),
      risk: malicious ? 0.86 + nodeIndex * 0.02 : 0.08 + ((domainIndex + nodeIndex) % 5) * 0.07,
      malicious,
      assessment: malicious ? (nodeIndex === 1 ? '疑似' : '过滤') :
        (domainIndex === 3 && nodeIndex === 4 ? '疑似' : '通过'),
      labels,
      classHistogram: partition.classHistogram,
      classProportions: partition.classProportions,
      coveredClassCount: partition.coveredClassCount,
      missingClassCount: partition.missingClassCount,
      dominantClassIndex: partition.dominantClassIndex,
      dominantClassRatio: partition.dominantClassRatio,
      labelEntropy: partition.labelEntropy,
      source: 'frontend_simulation',
    };
  }),
}));

export const nodeRows = domains.flatMap((domain) => domain.nodes);
