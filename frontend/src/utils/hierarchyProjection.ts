import type { HierarchyPhase } from '../types';

export function isUpperDirection(phase: HierarchyPhase): boolean {
  return ['节点上传', '域内聚合', '域级上传', '全域聚合'].includes(phase);
}

export function isCentralLinkActive(phase: HierarchyPhase): boolean {
  return ['中央下发', '域级上传', '全域聚合'].includes(phase);
}

export function isClientLinkActive(phase: HierarchyPhase): boolean {
  return ['域内广播', '节点处理', '节点上传', '域内聚合'].includes(phase);
}

export function canDomainAggregate(uploaded: number, total: number, filtered: number): boolean {
  return uploaded + filtered >= total && uploaded > 0;
}

export function canCentralAggregate(completedDomains: number, totalDomains: number): boolean {
  return totalDomains > 0 && completedDomains === totalDomains;
}
