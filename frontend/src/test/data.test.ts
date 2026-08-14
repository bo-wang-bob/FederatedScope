import { describe, expect, it } from 'vitest';
import {
  domains,
  membershipMetrics,
  nodeRows,
  propertyMetrics,
  reconstructionMetrics,
} from '../mock/data';

describe('演示场景数据', () => {
  it('每个域恰好包含一个唯一子服务器和多个节点', () => {
    expect(domains).toHaveLength(4);
    expect(new Set(domains.map((domain) => domain.serverId)).size).toBe(domains.length);
    domains.forEach((domain) => expect(domain.nodes.length).toBeGreaterThan(1));
  });

  it('节点编号全局唯一且归属于声明的域', () => {
    expect(new Set(nodeRows.map((node) => node.id)).size).toBe(nodeRows.length);
    domains.forEach((domain) => domain.nodes.forEach((node) => {
      expect(node.domainId).toBe(domain.id);
      expect(node.id).toMatch(/^D\d{2}-N\d{3}$/);
    }));
  });

  it('恶意真值与防御判断是两个独立字段', () => {
    expect(nodeRows.some((node) => node.malicious && node.assessment === '疑似')).toBe(true);
    expect(nodeRows.some((node) => !node.malicious && node.assessment === '疑似')).toBe(true);
  });
});

describe('隐私保护效果数据', () => {
  it.each([
    ['成员关系推断', membershipMetrics],
    ['属性推断', propertyMetrics],
    ['数据重建', reconstructionMetrics],
  ])('%s 包含攻击效果下降和任务准确率代价', (_, metrics) => {
    const attackMetrics = metrics.filter((metric) => metric.lowerIsBetter);
    expect(attackMetrics.length).toBeGreaterThan(0);
    attackMetrics.forEach((metric) => expect(metric.after).toBeLessThan(metric.before));
    const utility = metrics.find((metric) => metric.name === '任务准确率');
    expect(utility).toBeDefined();
    expect(utility!.after).toBeLessThanOrEqual(utility!.before);
  });
});
