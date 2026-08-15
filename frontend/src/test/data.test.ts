import { describe, expect, it } from 'vitest';
import {
  domains,
  membershipMetrics,
  nodeRows,
  propertyMetrics,
  reconstructionMetrics,
} from '../mock/data';

describe('演示场景数据', () => {
  it('四个后端域键映射到指定前端域名且每域包含 15 个客户端', () => {
    expect(domains).toHaveLength(4);
    expect(domains.map((domain) => [domain.id, domain.name])).toEqual([
      ['Art', '数字孪生域'],
      ['Clipart', '战术符号域'],
      ['Product', '装备数据库域'],
      ['Real_World', '实景侦察域'],
    ]);
    expect(new Set(domains.map((domain) => domain.serverId)).size).toBe(domains.length);
    domains.forEach((domain) => expect(domain.nodes).toHaveLength(15));
    expect(nodeRows).toHaveLength(60);
  });

  it('节点编号全局唯一且归属于声明的域', () => {
    expect(new Set(nodeRows.map((node) => node.id)).size).toBe(nodeRows.length);
    domains.forEach((domain) => domain.nodes.forEach((node) => {
      expect(node.domainId).toBe(domain.id);
      expect(node.id).toMatch(/^OH-(DT|TS|ED|FR)-C\d{2}$/);
      expect(node.classHistogram.reduce((sum, value) => sum + value, 0)).toBe(node.sampleCount);
      expect(node.classProportions.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
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
