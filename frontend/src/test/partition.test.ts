import { describe, expect, it } from 'vitest';
import { demoScenarioAdapter } from '../api/scenarioAdapter';
import { distributionSummary, generateScenarioPartition } from '../utils/partition';

describe('狄利克雷客户端划分预览', () => {
  it('相同 alpha 和随机种子产生完全相同的四域划分', () => {
    const first = generateScenarioPartition(0.3, 20260815);
    const second = generateScenarioPartition(0.3, 20260815);
    expect(second).toEqual(first);
    expect(first.domains).toHaveLength(4);
    expect(first.domains.flatMap((domain) => domain.clients)).toHaveLength(60);
  });

  it('每个域和客户端的样本与类别计数守恒', () => {
    const preview = generateScenarioPartition(0.1, 20260815);
    preview.domains.forEach((domain) => {
      expect(domain.clients).toHaveLength(15);
      expect(domain.clients.reduce((sum, client) => sum + client.sampleCount, 0)).toBe(domain.totalSamples);
      domain.clients.forEach((client) => {
        expect(client.sampleCount).toBeGreaterThan(0);
        expect(client.classHistogram).toHaveLength(65);
        expect(client.classHistogram.reduce((sum, value) => sum + value, 0)).toBe(client.sampleCount);
        expect(client.classProportions.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
        expect(client.coveredClassCount + client.missingClassCount).toBe(65);
      });
    });
  });

  it('较小 alpha 呈现更强的类别集中趋势', () => {
    const concentrated = distributionSummary(generateScenarioPartition(0.1, 20260815));
    const balanced = distributionSummary(generateScenarioPartition(10, 20260815));
    expect(concentrated.maxDominantRatio).toBeGreaterThan(balanced.maxDominantRatio);
    expect(concentrated.meanCoverage).toBeLessThan(balanced.meanCoverage);
  });

  it('适配器保持前后端兼容请求结构并标记演示来源', async () => {
    const preview = await demoScenarioAdapter.preview({
      dataset: 'office-home',
      domains: ['Art', 'Clipart', 'Product', 'Real_World'],
      clientsPerDomain: 15,
      partition: { strategy: 'dirichlet', alpha: 0.5, seed: 7 },
    });
    expect(preview.source).toBe('frontend_simulation');
    expect(preview.alpha).toBe(0.5);
    expect(preview.seed).toBe(7);
  });

  it('拒绝超出设计范围的 alpha', () => {
    expect(() => generateScenarioPartition(0.01)).toThrow(RangeError);
    expect(() => generateScenarioPartition(11)).toThrow(RangeError);
  });
});
