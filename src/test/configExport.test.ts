import { describe, expect, it } from 'vitest';
import { serializeExperimentConfig } from '../features/experiment/configExport';
import type { ExperimentConfig } from '../types';

describe('实验配置导出', () => {
  it('导出内容与创建实验请求体一致且只包含当前模式', () => {
    const config: ExperimentConfig = {
      schemaVersion: '1.0',
      idempotencyKey: 'idempotent-test',
      name: '后门防御实验',
      type: 'backdoor',
      scenarioId: 'SCN-TEST',
      common: {
        method: 'fedavg', rounds: 10, localEpochs: 1,
        participationRate: 1, batchSize: 8, learningRate: 0.001,
        seed: 42, device: 'cpu',
      },
      heterogeneity: null,
      privacy: null,
      backdoor: {
        attack: 'trigger_injection', defenseEnabled: true,
        maliciousRatio: 0.05, maliciousClients: ['OH-DT-C01'],
        startRound: 1, poisonRatio: 0.2, targetLabel: 0,
        featureStageDefense: true, trainingStageDefense: true,
      },
    };
    const exported = JSON.parse(serializeExperimentConfig(config));
    expect(exported).toEqual(config);
    expect(exported.privacy).toBeNull();
    expect(exported.backdoor.featureStageDefense).toBe(true);
  });
});
