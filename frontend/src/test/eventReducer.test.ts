import { afterEach, describe, expect, it, vi } from 'vitest';
import { createDemoTrainingAdapter } from '../api/trainingAdapter';
import type { TrainingEvent, TrainingSnapshot } from '../types';
import { mergeTrainingEvent, recoverTrainingSnapshot } from '../utils/eventReducer';

const baseSnapshot: TrainingSnapshot = {
  experimentId: 'demo',
  sequence: 10,
  phaseIndex: 0,
  round: 2,
  status: 'running',
  clients: {},
  source: 'frontend_simulation',
  updatedAt: '2026-08-15T00:00:00Z',
};

afterEach(() => vi.useRealTimers());

describe('训练事件适配与归并', () => {
  it('忽略重复和乱序事件', () => {
    const stale: TrainingEvent = {
      id: 'evt-9', sequence: 9, experimentId: 'demo', type: 'stage.changed',
      timestamp: '2026-08-15T00:00:01Z', source: 'backend', payload: { phaseIndex: 4, round: 3 },
    };
    expect(mergeTrainingEvent(baseSnapshot, stale)).toBe(baseSnapshot);
  });

  it('合并阶段和客户端状态并保留数据来源', () => {
    const stage = mergeTrainingEvent(baseSnapshot, {
      id: 'evt-11', sequence: 11, experimentId: 'demo', type: 'stage.changed',
      timestamp: '2026-08-15T00:00:02Z', source: 'backend', payload: { phaseIndex: 3, round: 3 },
    });
    expect(stage).toMatchObject({ sequence: 11, phaseIndex: 3, round: 3, source: 'backend' });
    const client = mergeTrainingEvent(stage, {
      id: 'evt-12', sequence: 12, experimentId: 'demo', type: 'client.status.changed',
      timestamp: '2026-08-15T00:00:03Z', source: 'backend',
      payload: { clientId: 'OH-DT-C01', domainKey: 'Art', status: '上传中', progress: 72, round: 3 },
    });
    expect(client.clients['OH-DT-C01']).toMatchObject({ progress: 72, source: 'backend' });
  });

  it('断流恢复只接受不旧于当前状态的快照', () => {
    expect(recoverTrainingSnapshot(baseSnapshot, { ...baseSnapshot, sequence: 8 })).toBe(baseSnapshot);
    const recovered = { ...baseSnapshot, sequence: 14, phaseIndex: 5, source: 'backend' as const };
    expect(recoverTrainingSnapshot(baseSnapshot, recovered)).toEqual(recovered);
  });

  it('合并指标并正确处理停止终态', () => {
    const metric = mergeTrainingEvent(baseSnapshot, {
      id: 'evt-11', sequence: 11, experimentId: 'demo', type: 'metric.updated',
      timestamp: '2026-08-15T00:00:02Z', source: 'backend',
      payload: { round: 3, accuracy: 0.75 },
    });
    expect(metric.metrics).toEqual([{ round: 3, accuracy: 0.75 }]);
    const stopped = mergeTrainingEvent(metric, {
      id: 'evt-12', sequence: 12, experimentId: 'demo', type: 'experiment.stopped',
      timestamp: '2026-08-15T00:00:03Z', source: 'backend', payload: {},
    });
    expect(stopped.status).toBe('stopped');
  });

  it('合并客户端隐私统计并应用结构化防御判定', () => {
    const seeded: TrainingSnapshot = {
      ...baseSnapshot,
      clients: {
        'OH-DT-C01': {
          clientId: 'OH-DT-C01', domainKey: 'Art', status: '待机',
          progress: 0, round: 0, sampleCount: 17,
          assessment: '通过', source: 'backend',
        },
      },
    };
    const privacy = mergeTrainingEvent(seeded, {
      id: 'evt-11', sequence: 11, experimentId: 'demo',
      type: 'client.metric.updated', timestamp: '2026-08-15T00:00:02Z',
      source: 'backend', payload: {
        clientId: 'OH-DT-C01', domainKey: 'Art', status: '上传中',
        progress: 85, round: 3, noiseStd: 0.12,
      },
    });
    expect(privacy.clients['OH-DT-C01']).toMatchObject({
      sampleCount: 17, noiseStd: 0.12,
    });
    const defended = mergeTrainingEvent(privacy, {
      id: 'evt-12', sequence: 12, experimentId: 'demo',
      type: 'defense.decision', timestamp: '2026-08-15T00:00:03Z',
      source: 'backend', payload: {
        round: 3, droppedClientIds: [1],
        truePositiveRate: 1, falsePositiveRate: 0,
      },
    });
    expect(defended.clients['OH-DT-C01']).toMatchObject({
      status: '已过滤', assessment: '过滤', progress: 100,
    });
    expect(defended.metrics).toEqual([{
      round: 3, truePositiveRate: 1, falsePositiveRate: 0,
    }]);
  });

  it('演示适配器按顺序输出阶段事件并报告连接状态', () => {
    vi.useFakeTimers();
    const events: TrainingEvent[] = [];
    const states: string[] = [];
    const adapter = createDemoTrainingAdapter(baseSnapshot, 1_000);
    const subscription = adapter.subscribe('demo', 10, (event) => events.push(event), (state) => states.push(state));
    vi.advanceTimersByTime(2_100);
    subscription.close();
    expect(events.map((event) => event.sequence)).toEqual([11, 12]);
    expect(events.map((event) => event.source)).toEqual(['frontend_simulation', 'frontend_simulation']);
    expect(states).toEqual(['connected', 'disconnected']);
  });
});
