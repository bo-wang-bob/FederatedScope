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
