import { beforeEach, describe, expect, it } from 'vitest';
import { phases, useAppStore } from '../store/useAppStore';

describe('实验演示状态', () => {
  beforeEach(() => useAppStore.setState({ mode: 'backdoor', phaseIndex: 4, round: 18, running: true, lastSequence: 130, connectionState: 'connected' }));

  it('隐私与后门模式使用单一互斥字段', () => {
    useAppStore.getState().setMode('privacy');
    expect(useAppStore.getState().mode).toBe('privacy');
    useAppStore.getState().setMode('backdoor');
    expect(useAppStore.getState().mode).toBe('backdoor');
  });

  it('完成一个阶段周期后轮次递增', () => {
    useAppStore.setState({ phaseIndex: phases.length - 1, round: 18 });
    useAppStore.getState().nextPhase();
    expect(useAppStore.getState().phaseIndex).toBe(0);
    expect(useAppStore.getState().round).toBe(19);
  });

  it('通过统一事件入口推进演示阶段并拒绝重复序号', () => {
    const event = {
      id: 'demo-131', sequence: 131, experimentId: 'demo', type: 'stage.changed' as const,
      timestamp: '2026-08-15T00:00:00Z', source: 'frontend_simulation' as const,
      payload: { phaseIndex: 5, round: 18 },
    };
    useAppStore.getState().applyTrainingEvent(event);
    expect(useAppStore.getState()).toMatchObject({ phaseIndex: 5, lastSequence: 131 });
    useAppStore.getState().applyTrainingEvent({ ...event, payload: { phaseIndex: 1, round: 99 } });
    expect(useAppStore.getState()).toMatchObject({ phaseIndex: 5, round: 18 });
  });
});
