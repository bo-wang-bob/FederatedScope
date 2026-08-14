import { beforeEach, describe, expect, it } from 'vitest';
import { phases, useAppStore } from '../store/useAppStore';

describe('实验演示状态', () => {
  beforeEach(() => useAppStore.setState({ mode: 'backdoor', phaseIndex: 4, round: 18, running: true }));

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
});
