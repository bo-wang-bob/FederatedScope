import { create } from 'zustand';
import type { ExperimentMode, HierarchyPhase } from '../types';

const phases: HierarchyPhase[] = [
  '中央下发', '域内广播', '节点处理', '节点上传', '域内聚合', '域级上传', '全域聚合',
];

interface AppState {
  mode: ExperimentMode;
  phaseIndex: number;
  round: number;
  running: boolean;
  revealTruth: boolean;
  selectedNodeId?: string;
  setMode: (mode: ExperimentMode) => void;
  setPhaseIndex: (index: number) => void;
  nextPhase: () => void;
  toggleRunning: () => void;
  toggleTruth: () => void;
  selectNode: (nodeId?: string) => void;
}

export { phases };

export const useAppStore = create<AppState>((set) => ({
  mode: 'backdoor',
  phaseIndex: 4,
  round: 18,
  running: true,
  revealTruth: true,
  selectedNodeId: undefined,
  setMode: (mode) => set({ mode }),
  setPhaseIndex: (phaseIndex) => set({ phaseIndex }),
  nextPhase: () => set((state) => {
    const next = (state.phaseIndex + 1) % phases.length;
    return {
      phaseIndex: next,
      round: next === 0 ? state.round + 1 : state.round,
    };
  }),
  toggleRunning: () => set((state) => ({ running: !state.running })),
  toggleTruth: () => set((state) => ({ revealTruth: !state.revealTruth })),
  selectNode: (selectedNodeId) => set({ selectedNodeId }),
}));
