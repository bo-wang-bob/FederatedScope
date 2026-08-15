import { create } from 'zustand';
import type {
  ExperimentMode,
  DataSourceKind,
  HierarchyPhase,
  TrainingConnectionState,
  TrainingEvent,
} from '../types';

const phases: HierarchyPhase[] = [
  '中央下发', '域内广播', '节点处理', '节点上传', '域内聚合', '域级上传', '全域聚合',
];

interface AppState {
  mode: ExperimentMode;
  phaseIndex: number;
  round: number;
  running: boolean;
  revealTruth: boolean;
  lastSequence: number;
  connectionState: TrainingConnectionState;
  dataSource: DataSourceKind;
  selectedNodeId?: string;
  setMode: (mode: ExperimentMode) => void;
  setPhaseIndex: (index: number) => void;
  nextPhase: () => void;
  toggleRunning: () => void;
  toggleTruth: () => void;
  selectNode: (nodeId?: string) => void;
  applyTrainingEvent: (event: TrainingEvent) => void;
  setConnectionState: (state: TrainingConnectionState) => void;
  setDataSource: (source: DataSourceKind) => void;
}

export { phases };

export const useAppStore = create<AppState>((set) => ({
  mode: 'backdoor',
  phaseIndex: 4,
  round: 18,
  running: true,
  revealTruth: true,
  lastSequence: 130,
  connectionState: 'connected',
  dataSource: 'frontend_simulation',
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
  applyTrainingEvent: (event) => set((state) => {
    if (event.sequence <= state.lastSequence || event.type !== 'stage.changed') return state;
    const payload = event.payload as { phaseIndex: number; round: number };
    return {
      phaseIndex: payload.phaseIndex,
      round: payload.round,
      lastSequence: event.sequence,
    };
  }),
  setConnectionState: (connectionState) => set({ connectionState }),
  setDataSource: (dataSource) => set({ dataSource }),
}));
