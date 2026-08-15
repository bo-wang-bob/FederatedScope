import type { TrainingEvent, TrainingSnapshot } from '../types';

type StagePayload = { phaseIndex: number; round: number };
type ClientStatusPayload = {
  clientId: string;
  domainKey: import('../types').DomainKey;
  status: import('../types').NodeStatus;
  progress: number;
  round: number;
};

export function mergeTrainingEvent(
  snapshot: TrainingSnapshot,
  event: TrainingEvent,
): TrainingSnapshot {
  if (event.experimentId !== snapshot.experimentId || event.sequence <= snapshot.sequence) {
    return snapshot;
  }

  const base: TrainingSnapshot = {
    ...snapshot,
    sequence: event.sequence,
    source: event.source,
    updatedAt: event.timestamp,
  };

  if (event.type === 'stage.changed' || event.type === 'round.started') {
    const payload = event.payload as unknown as StagePayload;
    return { ...base, phaseIndex: payload.phaseIndex, round: payload.round, status: 'running' };
  }
  if (event.type === 'client.status.changed') {
    const payload = event.payload as unknown as ClientStatusPayload;
    return {
      ...base,
      clients: {
        ...base.clients,
        [payload.clientId]: { ...payload, source: event.source },
      },
    };
  }
  if (event.type === 'experiment.completed') return { ...base, status: 'completed' };
  if (event.type === 'experiment.failed') return { ...base, status: 'failed' };
  return base;
}

export function recoverTrainingSnapshot(
  current: TrainingSnapshot,
  recovered: TrainingSnapshot,
): TrainingSnapshot {
  if (recovered.experimentId !== current.experimentId || recovered.sequence < current.sequence) {
    return current;
  }
  return recovered;
}
