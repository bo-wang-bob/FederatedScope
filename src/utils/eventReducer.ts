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
        [payload.clientId]: { ...base.clients[payload.clientId], ...payload, source: event.source },
      },
    };
  }
  if (event.type === 'client.metric.updated') {
    const payload = event.payload as unknown as ClientStatusPayload & Record<string, number>;
    return {
      ...base,
      clients: {
        ...base.clients,
        [payload.clientId]: { ...base.clients[payload.clientId], ...payload, source: event.source },
      },
    };
  }
  if (event.type === 'topology.status.changed') {
    const payload = event.payload as {
      node: string; label: string; status: string; ready: boolean;
    };
    return {
      ...base,
      topology: {
        ...(base.topology ?? {}),
        [payload.node]: { ...payload, source: event.source },
      },
    };
  }
  if (event.type === 'defense.decision') {
    const payload = event.payload as {
      round?: number;
      droppedClientIds?: Array<string | number>;
      truePositiveRate?: number;
      falsePositiveRate?: number;
    };
    const clients = { ...base.clients };
    const displayId = (value: string | number) => {
      if (typeof value === 'string' && value.startsWith('OH-')) return value;
      const index = Math.max(0, Number(value) - 1);
      const prefixes = ['OH-DT', 'OH-TS', 'OH-ED', 'OH-FR'];
      return `${prefixes[Math.min(3, Math.floor(index / 15))]}-C${String(index % 15 + 1).padStart(2, '0')}`;
    };
    for (const rawId of payload.droppedClientIds ?? []) {
      const clientId = displayId(rawId);
      const current = clients[clientId];
      if (current) clients[clientId] = {
        ...current, status: '已过滤', assessment: '过滤', progress: 100,
        round: payload.round ?? current.round, source: event.source,
      };
    }
    const metrics = [...(base.metrics ?? [])];
    if (payload.truePositiveRate !== undefined || payload.falsePositiveRate !== undefined) {
      const metric = {
        round: payload.round ?? base.round,
        truePositiveRate: payload.truePositiveRate,
        falsePositiveRate: payload.falsePositiveRate,
      };
      const existing = metrics.findIndex((item) => item.round === metric.round);
      if (existing >= 0) metrics[existing] = { ...metrics[existing], ...metric };
      else metrics.push(metric);
    }
    return { ...base, clients, metrics };
  }
  if (event.type === 'metric.updated') {
    const metric = event.payload as import('../types').ExperimentMetricPoint;
    const metrics = [...(base.metrics ?? [])];
    const existing = metrics.findIndex((item) => item.round === metric.round);
    if (existing >= 0) metrics[existing] = { ...metrics[existing], ...metric };
    else metrics.push(metric);
    return { ...base, round: Math.max(base.round, metric.round ?? 0), metrics };
  }
  if (event.type === 'experiment.stopping') return { ...base, status: 'stopping' };
  if (event.type === 'experiment.stopped') return { ...base, status: 'stopped' };
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
