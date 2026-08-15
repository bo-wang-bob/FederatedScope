import type {
  TrainingConnectionState,
  TrainingEvent,
  TrainingSnapshot,
} from '../types';

export interface TrainingSubscription {
  close(): void;
}

export interface TrainingDataAdapter {
  readonly source: 'backend' | 'frontend_simulation';
  getSnapshot(experimentId: string, signal?: AbortSignal): Promise<TrainingSnapshot>;
  subscribe(
    experimentId: string,
    afterSequence: number,
    onEvent: (event: TrainingEvent) => void,
    onConnectionState?: (state: TrainingConnectionState) => void,
  ): TrainingSubscription;
}

export function createBackendTrainingAdapter(apiBaseUrl: string): TrainingDataAdapter {
  const baseUrl = apiBaseUrl.replace(/\/$/, '');
  const getSnapshot = async (experimentId: string, signal?: AbortSignal) => {
    const response = await fetch(`${baseUrl}/api/experiments/${encodeURIComponent(experimentId)}/snapshot`, { signal });
    if (!response.ok) throw new Error(`训练快照获取失败：HTTP ${response.status}`);
    const payload = await response.json() as { data: TrainingSnapshot };
    return { ...payload.data, source: 'backend' as const };
  };

  return {
    source: 'backend',
    getSnapshot,
    subscribe(experimentId, afterSequence, onEvent, onConnectionState) {
      let stream: EventSource | undefined;
      let closed = false;
      let lastSequence = afterSequence;
      let reconnectTimer: number | undefined;
      let retryCount = 0;

      const connect = () => {
        if (closed) return;
        onConnectionState?.(retryCount === 0 ? 'connecting' : 'recovering');
        const query = new URLSearchParams({ afterSequence: String(lastSequence) });
        stream = new EventSource(`${baseUrl}/api/experiments/${encodeURIComponent(experimentId)}/events?${query}`);
        stream.onopen = () => {
          retryCount = 0;
          onConnectionState?.('connected');
        };
        stream.onmessage = (message) => {
          const event = JSON.parse(message.data) as TrainingEvent;
          if (event.sequence <= lastSequence) return;
          lastSequence = event.sequence;
          onEvent({ ...event, source: 'backend' });
        };
        stream.onerror = () => {
          stream?.close();
          if (closed) return;
          onConnectionState?.('recovering');
          const controller = new AbortController();
          getSnapshot(experimentId, controller.signal)
            .then((snapshot) => {
              if (closed || snapshot.sequence < lastSequence) return;
              lastSequence = snapshot.sequence;
              onEvent({
                id: `snapshot-${snapshot.sequence}`,
                sequence: snapshot.sequence,
                experimentId,
                type: 'stage.changed',
                timestamp: snapshot.updatedAt,
                source: 'backend',
                payload: { phaseIndex: snapshot.phaseIndex, round: snapshot.round },
              });
            })
            .catch(() => undefined)
            .finally(() => {
              retryCount += 1;
              const delay = Math.min(8_000, 500 * 2 ** retryCount);
              reconnectTimer = window.setTimeout(connect, delay);
            });
        };
      };

      connect();
      return {
        close() {
          closed = true;
          stream?.close();
          if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
          onConnectionState?.('disconnected');
        },
      };
    },
  };
}

export function createDemoTrainingAdapter(
  initialSnapshot: TrainingSnapshot,
  intervalMs = 3_200,
): TrainingDataAdapter {
  return {
    source: 'frontend_simulation',
    async getSnapshot() {
      return initialSnapshot;
    },
    subscribe(experimentId, afterSequence, onEvent, onConnectionState) {
      let sequence = Math.max(afterSequence, initialSnapshot.sequence);
      let phaseIndex = initialSnapshot.phaseIndex;
      let round = initialSnapshot.round;
      onConnectionState?.('connected');
      const timer = window.setInterval(() => {
        phaseIndex = (phaseIndex + 1) % 7;
        if (phaseIndex === 0) round += 1;
        sequence += 1;
        onEvent({
          id: `demo-${sequence}`,
          sequence,
          experimentId,
          type: 'stage.changed',
          timestamp: new Date().toISOString(),
          source: 'frontend_simulation',
          payload: { phaseIndex, round },
        });
      }, intervalMs);
      return {
        close() {
          window.clearInterval(timer);
          onConnectionState?.('disconnected');
        },
      };
    },
  };
}

export function resolveTrainingDataAdapter(initialSnapshot: TrainingSnapshot): TrainingDataAdapter {
  const requestedSource = import.meta.env.VITE_DATA_SOURCE;
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
  return requestedSource === 'backend' && apiBaseUrl
    ? createBackendTrainingAdapter(apiBaseUrl)
    : createDemoTrainingAdapter(initialSnapshot);
}
