import { useEffect, useRef, useState } from 'react';
import { api, key, terminal, PlatformApiError, type Job, type RequestConfig } from './api';

export const LAUNCH_KEY = 'federated-studio.launch.v2';
export interface LaunchIntent {
  request: RequestConfig; preflightKey: string; trainKey: string; preflightId?: string;
  phase: 'checking' | 'starting' | 'recover' | 'blocked'; error?: string;
  trainRequested?: boolean; cancelRequested?: boolean;
}
function restore(storageKey: string): LaunchIntent | undefined {
  try {
    const value = JSON.parse(sessionStorage.getItem(storageKey) || 'null');
    if (value?.request?.group && value?.request?.method && typeof value.preflightKey === 'string' && typeof value.trainKey === 'string')
      return { ...value, phase: value.phase === 'blocked' ? 'blocked' : 'recover', error: value.error || '上次启动尚未确认完成' };
  } catch { /* Private or corrupt storage does not block the in-memory workflow. */ }
}
export function useTrainingLaunch(open: (id: string) => void, storageKey = LAUNCH_KEY) {
  const [intent, setIntent] = useState<LaunchIntent | undefined>(() => restore(storageKey));
  const current = useRef(intent), alive = useRef(true), active = useRef(false);
  const save = (next?: LaunchIntent) => {
    current.current = next;
    try { if (next) sessionStorage.setItem(storageKey, JSON.stringify(next)); else sessionStorage.removeItem(storageKey); } catch { /* Keep the same keys in memory. */ }
    if (alive.current) setIntent(next);
  };
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const execute = async (value: LaunchIntent) => {
    if (active.current) return;
    active.current = true;
    let next = { ...value, error: undefined };
    try {
      // Once a train request might have reached the server, only reconcile the same key.
      if (!next.trainRequested) {
        save({ ...next, phase: 'checking' });
        let check = next.preflightId ? await api<Job>('jobs/' + next.preflightId) :
          await api<Job>('preflight', { ...next.request, idempotencyKey: next.preflightKey });
        next = { ...next, preflightId: check.id, cancelRequested: current.current?.cancelRequested, phase: 'checking' };
        save(next);
        while (!terminal(check.status) && !current.current?.cancelRequested) {
          await new Promise(resolve => setTimeout(resolve, 900));
          if (!alive.current) return;
          check = await api<Job>('jobs/' + check.id);
        }
        if (current.current?.cancelRequested) {
          next.cancelRequested = true;
          save(next);
          if (!terminal(check.status)) {
            const stopped = await api<Job>('jobs/' + check.id + '/stop', {});
            if (!terminal(stopped.status) || stopped.cleanup?.ok === false) throw new Error(stopped.cleanup?.message || '取消尚未确认，请重试确认状态');
          }
          save(undefined); return;
        }
        if (!alive.current) return;
        if (check.status !== 'completed') {
          save({ ...next, phase: 'blocked', error: check.error || '检查未通过，请修改配置后重试' });
          return;
        }
      }
      next = { ...next, trainRequested: true, phase: 'starting', cancelRequested: false };
      save(next);
      const job = await api<Job>('train', { ...next.request, preflightId: next.preflightId, idempotencyKey: next.trainKey });
      save(undefined);
      if (alive.current) open(job.id);
    } catch (error) {
      // The backend returns PLATFORM_ERROR before saving a new job. Timeouts/5xx are uncertain.
      const rejected = error instanceof PlatformApiError && error.code === 'PLATFORM_ERROR' && error.status >= 400 && error.status < 500;
      save({ ...next, cancelRequested: current.current?.cancelRequested, phase: rejected && !current.current?.cancelRequested ? 'blocked' : 'recover', error: (error as Error).message });
    } finally { active.current = false; }
  };
  return {
    intent, busy: !!intent && ['checking', 'starting'].includes(intent.phase),
    start: (request: RequestConfig) => {
      if (active.current || current.current) return;
      void execute({ request: { ...request }, preflightKey: key(), trainKey: key(), phase: 'checking' });
    },
    resume: () => { if (current.current && !active.current) void execute(current.current); },
    cancel: () => {
      if (current.current && ['checking','recover'].includes(current.current.phase) && !current.current.trainRequested) {
        const next = { ...current.current, cancelRequested: true };
        save(next);
        if (!active.current) void execute(next);
      }
    },
    edit: () => { if (!active.current && current.current?.phase === 'blocked') save(undefined); },
  };
}
export type TrainingLaunch = ReturnType<typeof useTrainingLaunch>;
