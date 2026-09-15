import type { ExperimentConfig } from '../../types';

export function serializeExperimentConfig(config: ExperimentConfig): string {
  return `${JSON.stringify(config, null, 2)}\n`;
}

export function exportExperimentConfig(config: ExperimentConfig) {
  const blob = new Blob([serializeExperimentConfig(config)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `experiment-config-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}
