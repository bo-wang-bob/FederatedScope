import type { Catalog, RequestConfig } from '../platform/api';

export const designMethods = [
  { value: 'heterogeneous_solution', label: '本架构' },
  { value: 'fedavg', label: 'FedAvg' },
  { value: 'fedprox', label: 'FedProx' },
];

// Form-only fixture, not a report of server capabilities, cache availability or models.
// It never enters the live launcher and is not mixed with the live browser draft.
const defaults: RequestConfig = {
  group: 'officehome_vit', method: 'heterogeneous_solution', name: '',
  rounds: 30, clientCount: 60, sampleClients: 0, batchSize: 32, localEpochs: 1,
  learningRate: .0001, seed: 42, splitSeed: 42, alpha: .1, gpu: 0,
  evaluationFrequency: 1, samplesPerClient: 0, augmentationMode: 'generate',
  generatedPerSample: 1, generatedPerPrototype: 1, targetPerClass: 0, covarianceScale: 1,
};
export const designCatalog: Catalog = {
  host: '', address: '', protocol: 'design-preview', evaluationPolicy: 'not-connected',
  groups: [{
    id: defaults.group, dataset: 'Office-Home', backbone: 'vit', domains: 4,
    cacheFound: true, cacheFiles: 0, cacheBytes: 0, partitionLocked: false,
    methods: designMethods.map(method => ({ id: method.value, label: method.label,
      enabled: true, reason: null, defaults: { ...defaults, method: method.value,
        augmentationMode: method.value === 'heterogeneous_solution' ? 'generate' : 'none' } })),
  }],
};
