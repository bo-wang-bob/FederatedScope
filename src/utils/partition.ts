import type {
  ClientPartitionPreview,
  DomainKey,
  DomainPartitionPreview,
  ScenarioPartitionPreview,
} from '../types';

export const officeHomeClassNames = Array.from(
  { length: 65 },
  (_, index) => `类别 ${String(index + 1).padStart(2, '0')}`,
);

export const officeHomeDomainTotals: Record<DomainKey, number> = {
  Art: 1_698,
  Clipart: 3_055,
  Product: 3_107,
  Real_World: 3_049,
};

const domainOrder: DomainKey[] = ['Art', 'Clipart', 'Product', 'Real_World'];
const clientPrefixes: Record<DomainKey, string> = {
  Art: 'OH-DT',
  Clipart: 'OH-TS',
  Product: 'OH-ED',
  Real_World: 'OH-FR',
};

function createRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function normal(random: () => number): number {
  const u = Math.max(random(), Number.EPSILON);
  const v = Math.max(random(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function gamma(shape: number, random: () => number): number {
  if (shape < 1) {
    return gamma(shape + 1, random) * Math.pow(Math.max(random(), Number.EPSILON), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    const x = normal(random);
    const v = Math.pow(1 + c * x, 3);
    if (v <= 0) continue;
    const u = random();
    if (u < 1 - 0.0331 * Math.pow(x, 4)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function allocateInteger(total: number, weights: number[]): number[] {
  const weightTotal = weights.reduce((sum, value) => sum + value, 0);
  const exact = weights.map((weight) => (weight / weightTotal) * total);
  const allocated = exact.map(Math.floor);
  let remainder = total - allocated.reduce((sum, value) => sum + value, 0);
  const order = exact
    .map((value, index) => ({ index, remainder: value - Math.floor(value) }))
    .sort((a, b) => b.remainder - a.remainder || a.index - b.index);
  for (let index = 0; index < remainder; index += 1) {
    allocated[order[index].index] += 1;
  }
  return allocated;
}

function ensureNonEmpty(histograms: number[][]) {
  histograms.forEach((histogram, emptyIndex) => {
    if (histogram.some((value) => value > 0)) return;
    let donorIndex = 0;
    let donorClass = 0;
    let donorValue = 0;
    histograms.forEach((candidate, candidateIndex) => {
      candidate.forEach((value, classIndex) => {
        if (value > donorValue) {
          donorIndex = candidateIndex;
          donorClass = classIndex;
          donorValue = value;
        }
      });
    });
    if (donorValue > 1) {
      histograms[donorIndex][donorClass] -= 1;
      histograms[emptyIndex][donorClass] += 1;
    }
  });
}

function summarizeClient(
  domainKey: DomainKey,
  domainTotal: number,
  clientIndex: number,
  classHistogram: number[],
): ClientPartitionPreview {
  const sampleCount = classHistogram.reduce((sum, value) => sum + value, 0);
  const classProportions = classHistogram.map((value) => value / sampleCount);
  const coveredClassCount = classHistogram.filter((value) => value > 0).length;
  const dominantClassRatio = Math.max(...classProportions);
  const dominantClassIndex = classProportions.indexOf(dominantClassRatio);
  const entropy = -classProportions.reduce(
    (sum, value) => value > 0 ? sum + value * Math.log(value) : sum,
    0,
  );
  return {
    clientId: `${clientPrefixes[domainKey]}-C${String(clientIndex + 1).padStart(2, '0')}`,
    domainKey,
    sampleCount,
    domainSampleRatio: sampleCount / domainTotal,
    classHistogram,
    classProportions,
    coveredClassCount,
    missingClassCount: classHistogram.length - coveredClassCount,
    dominantClassIndex,
    dominantClassRatio,
    labelEntropy: entropy,
  };
}

function buildDomainPartition(
  domainKey: DomainKey,
  alpha: number,
  seed: number,
): DomainPartitionPreview {
  const domainIndex = domainOrder.indexOf(domainKey);
  const random = createRandom(seed + (domainIndex + 1) * 104_729);
  const totalSamples = officeHomeDomainTotals[domainKey];
  const classWeights = officeHomeClassNames.map((_, classIndex) =>
    0.65 + ((classIndex * 17 + domainIndex * 11) % 19) / 20 + random() * 0.3,
  );
  const classTotals = allocateInteger(totalSamples, classWeights);
  const histograms = Array.from({ length: 15 }, () => Array(officeHomeClassNames.length).fill(0));

  classTotals.forEach((classTotal, classIndex) => {
    const weights = Array.from({ length: 15 }, () => gamma(alpha, random));
    allocateInteger(classTotal, weights).forEach((value, clientIndex) => {
      histograms[clientIndex][classIndex] = value;
    });
  });
  ensureNonEmpty(histograms);

  return {
    domainKey,
    totalSamples,
    classCount: officeHomeClassNames.length,
    clients: histograms.map((histogram, clientIndex) =>
      summarizeClient(domainKey, totalSamples, clientIndex, histogram),
    ),
  };
}

export function generateScenarioPartition(
  alpha = 0.3,
  seed = 20_260_815,
): ScenarioPartitionPreview {
  if (!Number.isFinite(alpha) || alpha < 0.05 || alpha > 10) {
    throw new RangeError('alpha 必须位于 0.05 到 10 之间');
  }
  return {
    datasetKey: 'office-home',
    alpha,
    seed,
    partitionVersion: `office-home-a${alpha.toFixed(2)}-s${seed}`,
    source: 'frontend_simulation',
    basis: 'built_in_simulation',
    domains: domainOrder.map((domainKey) => buildDomainPartition(domainKey, alpha, seed)),
  };
}

export function distributionSummary(preview: ScenarioPartitionPreview) {
  const clients = preview.domains.flatMap((domain) => domain.clients);
  const sampleCounts = clients.map((client) => client.sampleCount);
  return {
    minSamples: Math.min(...sampleCounts),
    maxSamples: Math.max(...sampleCounts),
    meanCoverage: clients.reduce((sum, client) => sum + client.coveredClassCount, 0) / clients.length,
    maxDominantRatio: Math.max(...clients.map((client) => client.dominantClassRatio)),
    meanEntropy: clients.reduce((sum, client) => sum + client.labelEntropy, 0) / clients.length,
  };
}
