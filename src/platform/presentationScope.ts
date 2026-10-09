import type { Catalog, Group, Job, Library, RequestConfig } from './api';

export const DEMO_METHODS = ['fedavg', 'fedprox', 'heterogeneous_solution'] as const;
export const DEMO_DRAFT_KEY = 'federated-studio.draft.aircraft.v1';
export const DEMO_LAUNCH_KEY = 'federated-studio.launch.aircraft.v1';
// The full research catalog is the default. A focused demo is opt-in only.
export const aircraftDemoEnabled = () => import.meta.env.VITE_DEMO_SCOPE === 'aircraft';
const normalize = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, '');
const retiredGroup = (id: string) => id.toLowerCase().startsWith('domainnet_');

// Match the server's dataset/backbone metadata, never rename another dataset.
export function isAircraftVit(group: Group) {
  return normalize(group.dataset) === 'militaryaircraft3d' && normalize(group.backbone) === 'vit';
}
export function isPresentationGroup(group: Group) {
  return group.id.startsWith('uploaded_') || isAircraftVit(group);
}
export function presentationCatalog(catalog: Catalog, restricted = aircraftDemoEnabled()): Catalog {
  const groups = catalog.groups.filter(group => !retiredGroup(group.id));
  if (!restricted) return groups.length === catalog.groups.length ? catalog : { ...catalog, groups };
  return { ...catalog, groups: groups.filter(isPresentationGroup).map(group => ({
    ...group, methods: group.methods.filter(method => DEMO_METHODS.some(id => id === method.id)),
    augmentationSources: group.augmentationSources?.filter(source =>
      source.request.group === group.id && DEMO_METHODS.some(id => id === source.request.method)),
  })) };
}
export function requestInPresentation(request: Pick<RequestConfig, 'group' | 'method'>, catalog?: Catalog, restricted = aircraftDemoEnabled()) {
  if (retiredGroup(request.group)) return false;
  return !restricted || !!catalog?.groups.some(group => isPresentationGroup(group) && group.id === request.group &&
    DEMO_METHODS.some(id => id === request.method) && group.methods.some(method => method.id === request.method));
}
export function presentationLibrary(library: Library, catalog?: Catalog, restricted = aircraftDemoEnabled()): Library {
  return { models: library.models.filter(item => requestInPresentation(item, catalog, restricted)),
    testsets: library.testsets.filter(item => requestInPresentation(item, catalog, restricted)) };
}
export function presentationJobs(jobs: Job[], catalog?: Catalog, restricted = aircraftDemoEnabled()) {
  return jobs.filter(job => requestInPresentation(job.request, catalog, restricted));
}
