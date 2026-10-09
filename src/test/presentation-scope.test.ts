import { beforeEach, expect, it, vi } from 'vitest';
import type { Catalog, Group, Job, Library, RequestConfig } from '../platform/api';
import { aircraftDemoEnabled, DEMO_DRAFT_KEY, presentationCatalog, presentationJobs, presentationLibrary, requestInPresentation } from '../platform/presentationScope';
import { DRAFT_KEY, initialDraft, saveDraft } from '../platform/draft';

beforeEach(() => { vi.stubEnv('VITE_DEMO_SCOPE', 'aircraft'); localStorage.clear(); });
const request = { group: 'military_vit', method: 'fedavg', name: '军机训练' } as RequestConfig;
const group = { id: 'military_vit', dataset: 'MilitaryAircraft-3D', backbone: 'vit', cacheFound: true,
  methods: ['fedavg','fedprox','heterogeneous_solution','fedopt'].map(id => ({id, label:id, enabled:true, reason:null, defaults:{...request,method:id}})) } as Group;
const office = {...group,id:'officehome_vit',dataset:'Office-Home', methods: group.methods.map(method => ({...method,defaults:{...method.defaults,group:'officehome_vit'}}))};
const catalog = { groups: [group, {...group,id:'military_cnn',backbone:'cnn'}, office, {...office,id:'officehome_cnn',backbone:'cnn'}] } as Catalog;
const model = { id:'plane-final',jobId:'plane',name:'军机模型',group:'military_vit',method:'fedavg',featureSpace:'plane' };
const library = {models:[model,{...model,id:'other',group:'officehome_vit'}, {...model,id:'fedopt',method:'fedopt'}],
  testsets:[{...model,id:'plane'},{...model,id:'office',group:'officehome_vit'}]} as Library;

it('defaults to the full catalog and keeps every existing dataset and saved artifact', () => {
  vi.stubEnv('VITE_DEMO_SCOPE', undefined);
  expect(aircraftDemoEnabled()).toBe(false);
  expect(presentationCatalog(catalog)).toBe(catalog);
  expect(presentationLibrary(library,catalog)).toEqual(library);
  expect(requestInPresentation({group:'officehome_vit',method:'fedavg'},catalog)).toBe(true);
  vi.stubEnv('VITE_DEMO_SCOPE', 'all');
  expect(aircraftDemoEnabled()).toBe(false);
  expect(presentationCatalog(catalog).groups.map(g=>g.id)).toEqual(['military_vit','military_cnn','officehome_vit','officehome_cnn']);
  expect(presentationLibrary(library,catalog).models.map(m=>m.id)).toEqual(['plane-final','other','fedopt']);
});
it('adds third-party groups without losing uploaded datasets, methods, histories or the military default', () => {
  vi.stubEnv('VITE_DEMO_SCOPE', undefined);
  const ids=['digit3_vit','digit3_cnn','mdsent_rnn','mdsent_lstm','uploaded_existing'];
  const expanded={...catalog,groups:[...ids.map(id=>({...group,id})),...catalog.groups]};
  expect(presentationCatalog(expanded).groups).toEqual(expanded.groups);
  const jobs=expanded.groups.map(g=>({id:g.id,request:{...request,group:g.id}})) as Job[];
  expect(presentationJobs(jobs,expanded)).toEqual(jobs);
  expect(initialDraft(expanded).group).toBe('military_vit');
  saveDraft({...request,group:'uploaded_existing'});
  expect(initialDraft(expanded).group).toBe('uploaded_existing');
});
it('removes retired DomainNet from stale catalogs and artifacts without hiding other datasets', () => {
  vi.stubEnv('VITE_DEMO_SCOPE', undefined);
  const retired = ['domainnet_vit','domainnet_cnn','domainnet_mixer'];
  const expanded = {...catalog,groups:[...catalog.groups,...retired.map(id=>({...group,id,dataset:'DomainNet'}))]};
  expect(presentationCatalog(expanded).groups).toEqual(catalog.groups);
  expect(expanded.groups).toHaveLength(catalog.groups.length + 3);
  const retiredItems = retired.map(id=>({...model,id,group:id}));
  expect(presentationLibrary({models:[...library.models,...retiredItems],testsets:[...library.testsets,...retiredItems]} as Library,expanded)).toEqual(library);
  const jobs = [...catalog.groups,...expanded.groups.slice(-3)].map(g=>({id:g.id,request:{...request,group:g.id}})) as Job[];
  expect(presentationJobs(jobs,expanded).map(job=>job.id)).toEqual(catalog.groups.map(g=>g.id));
  saveDraft({...request,group:'domainnet_vit'});
  expect(initialDraft(presentationCatalog(expanded)).group).toBe('military_vit');
});
it('keeps military ViT with three methods without mutation', () => {
  const filtered=presentationCatalog(catalog);
  expect(filtered.groups.map(g=>g.id)).toEqual(['military_vit']);
  for (const entry of filtered.groups) {
    expect(entry.methods.map(m=>m.id)).toEqual(['fedavg','fedprox','heterogeneous_solution']);
    expect(entry.methods.every(m=>m.defaults.group===entry.id)).toBe(true);
  }
  expect(catalog.groups).toHaveLength(4);
  expect(catalog.groups[0].methods).toHaveLength(4);
});
it('never relabels or substitutes another dataset when the military group is missing', () => {
  const missing={...catalog,groups:catalog.groups.filter(g=>g.id!=='military_vit')};
  expect(presentationCatalog(missing).groups.map(g=>g.id)).toEqual([]);
  expect(presentationLibrary(library,missing).models.map(m=>m.id)).toEqual([]);
  expect(presentationCatalog({...catalog,groups:catalog.groups.filter(g=>g.backbone==='cnn')}).groups).toEqual([]);
  expect(presentationLibrary(library)).toEqual({models:[],testsets:[]});
});
it('filters model and testset inventories and history with the same allowlist', () => {
  expect(presentationLibrary(library,catalog).models.map(m=>m.id)).toEqual(['plane-final']);
  expect(presentationLibrary(library,catalog).testsets.map(t=>t.id)).toEqual(['plane']);
  const jobs=[{id:'plane',request},{id:'office',request:{...request,group:'officehome_vit'}}, {id:'hidden-method',request:{...request,method:'fedopt'}}] as Job[];
  expect(presentationJobs(jobs,catalog).map(j=>j.id)).toEqual(['plane']);
  expect(requestInPresentation(jobs[1].request,catalog)).toBe(false);
  expect(requestInPresentation({group:'officehome_vit',method:'fedopt'},catalog)).toBe(false);
});
it('does not restore a hidden method from the demo draft or overwrite the full-view draft', () => {
  const original=JSON.stringify({version:2,request:{...request,group:'officehome_vit',name:'保留草稿'}});
  localStorage.setItem(DRAFT_KEY,original);
  saveDraft({...request,method:'fedopt'},DEMO_DRAFT_KEY);
  expect(initialDraft(presentationCatalog(catalog),undefined,DEMO_DRAFT_KEY).method).toBe('fedavg');
  saveDraft(request,DEMO_DRAFT_KEY);
  expect(localStorage.getItem(DRAFT_KEY)).toBe(original);
});
