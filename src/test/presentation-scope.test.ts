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

it('defaults to the aircraft demo and permits an explicit full-catalog build', () => {
  vi.stubEnv('VITE_DEMO_SCOPE', undefined);
  expect(aircraftDemoEnabled()).toBe(true);
  vi.stubEnv('VITE_DEMO_SCOPE', 'all');
  expect(aircraftDemoEnabled()).toBe(false);
  expect(presentationCatalog(catalog).groups.map(g=>g.id)).toEqual(['military_vit','military_cnn']);
  expect(presentationLibrary(library,catalog).models.map(m=>m.id)).toEqual(['plane-final','fedopt']);
  expect(requestInPresentation({group:'officehome_vit',method:'fedavg'},catalog)).toBe(false);
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
