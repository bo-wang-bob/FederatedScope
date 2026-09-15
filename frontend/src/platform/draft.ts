import type { Catalog, RequestConfig } from './api';
export const DRAFT_KEY = 'federated-studio.draft.v2';
export type Draft = Partial<RequestConfig>;
const numeric: [keyof RequestConfig, string, number, number, boolean][] = [
  ['rounds','通信轮数',1,1000,true], ['localEpochs','本地轮数',1,100,true],
  ['learningRate','学习率',1e-8,1,false], ['batchSize','批大小',1,1024,true],
  ['clientCount','客户端数量',1,240,true], ['sampleClients','每轮参与客户端',0,240,true],
  ['samplesPerClient','每客户端样本数',0,100000,true], ['seed','随机种子',0,2147483647,true],
  ['splitSeed','划分种子',0,2147483647,true], ['alpha','异构强度 α',.0001,100,false],
  ['evaluationFrequency','评测间隔',1,1000,true],
];
export function validateDraft(draft: Draft, catalog: Catalog): Record<string, string> {
  const errors: Record<string,string> = {};
  const group = catalog.groups.find(g => g.id === draft.group);
  if (!group || !group.cacheFound) errors.group = '该数据尚未就绪，请选择其他数据集';
  if (!group?.methods.some(m => m.id === draft.method && m.enabled)) errors.method = '请选择可用算法';
  if ((draft.name || '').length > 120) errors.name = '名称不能超过 120 字';
  for (const [field,label,min,max,integer] of numeric) {
    const value = draft[field];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || integer && !Number.isInteger(value))
      errors[field] = label + '应为 ' + min + '–' + max + (integer ? ' 的整数' : ' 之间的数值');
  }
  if (group && typeof draft.clientCount === 'number' && draft.clientCount % group.domains) errors.clientCount = '客户端数量须为 ' + group.domains + ' 的倍数';
  if ((draft.sampleClients || 0) > (draft.clientCount || 0)) errors.sampleClients = '不能超过客户端总数';
  if (draft.method === 'heterogeneous_solution') {
    for (const field of ['generatedPerSample','generatedPerPrototype','targetPerClass','covarianceScale'] as const) {
      const max = field === 'targetPerClass' ? 5000 : field === 'covarianceScale' ? 10 : 1000;
      const value = draft[field];
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > max || field !== 'covarianceScale' && !Number.isInteger(value))
        errors[field] = '请输入 0–' + max + (field === 'covarianceScale' ? ' 的数值' : ' 的整数');
    }
    if (!['generate','reuse'].includes(draft.augmentationMode || '')) errors.augmentationMode = '请选择增强方式';
    if (draft.augmentationMode === 'reuse' && !draft.augmentationSourceId && !draft.allowLegacyAugmentation)
      errors.allowLegacyAugmentation = '历史生成结果需要确认来源限制';
  }
  return errors;
}
export function initialDraft(catalog: Catalog, groupId?: string): Draft {
  const group = catalog.groups.find(g => g.id === groupId) || catalog.groups.find(g => g.cacheFound) || catalog.groups[0];
  const fallback = group?.methods.find(m => m.enabled)?.defaults;
  if (!groupId) {
    try {
      const saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null');
      if (saved?.version === 2) {
        const savedGroup = catalog.groups.find(g => g.id === saved.request?.group);
        if (savedGroup) {
          const base = savedGroup.methods.find(m => m.id === saved.request.method)?.defaults || savedGroup.methods.find(m => m.enabled)?.defaults;
          const safe = Object.fromEntries(Object.entries(requestFromDraft(saved.request)).filter(([,value]) => ['string','number','boolean'].includes(typeof value)));
          return { ...base, ...safe };
        }
      }
    } catch { /* Corrupt or unavailable storage does not block creating a new experiment. */ }
  }
  return { ...fallback };
}
export function saveDraft(draft: Draft): boolean {
  try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ version: 2, request: draft })); return true; } catch { return false; }
}
export function requestFromDraft(draft: Draft): RequestConfig {
  const fields = ['group','method','name','rounds','localEpochs','learningRate','batchSize','clientCount','sampleClients','samplesPerClient','seed','splitSeed','alpha','gpu','evaluationFrequency','augmentationMode','augmentationSourceId','allowLegacyAugmentation','generatedPerSample','generatedPerPrototype','targetPerClass','covarianceScale'];
  return Object.fromEntries(fields.filter(field => draft[field as keyof Draft] !== undefined).map(field => [field,draft[field as keyof Draft]])) as unknown as RequestConfig;
}
