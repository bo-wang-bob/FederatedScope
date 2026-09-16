import { useEffect, useState } from 'react';
import { Alert, Button, Input, InputNumber, Select } from 'antd';
import { ArrowLeftOutlined, ArrowRightOutlined, CheckOutlined, ExperimentOutlined } from '@ant-design/icons';
import { api, methodLabel, type Catalog, type Job, type RequestConfig } from './api';
import { initialDraft, requestFromDraft, saveDraft, validateDraft, withPresentationDefaults, type Draft } from './draft';
import type { TrainingLaunch } from './launch';

const coreFields = ['name','group','method'];
const sharedFields: (keyof RequestConfig)[] = ['name','rounds','localEpochs','learningRate','batchSize','clientCount','sampleClients','samplesPerClient','seed','splitSeed','alpha','evaluationFrequency'];
export function augmentationConfigLabel(source: { request: RequestConfig }) {
  const request = source.request;
  const target = request.targetPerClass ? `每类 ${request.targetPerClass}` : '不限制每类数量';
  return `${target} · 样本 ${request.generatedPerSample || 0} / 原型 ${request.generatedPerPrototype || 0} · ${request.rounds} 轮`;
}
export function TrainingForm({ catalog, initialGroup, sourceId, running, disconnected, launch, open, preview = false, previewDraft, onPreviewDraft, draftKey }: {
  catalog: Catalog; initialGroup?: string; sourceId?: string; running?: Job; disconnected: boolean;
  launch: TrainingLaunch; open: (id: string) => void;
  preview?: boolean; previewDraft?: Draft; onPreviewDraft?: (draft: Draft) => void;
  draftKey?: string;
}) {
  const [draft, setDraft] = useState<Draft>(() => preview ? previewDraft || initialDraft(catalog, catalog.groups[0]?.id) : launch.intent?.request || initialDraft(catalog, initialGroup, draftKey));
  const [step, setStep] = useState(launch.intent ? 2 : 0), [errors, setErrors] = useState<Record<string,string>>({});
  const [sourceLoading, setSourceLoading] = useState(!preview && !!sourceId), [sourceError, setSourceError] = useState('');
  const [reload, setReload] = useState(0);
  const [presetId, setPresetId] = useState('');
  const group = catalog.groups.find(g => g.id === draft.group);
  const augmentationPresets = (group?.augmentationSources || [])
    .filter(source => source.request.method === 'heterogeneous_solution'
      && source.request.rounds === 100
      && source.request.generatedPerSample === 20
      && source.request.generatedPerPrototype === 20
      && source.request.covarianceScale === 0.01
      && [10,20,40,60].includes(source.request.targetPerClass || 0))
    .filter((source,index,sources) => index === sources.findIndex(candidate => candidate.request.targetPerClass === source.request.targetPerClass))
    .sort((a,b) => (a.request.targetPerClass || 0) - (b.request.targetPerClass || 0));
  const ours = draft.method === 'heterogeneous_solution', frozen = !!launch.intent || sourceLoading;
  useEffect(() => {
    if (preview) { onPreviewDraft?.(draft); return; }
    if (!sourceLoading) saveDraft(draft, draftKey);
  }, [draft, sourceLoading, preview, onPreviewDraft, draftKey]);
  useEffect(() => {
    if (preview || !sourceId) return;
    let alive = true; setSourceLoading(true); setSourceError('');
    void api<Job>('jobs/' + sourceId).then(job => {
      if (!['train','inspect'].includes(job.action)) throw new Error('该记录不包含训练配置');
      if (!catalog.groups.some(group => group.id === job.request.group && group.methods.some(method => method.id === job.request.method)))
        throw new Error('该记录不在当前演示配置范围');
      if (alive) { setDraft({ ...requestFromDraft(job.request), name: (job.request.name ? job.request.name + ' · 副本' : '').slice(0,120) }); setStep(0); setErrors({}); }
    }).catch(e => { if (alive) setSourceError(e.message); }).finally(() => { if (alive) setSourceLoading(false); });
    return () => { alive = false; };
  }, [sourceId, reload, preview]);
  const change = (field: keyof Draft, value: unknown) => {
    if (['generatedPerSample','generatedPerPrototype','targetPerClass','covarianceScale'].includes(String(field))) setPresetId('');
    setDraft(old => ({ ...old, [field]: value }));
    setErrors(old => { const next = { ...old }; delete next[field]; return next; });
  };
  const chooseGroup = (id: string) => {
    const next = catalog.groups.find(g => g.id === id);
    const defaults = next?.methods.find(m => m.id === draft.method && m.enabled)?.defaults || next?.methods.find(m => m.enabled)?.defaults;
    setPresetId(''); setDraft(withPresentationDefaults({ ...defaults, name: draft.name || '' })); setErrors({});
  };
  const chooseMethod = (id: string) => {
    const defaults = group?.methods.find(m => m.id === id)?.defaults;
    setPresetId(''); setDraft(withPresentationDefaults({ ...defaults, ...Object.fromEntries(sharedFields.filter(k => draft[k] !== undefined).map(k => [k,draft[k]])) })); setErrors({});
  };
  const validate = (all = false) => {
    const next = validateDraft(draft, catalog);
    const relevant = all || step > 0 ? next : Object.fromEntries(Object.entries(next).filter(([field]) => coreFields.includes(field)));
    setErrors(relevant);
    if (Object.keys(relevant).length) {
      requestAnimationFrame(() => document.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus());
      return false;
    }
    return true;
  };
  const number = (field: keyof RequestConfig, label: string, min: number, max: number, _hint?: string, locked = false, increment = 1) =>
    <div className="wizard-field" key={field}><label htmlFor={'train-' + field}>{label}</label>
      <InputNumber id={'train-' + field} aria-label={label} aria-invalid={!!errors[field]} aria-describedby={errors[field] ? 'error-' + field : undefined} status={errors[field] ? 'error' : undefined} value={draft[field] as number} min={min} max={max} step={increment} disabled={frozen || locked} onChange={value => change(field, value)} />
      {errors[field] && <small id={'error-' + field} role="alert" className="field-error">{errors[field]}</small>}</div>;
  const participantOptions = [{ value: 0, label: '全部' }, ...Array.from({ length: draft.clientCount || 0 }, (_,index) => ({ value: index + 1, label: String(index + 1) }))];
  const start = () => {
    if (frozen || running || disconnected || sourceError || !validate(true)) return;
    const request = requestFromDraft(draft);
    request.name = request.name?.trim() || (group?.dataset + ' · ' + methodLabel(request.method));
    launch.start(request);
  };
  return <div className="training-wizard studio-page-enter">
    <aside className="wizard-rail"><h2>实验配置</h2>
      <ol className="wizard-steps" aria-label="训练配置步骤">{['选择方案','训练设置','确认启动'].map((label,index) => <li key={label} className={step === index ? 'active' : step > index ? 'done' : ''}><button disabled={frozen || index > step} aria-current={step === index ? 'step' : undefined} onClick={() => { setStep(index); setErrors({}); }}><span>{step > index ? <CheckOutlined /> : '0' + (index + 1)}</span><strong>{label}</strong></button></li>)}</ol>
    </aside>
    <section className="wizard-panel" aria-label="训练配置">
      {sourceLoading && <Alert type="info" title="正在载入实验配置…" />}
      {sourceError && <Alert type="error" title={sourceError} action={<Button onClick={() => setReload(x => x + 1)}>重试</Button>} />}
      {running && !launch.intent && <Alert type="info" title="当前有任务运行中，可先配置下一次实验" action={<Button onClick={() => open(running.id)}>查看任务</Button>} />}
      <div key={step} className="wizard-page">
      {step === 0 && <><div className="wizard-title"><h2>数据与算法</h2></div><div className="wizard-fields">
        <div className="wizard-field"><label htmlFor="train-name">实验名称</label><Input id="train-name" value={draft.name} maxLength={120} placeholder="输入实验名称" disabled={frozen} onChange={event => change('name',event.target.value)} /></div>
        <div className="wizard-field"><label htmlFor="train-group">数据集</label><Select id="train-group" aria-label="数据集" aria-invalid={!!errors.group} value={draft.group} disabled={frozen} onChange={chooseGroup} options={catalog.groups.map(g => ({ value:g.id,label:g.dataset + ' / ' + g.backbone.toUpperCase() + (!g.cacheFound ? ' · 暂不可用' : ''),disabled:!g.cacheFound }))} />{errors.group && <small className="field-error" role="alert">{errors.group}</small>}</div>
        </div><div className="wizard-field"><label>训练算法</label><div className="algorithm-options" role="radiogroup" aria-label="训练算法">{group?.methods.map(method => <button key={method.id} type="button" role="radio" aria-checked={draft.method === method.id} disabled={frozen || !method.enabled} title={method.reason || undefined} className={draft.method === method.id ? 'selected' : ''} onClick={() => chooseMethod(method.id)}><span className="algorithm-glyph">{method.id === 'heterogeneous_solution' ? 'f' : methodLabel(method.id).slice(0,1)}</span><strong>{methodLabel(method.id)}</strong><i>{draft.method === method.id && <CheckOutlined />}</i></button>)}</div>{errors.method && <small className="field-error" role="alert">{errors.method}</small>}</div>
      </>}
      {step === 1 && <><div className="wizard-title"><h2>训练参数</h2></div>
        <div className="wizard-fields">{number('rounds','通信轮数',1,1000)}{number('localEpochs','本地轮数',1,100)}{number('learningRate','学习率',1e-8,1,undefined,false,.0001)}{number('batchSize','批大小',1,1024)}{number('clientCount','客户端数量',group?.domains || 1,240,undefined,group?.partitionLocked,group?.domains)}<div className="wizard-field"><label htmlFor="train-sampleClients">每轮参与客户端</label><Select id="train-sampleClients" aria-label="每轮参与客户端" value={draft.sampleClients ?? 0} disabled={frozen} onChange={value => change('sampleClients',value)} options={participantOptions} />{errors.sampleClients && <small role="alert" className="field-error">{errors.sampleClients}</small>}</div></div>
        {ours && <section className="augmentation-settings"><div className="wizard-section-title"><h3>数据增强</h3><span>本架构</span></div>
          {!!augmentationPresets.length && <div className="wizard-field"><label htmlFor="train-augmentation-preset">已验证增强配置</label><Select id="train-augmentation-preset" aria-label="已验证增强配置" allowClear placeholder="选择已跑配置" value={presetId || undefined} disabled={frozen} onChange={id => { setPresetId(id || ''); const source=augmentationPresets.find(item => item.id === id); if(source) setDraft(old => ({...source.request,name:old.name,augmentationMode:'generate',augmentationSourceId:'',allowLegacyAugmentation:false})); }} options={augmentationPresets.map(source => ({value:source.id,label:augmentationConfigLabel(source)}))} /></div>}
          <div className="wizard-fields">{number('generatedPerSample','每个样本生成数',0,1000)}{number('generatedPerPrototype','每个原型生成数',0,1000)}{number('targetPerClass','每类目标样本数',0,5000,'0 不设目标数量')}{number('covarianceScale','协方差缩放',0,10,undefined,false,.1)}</div>
        </section>}
      </>}
      {step === 2 && <><div className="wizard-title"><h2>{draft.name || '确认实验配置'}</h2></div>
        <div className="review-identity"><span className="review-identity-icon"><ExperimentOutlined /></span><div><h3>{methodLabel(draft.method)}</h3><p>{group?.dataset} / {group?.backbone.toUpperCase()}</p></div><span className="review-identity-type">联邦训练</span></div>
        <dl className="review-grid">{[['通信轮数',draft.rounds],['客户端',draft.clientCount],['每轮参与',draft.sampleClients || '全部'],['本地轮数',draft.localEpochs],['学习率',draft.learningRate],['批大小',draft.batchSize],...(ours ? [['每类目标样本',draft.targetPerClass]] : [])].map(([label,value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      </>}
      </div>
      {Object.keys(errors).length > 0 && step > 0 && <Alert type="error" showIcon title="请检查以下参数" description={<ul className="validation-errors">{Object.entries(errors).map(([field,message]) => <li key={field}>{message}</li>)}</ul>} />}
      <div className="wizard-actions"><Button type="text" icon={<ArrowLeftOutlined />} disabled={step === 0 || frozen} onClick={() => {setStep(value => value-1);setErrors({});}}>上一步</Button><span>{step+1} / 3</span>{step < 2 ? <Button type="primary" disabled={frozen || !!sourceError} onClick={() => {if(validate()) setStep(value => value+1);}}>下一步 <ArrowRightOutlined /></Button> : <Button type="primary" size="large" loading={launch.busy} disabled={!!launch.intent || !!running || disconnected || !!sourceError} onClick={start}>启动训练 <ArrowRightOutlined /></Button>}</div>
    </section>
  </div>;
}
