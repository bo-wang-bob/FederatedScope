import { useEffect, useState } from 'react';
import { Alert, Button, Checkbox, Collapse, Input, InputNumber, Select, Tooltip } from 'antd';
import { ArrowLeftOutlined, ArrowRightOutlined, CheckOutlined, ExperimentOutlined, ReloadOutlined } from '@ant-design/icons';
import { api, methodLabel, type Catalog, type Job, type RequestConfig } from './api';
import { initialDraft, requestFromDraft, saveDraft, validateDraft, type Draft } from './draft';
import type { TrainingLaunch } from './launch';

const coreFields = ['name','group','method'];
const sharedFields: (keyof RequestConfig)[] = ['name','rounds','localEpochs','learningRate','batchSize','clientCount','sampleClients','samplesPerClient','seed','splitSeed','alpha','evaluationFrequency'];
export function TrainingForm({ catalog, initialGroup, sourceId, running, disconnected, launch, open }: {
  catalog: Catalog; initialGroup?: string; sourceId?: string; running?: Job; disconnected: boolean;
  launch: TrainingLaunch; open: (id: string) => void;
}) {
  const [draft, setDraft] = useState<Draft>(() => launch.intent?.request || initialDraft(catalog, initialGroup));
  const [step, setStep] = useState(launch.intent ? 2 : 0), [errors, setErrors] = useState<Record<string,string>>({});
  const [sourceLoading, setSourceLoading] = useState(!!sourceId), [sourceError, setSourceError] = useState('');
  const [saved, setSaved] = useState(true), [reload, setReload] = useState(0);
  const group = catalog.groups.find(g => g.id === draft.group);
  const ours = draft.method === 'heterogeneous_solution', frozen = !!launch.intent || sourceLoading;
  useEffect(() => { if (!sourceLoading) setSaved(saveDraft(draft)); }, [draft, sourceLoading]);
  useEffect(() => {
    if (!sourceId) return;
    let alive = true; setSourceLoading(true); setSourceError('');
    void api<Job>('jobs/' + sourceId).then(job => {
      if (!['train','inspect'].includes(job.action)) throw new Error('该记录不包含训练配置');
      if (alive) { setDraft({ ...requestFromDraft(job.request), name: (job.request.name ? job.request.name + ' · 副本' : '').slice(0,120) }); setStep(0); setErrors({}); }
    }).catch(e => { if (alive) setSourceError(e.message); }).finally(() => { if (alive) setSourceLoading(false); });
    return () => { alive = false; };
  }, [sourceId, reload]);
  const change = (field: keyof Draft, value: unknown) => {
    setDraft(old => ({ ...old, [field]: value }));
    setErrors(old => { const next = { ...old }; delete next[field]; return next; });
  };
  const chooseGroup = (id: string) => {
    const next = catalog.groups.find(g => g.id === id);
    const defaults = next?.methods.find(m => m.id === draft.method && m.enabled)?.defaults || next?.methods.find(m => m.enabled)?.defaults;
    setDraft({ ...defaults, name: draft.name || '' }); setErrors({});
  };
  const chooseMethod = (id: string) => {
    const defaults = group?.methods.find(m => m.id === id)?.defaults;
    setDraft({ ...defaults, ...Object.fromEntries(sharedFields.filter(k => draft[k] !== undefined).map(k => [k,draft[k]])) }); setErrors({});
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
  const number = (field: keyof RequestConfig, label: string, min: number, max: number, hint?: string, locked = false, increment = 1) =>
    <div className="wizard-field" key={field}><label htmlFor={'train-' + field}>{label}{hint && <Tooltip title={hint}><span className="field-help" aria-label={hint}>?</span></Tooltip>}</label>
      <InputNumber id={'train-' + field} aria-label={label} aria-invalid={!!errors[field]} aria-describedby={errors[field] ? 'error-' + field : undefined} status={errors[field] ? 'error' : undefined} value={draft[field] as number} min={min} max={max} step={increment} disabled={frozen || locked} onChange={value => change(field, value)} />
      {errors[field] && <small id={'error-' + field} role="alert" className="field-error">{errors[field]}</small>}</div>;
  const start = () => {
    if (frozen || running || disconnected || sourceError || !validate(true)) return;
    const request = requestFromDraft(draft);
    request.name = request.name?.trim() || (group?.dataset + ' · ' + methodLabel(request.method));
    launch.start(request);
  };
  return <div className="training-wizard studio-page-enter">
    <aside className="wizard-rail"><span className="studio-kicker">NEW EXPERIMENT</span><h2>构建你的<br />下一次实验<span>.</span></h2>
      <ol className="wizard-steps" aria-label="训练配置步骤">{['选择方案','训练设置','确认启动'].map((label,index) => <li key={label} className={step === index ? 'active' : step > index ? 'done' : ''}><button disabled={frozen || index > step} aria-current={step === index ? 'step' : undefined} onClick={() => { setStep(index); setErrors({}); }}><span>{step > index ? <CheckOutlined /> : '0' + (index + 1)}</span><strong>{label}</strong></button></li>)}</ol>
      <div className="wizard-draft-status"><i />{saved ? '配置自动保存在此浏览器' : '浏览器未允许保存草稿'}</div>
      <Button type="text" size="small" icon={<ReloadOutlined />} disabled={frozen} onClick={() => { setDraft(initialDraft(catalog, draft.group)); setStep(0); setErrors({}); }}>恢复默认</Button>
    </aside>
    <section className="wizard-panel" aria-label="训练配置">
      {sourceLoading && <Alert type="info" title="正在载入实验配置…" />}
      {sourceError && <Alert type="error" title={sourceError} action={<Button onClick={() => setReload(x => x + 1)}>重试</Button>} />}
      {running && !launch.intent && <Alert type="info" title="当前有任务运行中，可先配置下一次实验" action={<Button onClick={() => open(running.id)}>查看任务</Button>} />}
      <div key={step} className="wizard-page">
      {step === 0 && <><div className="wizard-title"><span>01 / SETUP</span><h2>从数据与算法开始</h2></div><div className="wizard-fields">
        <div className="wizard-field"><label htmlFor="train-name">实验名称<span className="field-optional">可选</span></label><Input id="train-name" value={draft.name} maxLength={120} placeholder="为这次实验命名" disabled={frozen} onChange={event => change('name',event.target.value)} /></div>
        <div className="wizard-field"><label htmlFor="train-group">数据集</label><Select id="train-group" aria-label="数据集" aria-invalid={!!errors.group} value={draft.group} disabled={frozen} onChange={chooseGroup} options={catalog.groups.map(g => ({ value:g.id,label:g.dataset + ' / ' + g.backbone.toUpperCase() + (!g.cacheFound ? ' · 暂不可用' : ''),disabled:!g.cacheFound }))} />{errors.group && <small className="field-error" role="alert">{errors.group}</small>}</div>
        </div><div className="wizard-field"><label>训练算法</label><div className="algorithm-options" role="radiogroup" aria-label="训练算法">{group?.methods.map(method => <button key={method.id} type="button" role="radio" aria-checked={draft.method === method.id} disabled={frozen || !method.enabled} title={method.reason || undefined} className={draft.method === method.id ? 'selected' : ''} onClick={() => chooseMethod(method.id)}><span className="algorithm-glyph">{method.id === 'heterogeneous_solution' ? 'f' : methodLabel(method.id).slice(0,1)}</span><strong>{methodLabel(method.id)}</strong>{method.id === 'heterogeneous_solution' && <small>数据增强</small>}<i>{draft.method === method.id && <CheckOutlined />}</i></button>)}</div>{errors.method && <small className="field-error" role="alert">{errors.method}</small>}</div>
      </>}
      {step === 1 && <><div className="wizard-title"><span>02 / PARAMETERS</span><h2>定义训练方式</h2><div className="wizard-selection">{group?.dataset} <i> / </i> {methodLabel(draft.method)}</div></div>
        <div className="wizard-fields">{number('rounds','通信轮数',1,1000,'不含第 0 轮初始化')}{number('localEpochs','本地轮数',1,100)}{number('learningRate','学习率',1e-8,1,undefined,false,.0001)}{number('batchSize','批大小',1,1024)}{number('clientCount','客户端数量',group?.domains || 1,240,'须为域数量的倍数；固定划分不可修改',group?.partitionLocked,group?.domains)}{number('sampleClients','每轮参与客户端',0,240,'0 表示全部参与')}</div>
        {ours && <section className="augmentation-settings"><div className="wizard-section-title"><h3>数据增强</h3><span>本架构</span></div><div className="wizard-field"><label htmlFor="train-augmentation">生成方式</label><Select id="train-augmentation" aria-label="生成方式" value={draft.augmentationMode} disabled={frozen} onChange={value => { setDraft(old => ({...old,augmentationMode:value,augmentationSourceId:'',allowLegacyAugmentation:false})); }} options={[{value:'generate',label:'按本次配置生成'},{value:'reuse',label:'复用历史生成结果',disabled:!group?.methods.find(m => m.id === 'heterogeneous_solution')?.augmentedCacheFound && !group?.augmentationSources?.length}]} /></div>
          {draft.augmentationMode === 'reuse' && <div className="wizard-field"><label htmlFor="train-source">历史生成记录</label><Select id="train-source" aria-label="历史生成记录" value={draft.augmentationSourceId || ''} disabled={frozen} onChange={id => { const source=group?.augmentationSources?.find(s => s.id === id); setDraft(old => source ? {...source.request,name:old.name,augmentationMode:'reuse',augmentationSourceId:id,allowLegacyAugmentation:false} : {...old,augmentationSourceId:'',allowLegacyAugmentation:false}); }} options={[{value:'',label:'历史结果 · 来源记录不完整',disabled:!group?.methods.find(m => m.id === 'heterogeneous_solution')?.augmentedCacheFound},...(group?.augmentationSources || []).map(source => ({value:source.id,label:source.name}))]} /></div>}
          <div className="wizard-fields">{number('generatedPerSample','每个样本生成数',0,1000)}{number('generatedPerPrototype','每个原型生成数',0,1000)}{number('targetPerClass','每类目标样本数',0,5000,'0 不设目标数量')}{number('covarianceScale','协方差缩放',0,10,undefined,false,.1)}</div>
          {draft.augmentationMode === 'reuse' && !draft.augmentationSourceId && <div className="legacy-consent"><Checkbox disabled={frozen} checked={draft.allowLegacyAugmentation} onChange={event => change('allowLegacyAugmentation',event.target.checked)}>允许来源不完整的历史结果试跑，不用于严格提升证明</Checkbox>{errors.allowLegacyAugmentation && <small role="alert" className="field-error">{errors.allowLegacyAugmentation}</small>}</div>}
        </section>}
        <Collapse ghost className="wizard-advanced" items={[{key:'advanced',label:'更多训练参数',children:<div className="wizard-fields">{number('samplesPerClient','每客户端样本数',0,100000,'0 使用全部样本')}{number('evaluationFrequency','评测间隔',1,1000)}{number('alpha','异构强度 α',.0001,100,'值越小，标签分布差异通常越大',group?.partitionLocked,.1)}{number('splitSeed','划分种子',0,2147483647,'固定划分不可修改',group?.partitionLocked)}{number('seed','随机种子',0,2147483647)}</div>}]} />
      </>}
      {step === 2 && <><div className="wizard-title"><span>03 / READY TO RUN</span><h2>{draft.name || '准备好开始了吗？'}</h2></div>
        <div className="review-identity"><span className="review-identity-icon"><ExperimentOutlined /></span><div><h3>{methodLabel(draft.method)}</h3><p>{group?.dataset} / {group?.backbone.toUpperCase()}</p></div><span className="review-identity-type">联邦训练</span></div>
        <dl className="review-grid">{[['通信轮数',draft.rounds],['客户端',draft.clientCount],['每轮参与',draft.sampleClients || '全部'],['本地轮数',draft.localEpochs],['学习率',draft.learningRate],['批大小',draft.batchSize],['数据增强',ours ? draft.augmentationMode === 'generate' ? '按配置生成' : '复用历史结果' : '无'],['随机种子',draft.seed]].map(([label,value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
        <div className="review-note"><CheckOutlined /><span>启动时自动检查数据与配置，实验结果独立保存。</span></div>
        {ours && draft.augmentationMode === 'reuse' && draft.allowLegacyAugmentation && <Alert type="warning" title="历史生成结果来源不完整，本次仅作试跑。" />}
      </>}
      </div>
      {Object.keys(errors).length > 0 && step > 0 && <Alert type="error" showIcon title="请检查以下参数" description={<ul className="validation-errors">{Object.entries(errors).map(([field,message]) => <li key={field}>{message}</li>)}</ul>} />}
      <div className="wizard-actions"><Button type="text" icon={<ArrowLeftOutlined />} disabled={step === 0 || frozen} onClick={() => {setStep(value => value-1);setErrors({});}}>上一步</Button><span>{step+1} / 3</span>{step < 2 ? <Button type="primary" disabled={frozen || !!sourceError} onClick={() => {if(validate()) setStep(value => value+1);}}>下一步 <ArrowRightOutlined /></Button> : <Button type="primary" size="large" loading={launch.busy} disabled={!!launch.intent || !!running || disconnected || !!sourceError} onClick={start}>启动训练 <ArrowRightOutlined /></Button>}</div>
    </section>
  </div>;
}
