import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Card, Collapse, Empty, Pagination, Select, Space, Spin, Tag } from 'antd';
import { ArrowRightOutlined, DownloadOutlined, PictureOutlined, ScanOutlined } from '@ant-design/icons';
import { api, methodLabel, percent, terminal, type Job, type Library, type Prediction, type SamplePage, type TestSample } from './api';
import { modelHref } from './navigation';
import './inference.css';

function SampleImage({ sample, large = false, onUnavailable }: { sample: TestSample; large?: boolean; onUnavailable?: () => void }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [sample.id, sample.imageSha256]);
  if (!sample.imageAvailable || failed) return <div className="experience-image-missing"><PictureOutlined /><span>{large ? sample.imageError || '原图暂时无法读取' : '原图不可用'}</span></div>;
  return <img src={sample.imageUrl} alt={large ? `测试原图 ${sample.filename}` : `${sample.className} · ${sample.filename}`} loading={large ? 'eager' : 'lazy'} onError={() => { setFailed(true); onUnavailable?.(); }} />;
}

export function PredictionPanel({ job, open }: { job?: Job; open?: (id: string) => void }) {
  const result = job?.status === 'completed' ? job.result as Prediction : undefined;
  return <Card className="platform-panel experience-prediction" title={<span><ScanOutlined /> 模型判断</span>} extra={<Tag color={result ? result.correct ? 'success' : 'warning' : 'default'}>{result ? '实际推理结果' : job && !terminal(job.status) ? '正在推理' : '等待预测'}</Tag>}>
    {!result ? <div className="experience-prediction-empty">{job && !terminal(job.status) ? <><Spin size="large" /><h3>{job.stage}</h3></> : <><ScanOutlined /><h3>{job?.error ? '推理未完成' : '等待模型预测'}</h3><p>{job?.error || '选择样本后开始预测'}</p></>}</div> : <>
      <div className="experience-verdict"><span>预测类别</span><h2>{result.predictedName.replaceAll('_', ' ')}</h2><div><strong>{percent(result.confidence)}</strong><span>Softmax 分数</span></div></div>
      <div className={`experience-ground-truth ${result.correct ? 'matched' : 'mismatched'}`}><div><span>真实标签</span><b>{result.labelName.replaceAll('_', ' ')}</b></div><Tag color={result.correct ? 'success' : 'warning'}>{result.correct ? '预测一致' : '预测不一致'}</Tag></div>
      <div className="experience-ranks"><h3>TOP {result.topK.length}<span>分类分数</span></h3>{result.topK.map((item, index) => <div className="experience-rank" key={item.classIndex}><div><span><i>{String(index + 1).padStart(2, '0')}</i>{item.className.replaceAll('_', ' ')}</span><b>{percent(item.score)}</b></div><div className="experience-rank-track"><span style={{ width: `${item.score * 100}%` }} /></div></div>)}</div>
      <div className="experience-runtime"><span>分类器批次计算 <b>{result.inferenceMs.toFixed(1)} ms</b></span><span>加载与推理 <b>{result.elapsedSeconds.toFixed(2)} s</b></span></div>
      <p className="experience-score-note">分数未经校准；单图结果不代表整体准确率。</p>
      <Space wrap>{open && <Button type="link" onClick={() => open(job!.id)}>完整记录 <ArrowRightOutlined /></Button>}<Button type="link" href={`/api/platform/jobs/${job!.id}/export`} icon={<DownloadOutlined />}>导出结果</Button></Space>
      <Collapse ghost size="small" items={[{ key: 'version', label: '版本与来源', children: <div className="experience-version">{[['模型 SHA-256', result.checkpointSha256], ['测试特征包 SHA-256', result.testBundleSha256], ['当前原图 SHA-256', result.imageSha256], ['样本清单 SHA-256', result.manifestSha256], ['样本关联方式', result.testProvenance], ['推理口径', result.inferenceContract]].map(([label, value]) => <p key={label}><span>{label}</span><code>{value}</code></p>)}</div> }]} />
    </>}
  </Card>;
}

export function ModelExperience({ library, initialModel, initialTestset, onSelectionChange, disabled, create, open }: {
  library: Library; initialModel?: string; initialTestset?: string; onSelectionChange?: (model:string,testset?:string)=>void; disabled: boolean; create: (action: string, payload: object) => Promise<Job>; open: (id: string) => void;
}) {
  const models = library.models.filter(m => /^(officehome|digit3|domainnet)_/.test(m.group));
  const [modelId, setModelId] = useState<string>(initialModel || '');
  const [testsetId, setTestsetId] = useState<string>();
  const [domain, setDomain] = useState<string>();
  const [label, setLabel] = useState<number>();
  const [page, setPage] = useState(1), [refresh, setRefresh] = useState(0);
  const [samples, setSamples] = useState<SamplePage>();
  const [selected, setSelected] = useState<TestSample>();
  const [imageFailed, setImageFailed] = useState(false);
  useEffect(() => setImageFailed(false), [selected?.id, selected?.imageSha256]);
  const [job, setJob] = useState<Job>();
  const [loading, setLoading] = useState(false), [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const model = models.find(m => m.id === modelId);
  const testsets = library.testsets.filter(t => t.featureSpace === model?.featureSpace);
  const test = testsets.find(t => t.id === testsetId);
  const busy = submitting || !!job && !terminal(job.status);
  const frozen = disabled || busy;
  useEffect(() => { if (!modelId && models.length) setModelId((models.find(m => m.kind === 'final') || models[0]).id); }, [modelId, models]);
  useEffect(() => {
    if (!model) return;
    setTestsetId(library.testsets.find(t => t.id === initialTestset && t.featureSpace === model.featureSpace)?.id || library.testsets.find(t => t.id === model.jobId && t.featureSpace === model.featureSpace)?.id || library.testsets.find(t => t.featureSpace === model.featureSpace)?.id);
    setDomain(undefined); setLabel(undefined); setPage(1); setJob(undefined);
  }, [model?.id]);
  useEffect(() => {
    let active = true;
    setSelected(undefined); setSamples(undefined); setJob(undefined); setError('');
    if (!testsetId) return;
    setLoading(true);
    const query = new URLSearchParams({ offset: String((page - 1) * 12), limit: '12' });
    if (domain) query.set('domain', domain);
    if (label != null) query.set('class', String(label));
    void api<SamplePage>(`testsets/${testsetId}/samples?${query}`).then(data => {
      if (active) { setSamples(data); setSelected(data.items.find(item => item.imageAvailable) || data.items[0]); }
    }).catch(e => { if (active) setError(e.message); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [model?.id, testsetId, domain, label, page, refresh]);
  useEffect(() => {
    if (!job || terminal(job.status)) return;
    let active = true, polling = false;
    const timer = setInterval(async () => {
      if (polling) return;
      polling = true;
      try { const next = await api<Job>(`jobs/${job.id}`); if (active) { setJob(next); setError(''); } }
      catch (e) { if (active) setError((e as Error).message); }
      finally { polling = false; }
    }, 800);
    return () => { active = false; clearInterval(timer); };
  }, [job?.id, job?.status]);
  const predict = async () => {
    if (!selected || !model || !test || frozen) return;
    setSubmitting(true); setError(''); setJob(undefined);
    try {
      const created = await create('predict', { modelId, testsetId, sampleId: selected.id, imageSha256: selected.imageSha256,
        name: `单图体验 · ${methodLabel(model.method)} · ${selected.filename}` });
      if (alive.current) setJob(created);
    } catch (e) { if (alive.current) setError((e as Error).message); }
    finally { if (alive.current) setSubmitting(false); }
  };
  if (!models.length) return <div className="studio-empty-state"><ScanOutlined /><h2>从第一个模型开始</h2><p>完成训练后，就能在这里验证图像模型。</p><a className="studio-button primary" href="/?view=train">新建训练 <ArrowRightOutlined /></a>{library.models.length > 0 && <a href="/?view=evaluate">文本模型 · 独立评测</a>}</div>;
  return <div className="model-experience">
    <div className="experience-selectors"><div><label htmlFor="experience-model">模型</label><Select id="experience-model" aria-label="体验模型" showSearch optionFilterProp="label" value={model ? modelId : undefined} placeholder="选择一个模型" disabled={frozen} onChange={value => { setModelId(value); setSelected(undefined); setJob(undefined); onSelectionChange?.(value); }} options={models.map(m => ({ value:m.id,label:m.name+' · '+methodLabel(m.method)+' / '+m.kind }))} /></div>
      <div><label htmlFor="experience-testset">测试集</label><Select id="experience-testset" aria-label="体验测试集" placeholder={model ? '选择兼容测试集' : '先选择模型'} value={testsetId} disabled={frozen || !model} onChange={value => {setTestsetId(value);setDomain(undefined);setLabel(undefined);setPage(1);onSelectionChange?.(modelId,value);}} options={testsets.map(t => ({value:t.id,label:t.name+' · '+t.samples?.toLocaleString()+' 个样本'}))} /></div>
      <div className="experience-model-meta"><span>{methodLabel(model?.method)} <i>·</i> {model?.kind || '—'}</span>{model && <a href={modelHref(model.id,true,testsetId)}>整集评测 <ArrowRightOutlined /></a>}</div>
    </div>
    {initialModel && !model && <Alert type="warning" title="指定模型不可用，请重新选择。" />}
    {model && <div className="experience-context"><span>{model.classes.length} 个类别<i>·</i>训练 {model.trainingRounds ?? '—'} 轮</span><span>{model.trainingRounds != null && model.trainingRounds < 5 ? '短轮试跑模型，不代表目标准确率' : '冻结特征推理'}{model.kind === 'best' ? ' · best 按训练期测试集择优' : ''}</span><a href={'/api/platform/jobs/'+model.jobId+'/model-'+model.kind}><DownloadOutlined /> 下载模型</a></div>}
    {model?.augmentationWarning && <Alert type="warning" title={model.augmentationWarning} />}
    {error && <Alert type="error" showIcon title={error} action={<Button onClick={() => setRefresh(x => x+1)} disabled={busy}>重新读取</Button>} />}
    <div className="experience-workspace">
      <section className="experience-sample-library"><div className="experience-gallery-head"><h3>测试样本<span>{samples?.total.toLocaleString() ?? '—'}</span></h3><div className="experience-filters"><Select aria-label="测试域筛选" placeholder="全部域" allowClear value={domain} disabled={frozen || !test} onChange={v => {setDomain(v);setPage(1);}} options={test?.domains.map(d => ({value:d.name,label:d.name}))} /><Select aria-label="测试类别筛选" placeholder="全部类别" allowClear showSearch optionFilterProp="label" value={label} disabled={frozen || !test} onChange={v => {setLabel(v);setPage(1);}} options={test?.classes.map((name,i) => ({value:i,label:name.replaceAll('_',' ')}))} /></div></div>
        <div className="experience-thumbnails" aria-busy={loading}>{loading ? <div className="sample-loading"><Spin /></div> : samples?.items.map(sample => <button key={sample.id} aria-label={'选择样本 '+sample.className.replaceAll('_',' ')+' · '+sample.filename+' · '+sample.domain} aria-pressed={selected?.id === sample.id} className={selected?.id === sample.id ? 'selected' : ''} disabled={frozen} onClick={() => {setSelected(sample);setJob(undefined);}}><SampleImage sample={sample} /><span>{sample.className.replaceAll('_',' ')}</span></button>)}</div>
        {!loading && samples?.total === 0 && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有符合筛选的样本" />}
        {!test && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="选择测试集" />}
        <Pagination simple size="small" current={page} pageSize={12} total={samples?.total || 0} showSizeChanger={false} disabled={frozen || loading} onChange={setPage} />
      </section>
      <section className="experience-stage"><div className="experience-stage-head"><span><PictureOutlined /> 测试原图</span>{selected && <span className="sample-domain">{selected.domain}</span>}</div>
        <div className={'experience-image '+(model?.group.startsWith('digit3') ? 'experience-digit' : '')}>{loading ? <Spin size="large" /> : selected ? <SampleImage sample={selected} large onUnavailable={() => setImageFailed(true)} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="选择左侧样本" />}</div>
        <div className="experience-caption"><div><span>真实标签</span><strong>{selected?.className.replaceAll('_',' ') || '—'}</strong></div><span title={selected?.filename}>{selected?.filename || '等待样本'}</span></div>
        <div className="experience-stage-footer"><span>测试集原图 · 只读</span><span>{selected ? '样本 '+(selected.index+1) : ''}</span></div>
      </section>
      <div className="experience-output"><Button className="experience-run" type="primary" size="large" icon={<ScanOutlined />} disabled={frozen || imageFailed || !selected?.imageAvailable || !selected.imageSha256 || loading || !test || !model} loading={busy} onClick={() => void predict()}>运行单图预测</Button>
        {job && !terminal(job.status) && <Button type="link" onClick={() => open(job.id)}>查看任务 / 停止</Button>}
        <PredictionPanel job={job} open={open} />
      </div>
    </div>
    <div className="experience-protocol"><Collapse ghost size="small" items={[{key:'contract',label:'推理口径与来源限制',children:<><p>展示测试原图；实际推理使用关联的冻结特征与已保存分类器，不会重新提取特征。</p><p>旧数据按划分与标签顺序关联，缺少内嵌原始样本 ID；当前图片哈希不能证明历史特征由此图片生成。单图判断不代表整体准确率。</p></>}]} />{model && <Button type="link" onClick={() => open(model.jobId)}>查看训练记录 <ArrowRightOutlined /></Button>}</div>
  </div>;
}
