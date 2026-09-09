import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Card, Collapse, Empty, Pagination, Select, Space, Spin, Tag } from 'antd';
import { ArrowRightOutlined, CheckCircleOutlined, DownloadOutlined, PictureOutlined, ScanOutlined } from '@ant-design/icons';
import { api, percent, terminal, type Job, type Library, type Prediction, type SamplePage, type TestSample } from './api';
import './inference.css';

function SampleImage({ sample, large = false }: { sample: TestSample; large?: boolean }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [sample.id, sample.imageSha256]);
  if (!sample.imageAvailable || failed) return <div className="experience-image-missing"><PictureOutlined /><span>{large ? sample.imageError || '原图暂时无法读取' : '原图不可用'}</span></div>;
  return <img src={sample.imageUrl} alt={large ? `测试原图 ${sample.filename}` : `${sample.className} · ${sample.filename}`} loading={large ? 'eager' : 'lazy'} onError={() => setFailed(true)} />;
}

export function PredictionPanel({ job, open }: { job?: Job; open?: (id: string) => void }) {
  const result = job?.status === 'completed' ? job.result as Prediction : undefined;
  return <Card className="platform-panel experience-prediction" title={<span><ScanOutlined /> 模型判断</span>} extra={<Tag color={result ? result.correct ? 'success' : 'warning' : 'default'}>{result ? '实际推理结果' : job && !terminal(job.status) ? '正在推理' : '等待预测'}</Tag>}>
    {!result ? <div className="experience-prediction-empty">{job && !terminal(job.status) ? <><Spin size="large" /><h3>{job.stage}</h3><p>正在读取已保存模型与样本特征</p></> : <><ScanOutlined /><h3>{job?.error ? '此次推理未完成' : '让模型给出自己的判断'}</h3><p>{job?.error || '选择一张测试图片，然后运行单图预测。'}</p></>}</div> : <>
      <div className="experience-verdict"><span>预测类别</span><h2>{result.predictedName.replaceAll('_', ' ')}</h2><div><strong>{percent(result.confidence)}</strong><span>Softmax 分数</span></div></div>
      <div className={`experience-ground-truth ${result.correct ? 'matched' : 'mismatched'}`}><div><span>真实标签</span><b>{result.labelName.replaceAll('_', ' ')}</b></div><Tag color={result.correct ? 'success' : 'warning'}>{result.correct ? '预测一致' : '预测不一致'}</Tag></div>
      <div className="experience-ranks"><h3>TOP {result.topK.length}<span>分类分数</span></h3>{result.topK.map((item, index) => <div className="experience-rank" key={item.classIndex}><div><span><i>{String(index + 1).padStart(2, '0')}</i>{item.className.replaceAll('_', ' ')}</span><b>{percent(item.score)}</b></div><div className="experience-rank-track"><span style={{ width: `${item.score * 100}%` }} /></div></div>)}</div>
      <div className="experience-runtime"><span>分类器批次计算 <b>{result.inferenceMs.toFixed(1)} ms</b></span><span>加载与推理 <b>{result.elapsedSeconds.toFixed(2)} s</b></span></div>
      <p className="experience-score-note">分数来自真实模型输出，未经置信度校准；单张预测不能代替完整测试集评测。</p>
      <Space wrap>{open && <Button type="link" onClick={() => open(job!.id)}>完整记录 <ArrowRightOutlined /></Button>}<Button type="link" href={`/api/platform/jobs/${job!.id}/export`} icon={<DownloadOutlined />}>导出结果</Button></Space>
      <Collapse ghost size="small" items={[{ key: 'version', label: '查看本次模型与样本版本', children: <div className="experience-version">{[['模型 SHA-256', result.checkpointSha256], ['测试特征包 SHA-256', result.testBundleSha256], ['当前原图 SHA-256', result.imageSha256], ['样本清单 SHA-256', result.manifestSha256], ['样本关联方式', result.testProvenance], ['推理口径', result.inferenceContract]].map(([label, value]) => <p key={label}><span>{label}</span><code>{value}</code></p>)}</div> }]} />
    </>}
  </Card>;
}

export function ModelExperience({ library, disabled, create, open }: {
  library: Library; disabled: boolean; create: (action: string, payload: object) => Promise<Job>; open: (id: string) => void;
}) {
  const models = library.models.filter(m => /^(officehome|digit3|domainnet)_/.test(m.group));
  const [modelId, setModelId] = useState<string>();
  const [testsetId, setTestsetId] = useState<string>();
  const [domain, setDomain] = useState<string>();
  const [label, setLabel] = useState<number>();
  const [page, setPage] = useState(1), [refresh, setRefresh] = useState(0);
  const [samples, setSamples] = useState<SamplePage>();
  const [selected, setSelected] = useState<TestSample>();
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
    setTestsetId(library.testsets.find(t => t.id === model.jobId)?.id);
    setDomain(undefined); setLabel(undefined); setPage(1); setJob(undefined);
  }, [modelId]);
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
  }, [testsetId, domain, label, page, refresh]);
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
        name: `单图体验 · ${model.method} · ${selected.filename}` });
      if (alive.current) setJob(created);
    } catch (e) { if (alive.current) setError((e as Error).message); }
    finally { if (alive.current) setSubmitting(false); }
  };
  if (!models.length) return <Empty description="暂无可体验的已完成图像模型。完成缓存训练后，模型和测试集会自动出现在这里；文本模型请使用独立评测。" />;
  return <div className="model-experience">
    <div className="experience-intro"><div><span className="platform-eyebrow">MODEL EXPERIENCE</span><h2>一个样本，看见模型的判断</h2><p>浏览真实测试图片，调用已保存模型，核对每一次预测。</p></div><Tag icon={<CheckCircleOutlined />} color="cyan">冻结特征分类器 · 真实推理</Tag></div>
    <div className="experience-selectors"><div><label htmlFor="experience-model">01 / 选择模型</label><Select id="experience-model" aria-label="体验模型" showSearch optionFilterProp="label" value={modelId} disabled={frozen} onChange={value => { setModelId(value); setSelected(undefined); setJob(undefined); }} options={models.map(m => ({ value: m.id, label: `${m.name} · ${m.method.toUpperCase()} · ${m.kind}` }))} /></div><div><label htmlFor="experience-testset">02 / 选择测试集</label><Select id="experience-testset" aria-label="体验测试集" value={testsetId} disabled={frozen} onChange={value => { setTestsetId(value); setDomain(undefined); setLabel(undefined); setPage(1); }} options={testsets.map(t => ({ value: t.id, label: `${t.group} · ${t.samples?.toLocaleString()} 个样本 · ${t.name}` }))} /></div><div className="experience-model-meta"><Tag>{model?.method.toUpperCase()}</Tag><span>{model?.classes.length} 类 · 训练 {model?.trainingRounds ?? '—'} 轮</span>{model && <a href={`/api/platform/jobs/${model.jobId}/model-${model.kind}`}>下载模型 <DownloadOutlined /></a>}</div></div>
    {model && model.trainingRounds != null && model.trainingRounds < 5 && <Alert type="warning" showIcon title={`当前模型只训练了 ${model.trainingRounds} 轮，适合核验流程，不代表已达到目标准确率。`} />}
    {model?.kind === 'best' && <Alert type="info" showIcon title="best 检查点按训练期测试集择优，不能视为无偏的泛化表现。" />}
    {error && <Alert type="error" showIcon title={error} action={<Button onClick={() => setRefresh(x => x + 1)} disabled={busy}>重新读取</Button>} />}
    <div className="experience-workspace"><div><section className="experience-stage"><div className="experience-stage-head"><span><PictureOutlined /> 测试原图</span><Space>{selected && <Tag>{selected.domain}</Tag>}<Tag>只读样本</Tag></Space></div><div className={`experience-image ${model?.group.startsWith('digit3') ? 'experience-digit' : ''}`}>{loading ? <Spin size="large" /> : selected ? <SampleImage sample={selected} large /> : <Empty description="请选择测试样本" />}</div><div className="experience-caption"><div><span>真实标签</span><strong>{selected?.className.replaceAll('_', ' ') || '—'}</strong></div><div><span>{selected?.filename || '等待样本'}</span><code>{selected?.id || ''}</code></div></div></section>
      <div className="experience-sample-library"><div className="experience-gallery-head"><h3>测试样本库 <span>{samples?.total.toLocaleString() ?? '—'} 张</span></h3><Space wrap><Select aria-label="测试域筛选" placeholder="全部域" allowClear value={domain} disabled={frozen} onChange={v => { setDomain(v); setPage(1); }} options={test?.domains.map(d => ({ value: d.name, label: d.name }))} style={{ minWidth: 130 }} /><Select aria-label="测试类别筛选" placeholder="全部类别" allowClear showSearch optionFilterProp="label" value={label} disabled={frozen} onChange={v => { setLabel(v); setPage(1); }} options={test?.classes.map((name, i) => ({ value: i, label: `${i} · ${name}` }))} style={{ minWidth: 165 }} /></Space></div>
        <div className="experience-thumbnails">{samples?.items.map(sample => <button key={sample.id} aria-label={`选择样本 ${sample.filename}`} aria-pressed={selected?.id === sample.id} className={selected?.id === sample.id ? 'selected' : ''} disabled={frozen} onClick={() => { setSelected(sample); setJob(undefined); }}><SampleImage sample={sample} /><span>{sample.className.replaceAll('_', ' ')}</span><small>{sample.domain}</small></button>)}</div>
        {!loading && samples?.total === 0 && <Empty description="这个筛选范围没有测试样本" />}<Pagination size="small" current={page} pageSize={12} total={samples?.total || 0} showSizeChanger={false} disabled={frozen} onChange={setPage} /></div>
    </div><div className="experience-output"><PredictionPanel job={job} open={open} /><Button className="experience-run" type="primary" size="large" icon={<ScanOutlined />} disabled={frozen || !selected?.imageAvailable || !selected.imageSha256 || loading} loading={busy} onClick={() => void predict()}>运行单图预测</Button>{job && !terminal(job.status) && <Button type="link" onClick={() => open(job.id)}>查看任务 / 停止</Button>}
      <div className="experience-protocol"><b>如何读取这份结果</b><p>展示的是测试原图；推理输入是它对应的已有冻结特征，运行的是已保存分类器，不会重新提取特征。</p><p>旧缓存按划分与标签顺序关联样本，未内嵌原始样本 ID。当前图片哈希仅证明本次读取版本，不证明历史特征由该图片重新计算。</p><Button type="link" onClick={() => model && open(model.jobId)}>查看模型训练记录 <ArrowRightOutlined /></Button></div>
    </div></div>
  </div>;
}
