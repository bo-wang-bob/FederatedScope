import { useEffect, useState } from 'react';
import { Alert, Button, Checkbox, Collapse, Empty, Form, Input, InputNumber, Select, Tag } from 'antd';
import { DatabaseOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { api, methodLabel, terminal, type Catalog, type DataInfo, type Job, type RequestConfig, type Resource } from './api';
import { Panel, State, Stat } from './ui';

export function TrainingForm({ catalog, initialGroup, sourceId, resources, running, disconnected, create, open }: {
  catalog: Catalog; initialGroup?: string; sourceId?: string; resources?: Resource; running?: Job; disconnected: boolean;
  create: (action: string, payload: object) => Promise<Job>; open: (id: string) => void;
}) {
  const [form] = Form.useForm<RequestConfig>();
  const [groupId, setGroupId] = useState(initialGroup || 'officehome_vit');
  const [preflight, setPreflight] = useState<Job>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const group = catalog.groups.find(g => g.id === groupId) || catalog.groups[0];
  const method = Form.useWatch('method', form);
  const augmentationMode = Form.useWatch('augmentationMode', form);
  const ours = method === 'heterogeneous_solution';
  useEffect(() => { const defaults = group.methods.find(m => m.enabled)?.defaults; if (defaults) form.setFieldsValue(defaults); setPreflight(undefined); }, [groupId, form]);
  useEffect(() => {
    if (!sourceId) return;
    let active = true;
    void api<Job>(`jobs/${sourceId}`).then(job => { if (active) form.setFieldsValue(job.request); }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [sourceId, form]);
  useEffect(() => {
    if (!preflight || terminal(preflight.status)) return;
    let active = true;
    const timer = setInterval(async () => { try { const j = await api<Job>(`jobs/${preflight.id}`); if (active) { setPreflight(j); setError(''); } }
      catch (e) { if (active) setError((e as Error).message); } }, 1000);
    return () => { active = false; clearInterval(timer); };
  }, [preflight?.id, preflight?.status]);
  const run = async (action: 'preflight' | 'train') => {
    setBusy(true); setError('');
    try { const validated = await form.validateFields(); const values = { ...form.getFieldsValue(true), ...validated }; const job = await create(action, { ...values, ...(action === 'train' ? { preflightId: preflight?.id } : {}) });
      if (action === 'preflight') setPreflight(job); else open(job.id);
    } catch (e) { setError(e instanceof Error ? e.message : '请检查参数'); } finally { setBusy(false); }
  };
  const info = preflight?.result as DataInfo | undefined;
  const blocked = !!running || disconnected || busy || (!!preflight && !terminal(preflight.status));
  const values = Form.useWatch([], form) as RequestConfig | undefined;
  const number = (name: keyof RequestConfig, label: string, min: number, max: number, extra?: string, disabled = false, step = 1) => <Form.Item name={name} label={label} tooltip={extra} rules={[{ required: true }]}><InputNumber min={min} max={max} step={step} disabled={disabled || undefined} style={{ width: '100%' }} /></Form.Item>;
  return <div className="platform-training-grid"><Panel title="实验配置" extra={<Tag color="purple">准确率</Tag>}>
    {running && <Alert type="info" title="当前有任务运行中" action={<Button onClick={() => open(running.id)}>查看任务</Button>} />}
    {error && <Alert type="error" showIcon title={error} />}
    <Form form={form} layout="vertical" disabled={busy || (!!preflight && !terminal(preflight.status))} onValuesChange={() => setPreflight(undefined)} initialValues={group.methods.find(m => m.enabled)?.defaults}>
      <div className="studio-form-section"><span>01</span><h3>基础配置</h3></div>
      <Form.Item name="name" label="实验名称"><Input maxLength={120} placeholder="例如：Office-Home · FedAvg 基线" /></Form.Item>
      <div className="platform-form-grid"><Form.Item name="group" label="数据集与特征主干"><Select onChange={setGroupId} options={catalog.groups.map(g => ({ value: g.id, label: `${g.dataset} / ${g.backbone.toUpperCase()}`, disabled: !g.cacheFound }))} /></Form.Item>
        <Form.Item name="method" label="算法"><Select onChange={id => { const defaults = group.methods.find(m => m.id === id)?.defaults; if (defaults) { const values = form.getFieldsValue(true); form.setFieldsValue({ ...defaults, ...Object.fromEntries(['rounds', 'localEpochs', 'learningRate', 'batchSize', 'sampleClients', 'clientCount', 'seed', 'splitSeed', 'alpha', 'gpu', 'samplesPerClient', 'evaluationFrequency', 'name'].filter(k => values[k as keyof RequestConfig] !== undefined).map(k => [k, values[k as keyof RequestConfig]])) }); } setPreflight(undefined); }} options={group.methods.map(m => ({ value: m.id, label: methodLabel(m.id), disabled: !m.enabled }))} /></Form.Item></div>
      <div className="studio-form-section"><span>02</span><h3>训练参数</h3></div><div className="platform-form-grid">
        {number('rounds', '通信轮数', 1, 1000, '实际模型更新次数，不含第 0 轮初始化')}{number('localEpochs', '本地训练轮数', 1, 100)}
        {number('learningRate', '学习率', 1e-8, 1, undefined, false, .0001)}{number('batchSize', 'Batch size', 1, 1024)}
        {number('clientCount', '客户端总数', group.domains, 240, `必须为 ${group.domains} 的倍数`, group.partitionLocked, group.domains)}{number('sampleClients', '每轮参与客户端数', 0, 240, '0 表示全部参与')}
        <Form.Item name="gpu" label="训练设备"><Select options={[...(resources?.gpus || []).map(g => ({ value: g.index, label: `GPU ${g.index} · ${g.name} · ${g.utilization}% 占用` })), { value: -1, label: 'CPU' }]} /></Form.Item>
        {number('samplesPerClient', '每客户端训练样本数', 0, 100000, '0 使用全部已有样本；正数启用确定性抽样')}
      </div>
      <Collapse ghost items={[{ key: 'advanced', label: '高级参数', children: <div className="platform-form-grid">
        {number('alpha', 'Dirichlet α', .0001, 100, '值越小，标签分布差异通常越大', group.partitionLocked, .1)}
        {number('splitSeed', '数据划分种子', 0, 2147483647, '改变后必须匹配已有测试缓存', group.partitionLocked)}
        {number('seed', '训练随机种子', 0, 2147483647)}{number('evaluationFrequency', '每多少轮评测一次', 1, 1000)}
      </div> }]} />
      {ours && <><div className="studio-form-section"><span>03</span><h3>数据增强</h3></div><Form.Item name="augmentationMode" label="增强方式"><Select onChange={() => form.setFieldsValue({ augmentationSourceId: '', allowLegacyAugmentation: false })} options={[{ value: 'generate', label: '重新生成增强数据' }, { value: 'reuse', label: '复用已有增强缓存', disabled: !group.methods.find(m => m.id === 'heterogeneous_solution')?.augmentedCacheFound && !group.augmentationSources?.length }]} /></Form.Item>
        {augmentationMode === 'reuse' && <Form.Item name="augmentationSourceId" label="增强缓存来源"><Select onChange={id => { const source = group.augmentationSources?.find(s => s.id === id); if (source) form.setFieldsValue({ ...source.request, name: '', augmentationSourceId: id, augmentationMode: 'reuse', allowLegacyAugmentation: false }); }} options={[{ value: '', label: '服务器历史缓存（来源有限制）', disabled: !group.methods.find(m => m.id === 'heterogeneous_solution')?.augmentedCacheFound }, ...(group.augmentationSources || []).map(s => ({ value: s.id, label: `${s.name} · ${s.id.slice(0, 8)}` }))]} /></Form.Item>}
        <div className="platform-form-grid">{number('generatedPerSample', '每个原始样本生成数', 0, 1000)}{number('generatedPerPrototype', '每个原型生成数', 0, 1000)}{number('targetPerClass', '每客户端每类目标样本数', 0, 5000, '0 不限制；正数由原算法选择或补齐')}{number('covarianceScale', '生成协方差缩放', 0, 10, '按源算法缩放生成协方差', false, .1)}</div>
        <Alert type={augmentationMode === 'reuse' ? 'warning' : 'info'} showIcon title={augmentationMode === 'reuse' ? '缓存须匹配增强参数与种子' : '启动后按当前配置生成增强数据'} />
        {augmentationMode === 'reuse' && <Form.Item name="allowLegacyAugmentation" valuePropName="checked"><Checkbox>允许历史缓存试跑：来源记录不完整，结果不能直接作为严格提升证明</Checkbox></Form.Item>}
      </>}
      <div className="platform-form-actions"><Button icon={<DatabaseOutlined />} loading={busy} disabled={blocked} onClick={() => void run('preflight')}>检查完整缓存</Button><Button type="primary" icon={<PlayCircleOutlined />} disabled={blocked || preflight?.status !== 'completed'} onClick={() => void run('train')}>启动训练</Button></div>
    </Form>
  </Panel><div className="studio-training-aside"><section className="studio-run-summary"><span className="studio-kicker">EXPERIMENT</span><h3>{values?.name || '未命名实验'}</h3><div className="studio-summary-method">{methodLabel(values?.method)}<Tag>{group.backbone.toUpperCase()}</Tag></div><dl><div><dt>数据集</dt><dd>{group.dataset}</dd></div><div><dt>通信轮数</dt><dd>{values?.rounds ?? '—'}</dd></div><div><dt>客户端</dt><dd>{values?.clientCount ?? '—'}</dd></div><div><dt>学习率</dt><dd>{values?.learningRate ?? '—'}</dd></div><div><dt>增强</dt><dd>{ours ? augmentationMode === 'generate' ? '重新生成' : '缓存复用' : '无'}</dd></div></dl></section><Panel title="缓存预检" extra={preflight && <State value={preflight.status} />}>
    {!preflight ? <Empty description="等待预检" /> : <><p>{preflight.stage}</p>{preflight.error && <Alert type="error" title={preflight.error} />}
      {info && <><div className="platform-stats compact"><Stat label="客户端" value={info.clientCount} /><Stat label="训练样本" value={info.trainSamples} /><Stat label="测试样本" value={info.testSamples} /></div><Tag>版本 {info.testFingerprint.slice(0, 12)}</Tag><Collapse ghost items={[{ key: 'provenance', label: '数据来源', children: '旧缓存按划分、样本数与标签顺序核验；缺少原始样本 ID 时不能完整证明逐样本来源。' }]} /></>}
      {info?.augmentation?.warning && <Alert type="warning" title={info.augmentation.warning} />}
      {info?.augmentation?.mode === 'generate' && <Alert type="info" title="原始特征已核验 · 待生成增强数据" />}
      <Button type="link" onClick={() => open(preflight.id)}>查看预检详情</Button></>}
  </Panel><p className="studio-footnote">原始特征缓存只读；增强数据按实验独立保存。</p></div></div>;
}
