import { useEffect, useState } from 'react';
import { Alert, Button, Card, Empty, Form, Input, Select, Space, Table, Tag } from 'antd';
import { DownloadOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { api, methodLabel, percent, type EvaluationResult, type Job, type Library, type LibraryItem } from './api';
import { Confusion, Curves, DomainBars } from './charts';

export function EvaluationPanel({ library, disabled, create, open }: {
  library: Library; disabled: boolean; create: (action: string, payload: object) => Promise<Job>; open: (id: string) => void;
}) {
  const [modelId, setModelId] = useState<string>();
  const [testsetId, setTestsetId] = useState<string>();
  const [domains, setDomains] = useState<string[]>([]);
  const [classes, setClasses] = useState<number[]>([]);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false), [error, setError] = useState('');
  const model = library.models.find(m => m.id === modelId);
  const testsets = library.testsets.filter(t => t.featureSpace === model?.featureSpace);
  const test = testsets.find(t => t.id === testsetId);
  const submit = async () => {
    setBusy(true); setError('');
    try { const job = await create('evaluate', { modelId, testsetId, domains, classes, name }); open(job.id); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  return <><div className="platform-training-grid"><Card className="platform-panel" title="独立模型评测" extra={<Tag>CPU · 不加载特征主干</Tag>}>
    <p className="platform-muted">重新加载已保存的分类器，使用冻结的测试特征。后台会验证模型和测试包哈希，并拒绝训练集泄漏。</p>
    {error && <Alert type="error" showIcon title={error} />}
    {!library.models.length ? <Empty description="还没有已完成训练的模型。先运行一次缓存训练。" /> : <Form layout="vertical" style={{ marginTop: 20 }}>
      <Form.Item label="评测名称"><Input value={name} onChange={e => setName(e.target.value)} maxLength={120} placeholder="例如：FedAvg · 全域独立评测" /></Form.Item>
      <Form.Item label="选择已有模型" required><Select showSearch optionFilterProp="label" value={modelId} onChange={id => { setModelId(id); setTestsetId(undefined); setDomains([]); setClasses([]); }}
        options={library.models.map(m => ({ value: m.id, label: `${m.name} · ${m.group} · ${methodLabel(m.method)} / ${m.kind}` }))} /></Form.Item>
      <Form.Item label="选择兼容测试集" required><Select value={testsetId} disabled={!model} onChange={id => { setTestsetId(id); setDomains([]); setClasses([]); }}
        options={testsets.map(t => ({ value: t.id, label: `${t.name} · ${t.samples} 样本 · ${t.id.slice(0, 8)}` }))} /></Form.Item>
      <Form.Item label="测试域（留空表示全部）"><Select mode="multiple" value={domains} disabled={!test} onChange={setDomains} options={test?.domains.map(d => ({ value: d.name, label: `${d.name} (${d.testSamples})` }))} /></Form.Item>
      <Form.Item label="测试类别（留空表示全部）"><Select mode="multiple" showSearch optionFilterProp="label" value={classes} disabled={!test} onChange={setClasses} maxTagCount={5}
        options={test?.classes.map((c, i) => ({ value: i, label: `${i}: ${c}` }))} /></Form.Item>
      {model?.kind === 'best' && <Alert type="warning" title="best 模型用训练期测试集择优，评测不可视为无偏泛化验证。正式对比建议使用 final。" />}
      {model?.augmentationWarning && <Alert type="warning" title={model.augmentationWarning} />}
      <Button type="primary" icon={<PlayCircleOutlined />} loading={busy} disabled={disabled || !model || !test} onClick={() => void submit()}>启动独立评测</Button>
      {disabled && <p className="platform-muted">请等待当前任务结束并确认服务器连接正常。</p>}
    </Form>}
  </Card><Card className="platform-panel" title="评测口径"><div className="platform-note"><strong>总体准确率</strong><p>全部选中测试样本加权，不等同于分域准确率的平均。</p><br /><strong>分域与分类指标</strong><p>输出准确率、Precision、Recall、Macro-F1、混淆矩阵及各类别样本数。Macro 按真值或预测中出现的类别等权。</p><br /><strong>可复现的测试集</strong><p>测试特征、划分、标签版本和模型哈希与评测结果一并保存。不会下载或临时生成测试数据。</p></div></Card></div>
    <Card className="platform-panel" title="已保存模型"><Table<LibraryItem> rowKey="id" dataSource={library.models} pagination={{ pageSize: 8 }} scroll={{ x: 650 }} columns={[
      { title: '模型', render: (_, m) => <span>{m.name}<small style={{ display: 'block' }}>{m.group} / {methodLabel(m.method)}</small></span> },
      { title: '检查点', dataIndex: 'kind', render: k => <Tag color={k === 'final' ? 'blue' : 'default'}>{k}</Tag> },
      { title: 'SHA-256', dataIndex: 'sha256', render: value => <code title={value}>{value.slice(0, 16)}</code> },
      { title: '操作', render: (_, m) => <Space><Button type="link" onClick={() => open(m.jobId)}>训练记录</Button><Button type="link" href={`/api/platform/jobs/${m.jobId}/model-${m.kind}`} icon={<DownloadOutlined />}>下载</Button></Space> },
    ]} /></Card></>;
}

export function EvaluationResults({ job, library }: { job: Job; library: Library }) {
  const [domain, setDomain] = useState('overall');
  const result = job.result as EvaluationResult | undefined;
  const model = library.models.find(m => m.id === job.request.modelId);
  if (!result) return <Empty description="等待独立评测结果" />;
  const selected = domain === 'overall' ? result : result.domains[domain] || result;
  const labels = model?.classes || result.confusionMatrix.map((_, i) => String(i));
  return <><div className="platform-stats">{[['总体准确率', percent(result.accuracy)], ['Macro-F1', percent(result.macroF1)], ['分域平均', percent(result.domainMean)], ['测试样本', result.samples]].map(([name, value]) => <div className="platform-stat" key={name}><span>{name}</span><strong>{value}</strong></div>)}</div>
    {model?.augmentationWarning && <Alert type="warning" title={model.augmentationWarning} />}
    <div className="platform-grid"><Card className="platform-panel" title="分域准确率"><DomainBars domains={result.domains} /></Card>
      <Card className="platform-panel" title="分域指标"><Table rowKey="domain" pagination={false} scroll={{ x: 420 }} size="small" dataSource={Object.entries(result.domains).map(([domain, m]) => ({ domain, ...m }))} columns={[
        { title: '域', dataIndex: 'domain' }, { title: '样本', dataIndex: 'samples' }, { title: '准确率', dataIndex: 'accuracy', render: percent }, { title: 'Macro-F1', dataIndex: 'macroF1', render: percent },
      ]} /><p className="platform-muted" style={{ marginTop: 15 }}>最差域 {percent(result.worstDomain)} · 域间差距 {percent(result.domainGap)}</p></Card></div>
    <Card className="platform-panel" title="分类指标与混淆矩阵" extra={<Space><Select value={domain} onChange={setDomain} style={{ minWidth: 150 }} options={[{ value: 'overall', label: '总体' }, ...Object.keys(result.domains).map(d => ({ value: d, label: d }))]} /><Button href={`/api/platform/jobs/${job.id}/csv`}>导出 CSV</Button></Space>}>
      <div className="platform-grid"><Confusion matrix={selected.confusionMatrix} classes={labels} /><Table rowKey="classIndex" dataSource={selected.perClass} pagination={{ pageSize: 8 }} size="small" scroll={{ x: 500 }} columns={[
        { title: '类别', dataIndex: 'classIndex', render: i => `${i}: ${labels[i]}` }, { title: '样本数', dataIndex: 'support' },
        { title: 'Precision', dataIndex: 'precision', render: percent }, { title: 'Recall', dataIndex: 'recall', render: percent }, { title: 'F1', dataIndex: 'f1', render: percent },
      ]} /></div></Card>
    <Alert type="info" title="Macro 指标按当前子集中出现的真值或预测类别计算；筛选类别只筛选真值，不限制模型预测类别。" />
  </>;
}

export function ComparisonPanel({ jobs, open }: { jobs: Job[]; open: (id: string) => void }) {
  const [ids, setIds] = useState<string[]>([]), [selected, setSelected] = useState<Job[]>([]), [error, setError] = useState('');
  const [kind, setKind] = useState('evaluate');
  useEffect(() => { let active = true; setError(''); void Promise.all(ids.map(id => api<Job>(`jobs/${id}`))).then(j => { if (active) setSelected(j); }).catch(e => { if (active) setError(e.message); }); return () => { active = false; }; }, [ids]);
  const signature = (j: Job) => j.action === 'evaluate' ? JSON.stringify([j.request.testsetId, [...(j.request.domains || [])].sort(), [...(j.request.classes || [])].sort()]) : JSON.stringify([j.request.group, j.data?.testFingerprint, j.data?.partitionFingerprint, ...['rounds', 'localEpochs', 'learningRate', 'batchSize', 'sampleClients', 'clientCount', 'seed', 'splitSeed', 'alpha', 'samplesPerClient'].map(k => j.request[k as keyof Job['request']])]);
  const comparable = new Set(selected.map(signature)).size <= 1;
  const download = () => { const blob = new Blob([JSON.stringify({ comparable, definition: 'same testset and selected subset; see each run config for training differences', jobs: selected }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'federatedscope-comparison.json'; a.click(); URL.revokeObjectURL(url); };
  return <><Card className="platform-panel" title="选择已完成实验" extra={<Select value={kind} onChange={v => { setKind(v); setIds([]); }} options={[{ value: 'evaluate', label: '独立评测' }, { value: 'train', label: '训练曲线' }]} />}>
    <Table<Job> rowKey="id" dataSource={jobs.filter(j => j.action === kind && j.status === 'completed')} pagination={{ pageSize: 8 }} rowSelection={{ selectedRowKeys: ids, onChange: keys => setIds(keys.map(String).slice(-4)) }} columns={[
      { title: '实验', render: (_, j) => <Button type="link" onClick={() => open(j.id)}>{j.request.name || j.id.slice(0, 8)}</Button> },
      { title: '配置', render: (_, j) => `${j.request.group} / ${methodLabel(j.request.method)}` }, { title: '时间', dataIndex: 'createdAt', render: v => new Date(v).toLocaleString() },
    ]} /></Card>
    {error && <Alert type="error" title={error} />}
    {!!selected.length && <><Space style={{ marginBottom: 18 }}><Tag>{selected.length} 项实验（最多 4 项）</Tag><Button icon={<DownloadOutlined />} onClick={download}>导出对比 JSON</Button></Space>
      {!comparable && <Alert type="warning" title="测试集、划分或筛选范围不一致，不应直接比较优劣。请统一协议后重跑。" />}
      <Alert type="info" title="同轮数不等于同计算量：增强可能改变训练样本数。请同时核对学习率、本地轮数、参与客户端和随机种子；短轮试跑不证明稳定提升。" />
      {selected.some(j => j.data?.augmentation?.warning) && <Alert type="warning" title="包含历史增强缓存实验，生成来源未完整核验；这里只展示观察结果。" />}
      {kind === 'evaluate' ? <Card className="platform-panel" title="独立评测对比"><Table<Job> rowKey="id" dataSource={selected} pagination={false} scroll={{ x: 700 }} columns={[
        { title: '实验', render: (_, j) => j.request.name || j.id.slice(0, 8) }, { title: '算法', render: (_, j) => methodLabel(j.request.method) },
        ...(['accuracy', 'macroF1', 'domainMean', 'worstDomain'] as const).map((metric, i) => ({ title: ['总体准确率', 'Macro-F1', '分域平均', '最差域'][i], render: (_: unknown, j: Job) => percent((j.result as EvaluationResult)[metric]) })),
        { title: '样本数', render: (_, j) => (j.result as EvaluationResult).samples },
      ]} /></Card> : <div className="platform-grid">{selected.map(j => <Card className="platform-panel" title={`${j.request.name || j.id.slice(0, 8)} · ${methodLabel(j.request.method)}`} key={j.id}><Curves points={j.metrics} /><p className="platform-muted">lr={j.request.learningRate} · local={j.request.localEpochs} · seed={j.request.seed} · clients={j.request.clientCount}</p></Card>)}</div>}
    </>}
  </>;
}
