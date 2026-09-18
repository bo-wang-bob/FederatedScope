import { useEffect, useState } from 'react';
import { Alert, Button, Card, Empty, Image, Select, Space, Spin, Table } from 'antd';
import { DownloadOutlined, ScanOutlined } from '@ant-design/icons';
import { api, methodLabel, percent, terminal, terminology, type Job, type Library, type UploadedPrediction } from './api';
import { DatasetUpload, type UploadedDataset } from './datasetUpload';

export function UploadedResults({ job }: { job: Job }) {
  const result = job.status === 'completed' ? job.result as UploadedPrediction : undefined;
  if (!result) return <>{!terminal(job.status) && <Space><Spin /><span>{terminology(job.stage)}</span></Space>}{job.error && <Alert type="error" showIcon title={terminology(job.error)} />}</>;
  return <Card className="platform-panel" title={`预测结果 · ${result.samples} 张`} extra={<Button icon={<DownloadOutlined />} href={`/api/platform/jobs/${job.id}/export`}>导出结果</Button>}>
    {result.labelled && result.metrics && <Space wrap style={{ marginBottom: 20 }}><strong>准确率 {percent(result.metrics.accuracy)}</strong><span>宏平均 F1 {percent(result.metrics.macroF1)}</span><span>精确率 {percent(result.metrics.precision)}</span><span>召回率 {percent(result.metrics.recall)}</span></Space>}
    <Table rowKey="index" dataSource={result.items} pagination={{ pageSize: 8, showSizeChanger: false, hideOnSinglePage: true }} scroll={{ x: 650 }} columns={[
      { title: '图片', key: 'image', width: 110, render: (_, item) => <Image src={item.imageUrl} alt={item.filename} width={72} height={72} style={{ objectFit: 'contain' }} /> },
      { title: '文件名', dataIndex: 'filename', ellipsis: true },
      { title: '预测类别', dataIndex: 'predictedName' },
      { title: '分类分数', dataIndex: 'confidence', render: percent },
      ...(result.labelled ? [
        { title: '真实类别', dataIndex: 'labelName' },
        { title: '结果', dataIndex: 'correct', render: (correct: boolean) => correct ? '正确' : '错误' },
      ] : []),
    ]} />
  </Card>;
}

export function UploadedTesting({ library, initialModel, disabled, create, open }: {
  library: Library; initialModel?: string; disabled: boolean;
  create: (action: string, payload: object) => Promise<Job>; open: (id: string) => void;
}) {
  // Only these models currently have a verified, reproducible raw-image backbone.
  const models = library.models.filter(m => m.group.startsWith('uploaded_'));
  const [modelId, setModelId] = useState(initialModel || '');
  const [datasets, setDatasets] = useState<UploadedDataset[]>([]), [uploadId, setUploadId] = useState('');
  const [job, setJob] = useState<Job>(), [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(''), [refresh, setRefresh] = useState(0);
  const model = models.find(m => m.id === modelId), dataset = datasets.find(d => d.id === uploadId);
  const busy = submitting || !!job && !terminal(job.status), frozen = disabled || busy;
  useEffect(() => {
    if (!models.some(m => m.id === modelId)) setModelId((models.find(m => m.kind === 'final') || models[0])?.id || '');
  }, [library.models, modelId]);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try { const values = await api<UploadedDataset[]>('datasets'); if (alive) { setDatasets(previous => [...previous.filter(d => !values.some(v => v.id === d.id)), ...values.filter(d => d.kind === 'test')]); setError(''); } }
      catch (e) { if (alive) setError((e as Error).message); }
    };
    void load(); window.addEventListener('datasets:changed', load);
    return () => { alive = false; window.removeEventListener('datasets:changed', load); };
  }, [refresh]);
  useEffect(() => {
    if (!job || terminal(job.status)) return;
    let alive = true, polling = false;
    const timer = setInterval(async () => {
      if (polling) return; polling = true;
      try { const next = await api<Job>(`jobs/${job.id}`); if (alive) { setJob(next); setError(''); } }
      catch (e) { if (alive) setError((e as Error).message); }
      finally { polling = false; }
    }, 1200);
    return () => { alive = false; clearInterval(timer); };
  }, [job?.id, job?.status]);
  const saved = (value: UploadedDataset) => { setDatasets(old => [value, ...old.filter(d => d.id !== value.id)]); setUploadId(value.id); setJob(undefined); setError(''); };
  const run = async () => {
    if (!model || !dataset || frozen) return;
    setSubmitting(true); setError(''); setJob(undefined);
    try { setJob(await create('test-upload', { modelId, uploadId, name: `图片测试 · ${model.name} · ${dataset.name}` })); }
    catch (e) { setError((e as Error).message); }
    finally { setSubmitting(false); }
  };
  return <div className="model-experience uploaded-testing">
    <div className="experience-selectors">
      <div><label htmlFor="upload-test-model">模型</label><Select id="upload-test-model" aria-label="上传测试模型" showSearch optionFilterProp="label" value={model?.id} placeholder="选择模型" disabled={frozen} onChange={value => { setModelId(value); setJob(undefined); setError(''); }} options={models.map(m => ({ value: m.id, label: `${m.name} · ${methodLabel(m.method)} / ${m.kind}` }))} /></div>
      <div><label htmlFor="upload-test-dataset">测试集</label><Select id="upload-test-dataset" aria-label="上传测试数据集" showSearch optionFilterProp="label" value={dataset?.id} placeholder="选择测试集" disabled={frozen} onChange={value => { setUploadId(value); setJob(undefined); setError(''); }} options={datasets.map(d => ({ value: d.id, label: `${d.name} · ${d.count} 张` }))} /></div>
      <Space wrap><DatasetUpload kind="test" disabled={frozen} onSaved={saved} /><DatasetUpload kind="test" single disabled={frozen} onSaved={saved} /></Space>
    </div>
    {error && <Alert type="error" showIcon title={error} action={<Button onClick={() => setRefresh(n => n + 1)}>重新读取</Button>} />}
    {!models.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无上传训练集的已保存模型"><Button href="/?view=train">新建训练</Button></Empty>}
    <Space><Button type="primary" icon={<ScanOutlined />} disabled={frozen || !model || !dataset} loading={busy} onClick={() => void run()}>开始测试</Button>{job && <Button onClick={() => open(job.id)}>{terminal(job.status) ? '查看记录' : '查看任务 / 停止'}</Button>}</Space>
    {job && <UploadedResults job={job} />}
  </div>;
}
