import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Empty, Form, Input, InputNumber, Popconfirm, Progress, Select, Space, Steps, Table, Tabs, Tag } from 'antd';
import { ExperimentOutlined, ReloadOutlined, SafetyCertificateOutlined, StopOutlined } from '@ant-design/icons';
import { api, key, percent, terminal, type Catalog, type Job, type RequestConfig } from './api';
import { Curves, LossChart } from './charts';
import { Panel, State, Stat } from './ui';
import './privacyLab.css';

type PrivacyRequest = Pick<RequestConfig, 'group' | 'method' | 'name' | 'rounds' | 'clientCount'> & { defense: boolean };
type PrivacyCatalog = Omit<Catalog, 'groups'> & { groups: { id: string; dataset: string; backbone: string; domains: number; datasetFound: boolean; methods: { id: string; label: string; enabled: boolean; defaults: PrivacyRequest; defenseDefaults: PrivacyRequest }[] }[] };
type PrivacyJob = Omit<Job, 'request'> & { request: PrivacyRequest; events?: { type: string }[]; featureStorage?: { files: number; rounds: number[]; directory: string; resultsReady: boolean }; storageError?: string };
type AttackMetric = { auc: number; tprAt1Fpr: number; actualFpr: number; threshold: number; members: number; nonmembers: number };
type AttackSample = { domain: string; className: string; scores: Record<string, number>; predictions: Record<string, 'member' | 'nonmember'> };
type Results = { dataset: string; defense: boolean; clientId: number; clientIds: number[]; thresholdPolicy: string; completeTraining?: boolean;
  indexedWarning: string | null; rounds: number[]; metrics: Record<string, AttackMetric>; samples: Record<string, AttackSample[]> };
const path = 'privacy/experiments/';
const label = (value: string) => value === 'member' ? '成员' : '非成员';

export function PrivacyLab() {
  const [tab, setTab] = useState<'train' | 'results'>('train');
  return <div className="privacy-workbench">
    <header className="privacy-workbench-header">
      <div role="tablist" aria-label="隐私研究功能">
        <button role="tab" aria-selected={tab === 'train'} onClick={() => setTab('train')}><ExperimentOutlined /><span>训练实验</span></button>
        <button role="tab" aria-selected={tab === 'results'} onClick={() => setTab('results')}><SafetyCertificateOutlined /><span>结果展示</span></button>
      </div>
    </header>
    <PrivacyExperiments stageTab={tab} onStageChange={setTab} />
  </div>;
}

function PrivacyExperiments({ stageTab, onStageChange }: { stageTab: 'train' | 'results'; onStageChange: (tab: 'train' | 'results') => void }) {
  const [catalog, setCatalog] = useState<PrivacyCatalog>();
  const [jobs, setJobs] = useState<PrivacyJob[]>([]);
  const [job, setJob] = useState<PrivacyJob>();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [preflight, setPreflight] = useState<PrivacyJob>();
  const [form] = Form.useForm<PrivacyRequest>();
  const selected = useRef<string | undefined>(undefined);
  const groupId = Form.useWatch('group', form);
  const group = catalog?.groups.find(item => item.id === groupId);
  const active = jobs.some(item => !terminal(item.status));
  const ready = preflight?.status === 'completed';

  useEffect(() => {
    let alive = true;
    void api<PrivacyCatalog>(path + 'catalog').then(value => {
      if (!alive) return;
      setCatalog(value);
      const initial = value.groups.find(item => item.id === 'military_cnn') || value.groups[0];
      if (initial) form.setFieldsValue(initial.methods[0].defaults);
    }).catch(e => { if (alive) setError((e as Error).message); });
    return () => { alive = false; };
  }, [form]);

  useEffect(() => {
    let alive = true, updating = false;
    const update = async () => {
      if (updating) return;
      updating = true;
      try {
        const list = await api<PrivacyJob[]>(path + 'jobs');
        if (!alive) return;
        setJobs(list);
        const id = selected.current || list.find(item => !terminal(item.status))?.id;
        if (id) {
          selected.current = id;
          const detail = await api<PrivacyJob>(path + 'jobs/' + id);
          if (!alive || selected.current !== id) return;
          setJob(detail);
          if (detail.action === 'inspect') setPreflight(previous => previous?.id === id ? detail : previous);
        }
      } catch (e) { if (alive) setError((e as Error).message); }
      finally { updating = false; }
    };
    void update();
    const timer = setInterval(() => void update(), 2500);
    return () => { alive = false; clearInterval(timer); };
  }, []);

  async function launch(train: boolean) {
    setBusy(true); setError('');
    try {
      const values = train ? preflight!.request : await form.validateFields();
      const task = await api<PrivacyJob>(path + (train ? 'train' : 'preflight'), {
        ...values, idempotencyKey: key(), ...(train && { preflightId: preflight!.id }),
      });
      selected.current = task.id; setJob(task);
      if (!train) { setPreflight(task); form.setFieldsValue(task.request); }
      else setPreflight(undefined);
      setJobs(previous => [task, ...previous.filter(item => item.id !== task.id)]);
    } catch (e) { setError((e as Error).message || '请检查实验参数'); }
    finally { setBusy(false); }
  }
  async function open(id: string) {
    selected.current = id;
    try {
      const detail = await api<PrivacyJob>(path + 'jobs/' + id);
      if (selected.current === id) {
        setJob(detail);
        if (detail.action === 'train' && detail.featureStorage?.resultsReady) onStageChange('results');
      }
    }
    catch (e) { setError((e as Error).message); }
  }
  async function stop() {
    if (!job) return;
    try { setJob(await api<PrivacyJob>(path + 'jobs/' + job.id + '/stop', {})); }
    catch (e) { setError((e as Error).message); }
  }
  return <section className="privacy-lab-experiments" aria-label="独立隐私攻击实验">
    {stageTab === 'train' && <div className="privacy-lab-intro"><div><h2>发起隐私实验</h2></div></div>}
    {error && <Alert type="error" showIcon closable title={error} onClose={() => setError('')} />}
    <div hidden={stageTab !== 'train'}>
    <div className="privacy-lab-workflow">
      <Panel title="实验配置" extra={<Tag>{ready ? '预检通过' : '待预检'}</Tag>}>
        <Form form={form} layout="vertical" disabled={!catalog || active || busy} onValuesChange={() => setPreflight(undefined)}>
          <div className="privacy-lab-form-grid">
            <Form.Item name="group" label="数据集" rules={[{ required: true }]}><Select options={catalog?.groups.map(g => ({ value: g.id, label: g.dataset }))} onChange={value => {
              const next = catalog!.groups.find(g => g.id === value)!; form.setFieldsValue(next.methods[0].defaults); setPreflight(undefined);
            }} /></Form.Item>
            <Form.Item name="method" label="训练方案" rules={[{ required: true }]}><Select options={group?.methods.map(m => ({ value: m.id, label: m.label, disabled: !m.enabled }))} onChange={value => {
              form.setFieldsValue(group!.methods.find(m => m.id === value)!.defaults); setPreflight(undefined);
            }} /></Form.Item>
            <Form.Item name="name" label="实验名称"><Input placeholder="为本次隐私实验命名" /></Form.Item>
            <Form.Item name="defense" label="防御配置"><Select options={[{ value: false, label: '无防御' }, { value: true, label: '有防御' }]} onChange={value => {
              const method = group?.methods.find(item => item.id === form.getFieldValue('method'));
              if (method) { const currentName = form.getFieldValue('name'); form.setFieldsValue(value ? method.defenseDefaults : method.defaults); form.setFieldValue('name', currentName); }
              setPreflight(undefined);
            }} /></Form.Item>
            <Form.Item name="rounds" label="训练轮数" rules={[{ required: true }]}><InputNumber min={1} max={1000} /></Form.Item>
            <Form.Item name="clientCount" label="客户端数量" rules={[{ required: true }]}><InputNumber min={group?.domains || 3} max={240} step={group?.domains || 3} /></Form.Item>
          </div>
        </Form>
        {group && !group.datasetFound && <Alert type="warning" showIcon title="尚未找到数据集" />}
        <Space className="privacy-lab-actions"><Button icon={<SafetyCertificateOutlined />} loading={busy && !ready} disabled={active || !catalog} onClick={() => void launch(false)}>预检资源与划分</Button>
          <Button type="primary" icon={<ExperimentOutlined />} loading={busy && !!ready} disabled={!ready || active} onClick={() => void launch(true)}>开始训练与攻击</Button></Space>
      </Panel>
      <div className="privacy-lab-monitor-column">
        {job ? <>
          <Panel title="训练进度" extra={<State value={job.status} />}>
            <div className="privacy-lab-task-heading"><strong>{job.request.name || job.id.slice(0, 8)}</strong><Tag>{job.request.defense ? '有防御' : '无防御'}</Tag></div>
            <Steps size="small" current={job.action === 'inspect' ? 0 : terminal(job.status) && job.status === 'completed' ? 3 : job.stage.includes('评测') || job.stage.includes('结果') ? 2 : 1} items={[{ title: '预检' }, { title: '训练 / 保存' }, { title: '攻击评测' }, { title: '完成' }]} />
            <Progress percent={Math.min(100, Math.round((job.metrics.at(-1)?.round || 0) / job.request.rounds * 100))} status={job.status === 'failed' ? 'exception' : job.status === 'completed' ? 'success' : 'active'} />
            <div className="privacy-lab-stats"><Stat label="评估轮次" value={`${job.metrics.at(-1)?.round || 0} / ${job.request.rounds}`} /><Stat label="当前准确率" value={percent(job.metrics.at(-1)?.accuracy)} /><Stat label="已保存攻击特征" value={job.featureStorage?.files ?? '—'} sub={job.featureStorage?.files ? `${job.featureStorage.rounds.length} 个保存轮次` : '等待保存'} /></div>
            {job.storageError && <Alert type="error" title="特征保存失败" description={job.storageError} />}
            {job.error && <Alert type="error" showIcon title={job.error} />}
            {job.data && <p className="privacy-lab-note">{job.data.clientCount} 个客户端 · {job.data.trainSamples} 训练样本 · {job.data.testSamples} 测试样本</p>}
            <Space wrap>{job.featureStorage?.resultsReady && <Button type="primary" onClick={() => onStageChange('results')}>查看攻击结果</Button>}{!terminal(job.status) && <Popconfirm title="停止本次隐私实验？已保存特征会保留。" onConfirm={() => void stop()}><Button danger icon={<StopOutlined />}>停止任务</Button></Popconfirm>}
              {terminal(job.status) && <Button icon={<ReloadOutlined />} onClick={() => { form.setFieldsValue(job.request); setPreflight(job.action === 'inspect' && job.status === 'completed' ? job : undefined); }}>载入此配置</Button>}</Space>
          </Panel>
          <Tabs items={[{ key: 'curves', label: '训练曲线', children: <Panel title="训练指标"><Curves points={job.metrics} /><LossChart points={job.metrics} /></Panel> },
            { key: 'clients', label: '客户端状态', children: <Table rowKey="id" dataSource={Object.values(job.clients)} size="small" pagination={{ pageSize: 8 }} columns={[{ title: '客户端', dataIndex: 'id' }, { title: '域', dataIndex: 'domain' }, { title: '样本数', dataIndex: 'samples' }, { title: '状态', dataIndex: 'stage' }, { title: '损失', dataIndex: 'loss', render: (v: number | undefined) => v?.toFixed(4) ?? '—' }]} /> },
            { key: 'logs', label: '执行日志', children: <PrivacyLogs id={job.id} /> },
          ]} />
        </> : <Panel title="训练进度" extra={<Tag>等待启动</Tag>}><div className="privacy-lab-await"><div className="privacy-lab-empty-icon"><ExperimentOutlined /></div><h3>准备开始训练</h3></div></Panel>}
      </div>
    </div>
    </div>
    <div hidden={stageTab !== 'results'}>
      <div className="privacy-results-stage">
      <Panel title="实验结果">
        <Table<PrivacyJob> rowKey="id" size="small" dataSource={jobs.filter(item => item.action === 'train')} pagination={{ pageSize: 6 }}
          rowClassName={item => 'privacy-experiment-row' + (job?.id === item.id ? ' is-selected' : '')}
          onRow={item => ({ tabIndex: 0, 'aria-label': '查看实验 ' + (item.request.name || item.id.slice(0, 8)), 'aria-selected': job?.id === item.id,
            onClick: () => void open(item.id),
            onKeyDown: event => { if (event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); void open(item.id); } },
          })} columns={[
          { title: '实验', render: (_, item) => <Button type="link" onClick={event => { event.stopPropagation(); void open(item.id); }}>{item.request.name || item.id.slice(0, 8)}</Button> },
          { title: '数据集', render: (_, item) => catalog?.groups.find(g => g.id === item.request.group)?.dataset || item.request.group },
          { title: '防御配置', render: (_, item) => item.request.defense ? '有防御' : '无防御' },
          { title: '已保存特征', render: (_, item) => `${item.featureStorage?.files ?? 0} 个文件` },
          { title: '状态', dataIndex: 'status', render: value => <State value={value} /> },
        ]} />
      </Panel>
      {job?.action === 'train' && job.featureStorage?.resultsReady
        ? <PrivacyResults key={job.id} id={job.id} />
        : <Panel title="攻击结果"><div className="privacy-lab-await"><ExperimentOutlined /><h3>请选择已完成实验</h3></div></Panel>}
      </div>
    </div>
  </section>;
}

function PrivacyLogs({ id }: { id: string }) {
  const [logs, setLogs] = useState('读取日志…');
  useEffect(() => {
    let alive = true, busy = false;
    const update = async () => { if (busy) return; busy = true; try { const value = await api<string>(path + 'jobs/' + id + '/logs'); if (alive) setLogs(value); } catch (e) { if (alive) setLogs((e as Error).message); } finally { busy = false; } };
    void update(); const timer = setInterval(() => void update(), 3000);
    return () => { alive = false; clearInterval(timer); };
  }, [id]);
  return <pre className="privacy-lab-logs" tabIndex={0}>{logs}</pre>;
}

function PrivacyResults({ id }: { id: string }) {
  const [client, setClient] = useState(1);
  const plugin = 'fedmia_ii';
  const [result, setResult] = useState<Results>();
  const [error, setError] = useState('');
  useEffect(() => { let alive = true; setResult(undefined); setError(''); void api<Results>(path + `jobs/${id}/results?clientId=${client}`).then(value => { if (alive) setResult(value); }).catch(e => { if (alive) setError((e as Error).message); }); return () => { alive = false; }; }, [id, client]);
  if (error) return <Alert type="error" title={error} />;
  if (!result) return <Empty description="读取已保存攻击结果…" />;
  const metric = result.metrics[plugin];
  return <Panel title="逐客户端攻击评测" extra={<Space><Select aria-label="攻击评测客户端" value={client} onChange={setClient} options={result.clientIds.map(value => ({ value, label: 'Client ' + value }))} /><Tag>FedMIA-II</Tag></Space>}>
    {result.completeTraining === false && <Alert type="warning" showIcon title="实验未完成" />}
    <div className="privacy-lab-stats"><Stat label="AUC" value={percent(metric.auc)} /><Stat label="TPR@FPR≤1%" value={percent(metric.tprAt1Fpr)} /><Stat label="实际 FPR" value={percent(metric.actualFpr)} /><Stat label="校准阈值" value={metric.threshold.toFixed(4)} /></div>
    <ScoreDistribution result={result} plugin={plugin} />
    <Tabs items={['member', 'nonmember'].map(group => ({ key: group, label: group === 'member' ? '客户端训练样本' : '非训练样本', children: <div className="privacy-lab-samples" role="region" tabIndex={0} aria-label={group === 'member' ? '训练样本攻击结果' : '非训练样本攻击结果'}>{result.samples[group].map((sample, index) => <article key={index}>
      <img loading="lazy" src={'/api/platform/' + path + `jobs/${id}/images/${client}/${group}/${index}`} alt={sample.className} />
      <div><strong>{sample.className}</strong><small>{sample.domain}</small><span>攻击预测：<b>{label(sample.predictions[plugin])}</b></span><span>攻击分数：{sample.scores[plugin].toFixed(4)}</span><Tag color={sample.predictions[plugin] === group ? 'success' : 'warning'}>{sample.predictions[plugin] === group ? '攻击判断正确' : '攻击判断错误'}</Tag></div>
    </article>)}</div> }))} />
  </Panel>;
}

function ScoreDistribution({ result, plugin }: { result: Results; plugin: string }) {
  const histograms = ['member', 'nonmember'].map(group => {
    const values = result.samples[group].map(sample => sample.scores[plugin]);
    const bins = Array<number>(25).fill(0);
    values.forEach(value => { bins[Math.min(24, Math.max(0, Math.floor(value * 25)))]++; });
    return { group, bins: bins.map(count => count / Math.max(1, values.length)), mean: values.reduce((a, b) => a + b, 0) / Math.max(1, values.length) };
  });
  const max = Math.max(.01, ...histograms.flatMap(item => item.bins));
  return <div className="privacy-lab-distribution"><header><strong>攻击分数分布</strong><span>均值差 {(histograms[0].mean - histograms[1].mean).toFixed(4)}</span></header>
    <svg viewBox="0 0 600 190" role="img" aria-label="真实图片攻击分数分布"><line x1="35" y1="160" x2="575" y2="160" stroke="#54756c" />
      {histograms.map((item, series) => item.bins.map((height, index) => <rect key={`${series}-${index}`} x={35 + index * 21.6} y={160 - height / max * 130} width={20} height={height / max * 130} fill={series ? '#65c7ac' : '#62afda'} opacity=".6" />))}
      <text x="35" y="182">0.0</text><text x="295" y="182">0.5</text><text x="555" y="182">1.0</text>
    </svg><footer><span>● 客户端训练样本</span><span>● 非训练样本</span></footer>
  </div>;
}
