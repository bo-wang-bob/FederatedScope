import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Alert, Button, Card, Empty, Modal, Select, Space, Spin, Tag, Tooltip } from 'antd';
import { ArrowRightOutlined, ExpandOutlined, ReloadOutlined, SafetyCertificateOutlined, ScanOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { api, backdoorImageUrl, key, terminal, type BackdoorJob, type BackdoorPick, type BackdoorTestset } from './api';
import { IMAGE_KEYS, imageMeta, readable, type ImageKey } from './backdoorShared';
import { backdoorCompareHref } from './navigation';
import { BackdoorTrainingPanel } from './backdoorTraining';
import './backdoor.css';

const MAX_IDS = 20;

export function BackdoorLab() {
  const [query, setQuery] = useSearchParams();
  const requestedJob = query.get('job') || undefined;
  const [testset, setTestset] = useState<BackdoorTestset>();
  const [loadError, setLoadError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [count, setCount] = useState(MAX_IDS);
  const [domain, setDomain] = useState<string>();
  const [label, setLabel] = useState<number>();
  const [seed, setSeed] = useState(1);
  const [candidates, setCandidates] = useState<{ id: string; label: number }[]>([]);
  const [pickInfo, setPickInfo] = useState<BackdoorPick>();
  const [chosen, setChosen] = useState<string[]>([]);
  const [picking, setPicking] = useState(false);
  const [job, setJob] = useState<BackdoorJob>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState<ImageKey>();
  const pending = useRef<{signature:string;key:string} | undefined>(undefined);
  const alive = useRef(true);
  const ownJob = useRef<string | undefined>(undefined);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);

  // 地址栏带 job 时（复制分享链接或刷新）恢复那次对比结果；本地刚提交的任务不需要再拉一次。
  useEffect(() => {
    let active = true;
    if (!requestedJob || ownJob.current === requestedJob) return;
    api<BackdoorJob>('backdoor/jobs/' + requestedJob)
      .then(existing => { if (active) { setJob(existing); ownJob.current = existing.id; setError(''); } })
      .catch(() => { if (active) setQuery({ view: 'backdoor' }, { replace: true }); });
    return () => { active = false; };
  }, [requestedJob, setQuery]);

  useEffect(() => {
    let active = true;
    setLoadError('');
    void api<BackdoorTestset>('backdoor/testset').then(data => { if (active) setTestset(data); })
      .catch(e => { if (active) setLoadError(String((e as Error).message || e)); });
    return () => { active = false; };
  }, [refresh]);

  useEffect(() => {
    let active = true;
    if (!testset?.exported) return;
    setPicking(true);
    void api<BackdoorPick>('backdoor/pick', { count, ...(domain ? { domain } : {}), ...(label != null ? { label } : {}), seed })
      .then(data => {
        if (!active) return;
        setCandidates(data.ids.map((id, index) => ({ id, label: data.labels[index] })));
        setPickInfo(data);
        setChosen(data.ids);
        setError('');
      })
      .catch(e => { if (active) setError(String((e as Error).message || e)); })
      .finally(() => { if (active) setPicking(false); });
    return () => { active = false; };
  }, [testset?.exported, count, domain, label, seed]);

  useEffect(() => {
    if (!job || terminal(job.status)) return;
    let active = true, polling = false;
    const timer = setInterval(async () => {
      if (polling) return;
      polling = true;
      try { const next = await api<BackdoorJob>('backdoor/jobs/' + job.id); if (active) setJob(next); }
      catch (e) { if (active) setError(String((e as Error).message || e)); }
      finally { polling = false; }
    }, 1200);
    return () => { active = false; clearInterval(timer); };
  }, [job?.id, job?.status]);

  const toggle = (id: string) => setChosen(old => old.includes(id) ? old.filter(x => x !== id) : old.length >= MAX_IDS ? old : [...old, id]);
  const reroll = useCallback(() => setSeed(value => value + 1), []);
  const ids = candidates.filter(item => chosen.includes(item.id)).map(item => item.id);
  const result = job?.status === 'completed' ? job.result : undefined;
  const busy = submitting || !!job && !terminal(job.status);
  const classNames = testset?.classNames || result?.classNames || [];
  const nameOf = (index: number) => index < classNames.length ? classNames[index] : String(index);

  const generate = async () => {
    if (!ids.length || busy) return;
    setSubmitting(true); setError(''); setJob(undefined);
    const signature=JSON.stringify(ids);
    if (pending.current?.signature !== signature) pending.current = {signature,key:key()};
    try {
      const created = await api<BackdoorJob>('backdoor/jobs', { ids, name: `后门对比 · ${ids.length} 张`, idempotencyKey: pending.current!.key });
      pending.current = undefined;
      if (alive.current) {
        ownJob.current = created.id;
        setJob(created);
        setQuery({ view: 'backdoor', job: created.id }, { replace: true });
      }
    } catch (e) { if (alive.current) setError(String((e as Error).message || e)); }
    finally { if (alive.current) setSubmitting(false); }
  };

  if (loadError) return <Alert className="studio-connection-alert" type="error" showIcon title="后门防御接口不可用" description={loadError} action={<Button icon={<ReloadOutlined />} onClick={() => setRefresh(v => v + 1)}>重试</Button>} />;

  // 训练面板常驻顶部: 上传数据集 -> 启动三个实验 -> 自动导出测试集, 完成后刷新本页测试集。
  return <div className="backdoor-lab">
    <BackdoorTrainingPanel onCompleted={() => setRefresh(value => value + 1)} />
    {!testset ? <div className="studio-loading"><Spin size="large" /></div> : !testset.exported ?
      <div className="studio-empty-state"><SafetyCertificateOutlined /><h2>测试集尚未导出</h2><p>{testset.message || '还没有可对比的攻防实验结果'}</p><p className="platform-muted">在上方选择数据集并启动训练，完成后会自动导出测试集图片。</p></div> : <>
    <Card className="platform-panel backdoor-picker" title={<span><ScanOutlined /> 选择测试图片</span>}
      extra={<Space><Select aria-label="图片数量" value={count} disabled={busy} onChange={setCount} options={[5, 10, 15, 20].map(value => ({ value, label: value + ' 张' }))} />
        <Button icon={<ReloadOutlined />} disabled={busy || picking} onClick={reroll}>换一批</Button></Space>}>
      <div className="backdoor-filters">
        <label><span>测试域</span><Select aria-label="测试域" placeholder="全部域" allowClear value={domain} disabled={busy} onChange={value => { setDomain(value); setLabel(undefined); }} options={testset.domains.map(item => ({ value: item.name, label: `${item.name} (${item.count})` }))} /></label>
        <label><span>类别</span><Select aria-label="类别" placeholder="全部类别" allowClear showSearch optionFilterProp="label" value={label} disabled={busy} onChange={setLabel} options={testset.labels.map(item => ({ value: item.index, label: `${readable(item.name)} (${item.count})` }))} /></label>
        <span className="platform-muted">{pickInfo?.filtered ? '加权展示样本 · 不代表整体指标' : '随机抽样'} · 编号格式 {testset.domains[0]?.name ?? 'Art'}_00001</span>
      </div>
      <div className="backdoor-thumbnails" aria-busy={picking}>
        {picking ? <div className="backdoor-loading"><Spin /></div> : candidates.map(item => <button key={item.id} className={chosen.includes(item.id) ? 'selected' : ''} disabled={busy}
          aria-pressed={chosen.includes(item.id)} aria-label={`样本 ${item.id} · ${readable(nameOf(item.label))}`} onClick={() => toggle(item.id)}>
          <img src={backdoorImageUrl(item.id)} alt={`测试样本 ${item.id}`} loading="lazy" /><span>{readable(nameOf(item.label))}</span></button>)}
      </div>
      <div className="backdoor-submit">
        {job && !terminal(job.status) && <Button disabled={job.status==='stopping'} onClick={() => void api<BackdoorJob>('backdoor/jobs/'+job.id+'/stop',{}).then(next => { if(alive.current) setJob(next); }).catch(e=>setError(String(e.message)))}>停止生成</Button>}
        <Button type="primary" size="large" icon={<ThunderboltOutlined />} disabled={busy || !ids.length} loading={busy} onClick={() => void generate()}>{busy ? '正在生成对比' : '生成对比'}</Button>
      </div>
      {error && <Alert type="error" showIcon title={error} />}
    </Card>

    {job && <Card className="platform-panel backdoor-output" title={<span><SafetyCertificateOutlined /> 对比结果</span>}
      extra={<Tag color={job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : 'processing'}>{job.stage}</Tag>}>
      {job.status !== 'completed' ? <div className="backdoor-progress">{terminal(job.status) ? <><h3>{job.stage}</h3>{job.error && <p role="alert">{job.error}</p>}</> : <><Spin size="large" /><h3>{job.stage}</h3><p className="platform-muted">正在计算 {job.ids.length} 张图片的攻防预测</p></>}</div> : result && <>
        <div className="backdoor-figures">
          {IMAGE_KEYS.map(imageKey => {
            const url = job.images?.[imageKey];
            return <figure key={imageKey} className={'backdoor-figure ' + imageMeta[imageKey].tone}>
              <div className="backdoor-figure-head"><h3>{imageMeta[imageKey].title}</h3>{url && <Tooltip title="查看大图"><Button type="text" aria-label={'放大' + imageMeta[imageKey].title} icon={<ExpandOutlined />} onClick={() => setExpanded(imageKey)} /></Tooltip>}</div>
              <div className="backdoor-figure-frame">{url ? <img src={url} alt={imageMeta[imageKey].title + '预测结果'} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="未生成" />}</div>
              <figcaption><span>{imageMeta[imageKey].caption}</span></figcaption>
            </figure>;
          })}
        </div>
        <div className="backdoor-submit">
          <Link to={backdoorCompareHref(job.id)}><Button type="primary">查看逐样本对照 <ArrowRightOutlined /></Button></Link>
        </div>
      </>}
    </Card>}

    <Modal open={!!expanded} title={expanded ? imageMeta[expanded].title : ''} onCancel={() => setExpanded(undefined)} footer={null} width={1100} className="design-modal">
      {expanded && job?.images?.[expanded] && <div className="backdoor-full-frame"><img src={job.images[expanded]} alt={imageMeta[expanded].title} /></div>}
    </Modal>
    </>}
  </div>;
}
