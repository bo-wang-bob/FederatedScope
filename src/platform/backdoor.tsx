import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Alert, Button, Card, Empty, Modal, Select, Space, Spin, Tag, Tooltip } from 'antd';
import { ArrowRightOutlined, ExpandOutlined, ExperimentOutlined, ReloadOutlined, SafetyCertificateOutlined, ScanOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { api, backdoorImageUrl, key, terminal, type BackdoorJob, type BackdoorPick, type BackdoorTestset } from './api';
import { IMAGE_KEYS, metaFor, readable, type ImageKey } from './backdoorShared';
import { backdoorCompareHref } from './navigation';
import { BackdoorTrainingPanel } from './backdoorTraining';
import { DatasetUpload, type UploadedDataset } from './datasetUpload';
import './backdoor.css';

const MAX_IDS = 20;

export function BackdoorLab() {
  const [query, setQuery] = useSearchParams();
  const requestedJob = query.get('job') || undefined;
  // 页面拆成「训练实验 / 对比测试」两个可切换区域。带 ?job= 的分享链接
  // 直接落在对比结果上, 其余情况先进训练实验。
  const [tab, setTab] = useState<'train' | 'compare'>(requestedJob ? 'compare' : 'train');
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
  // 无触发器攻击 (label_flip: 数据投毒 / 模型投毒) 不产生 triggered 图,
  // 第三格改用后端的 defenseClean 图 (两个模型看的是同一批干净图)。
  const meta = metaFor(result?.triggerless);
  const figureUrl = (imageKey: ImageKey) => result?.triggerless && imageKey === 'defense'
    ? job?.images?.defenseClean : job?.images?.[imageKey];
  const busy = submitting || !!job && !terminal(job.status);
  const classNames = testset?.classNames || result?.classNames || [];
  const nameOf = (index: number) => index === -1 ? '未标注' : index < classNames.length ? classNames[index] : String(index);

  // 上传测试集后立即应用到当前结果组: 重建挑图/对比用的测试集图片
  const applyTestset = async (dataset: UploadedDataset) => {
    if (dataset.kind !== 'test') return;
    try {
      await api<{ count: number }>('backdoor/testset/apply', { datasetId: dataset.id });
      setError('');
      setRefresh(value => value + 1);
    } catch (e) { setError(String((e as Error).message || e)); }
  };

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

  // 训练实验区常驻在「训练实验」标签下: 上传数据集 -> 启动三个实验 -> 自动导出
  // 测试集, 完成后刷新本页测试集。「对比测试」标签保留原来的挑图 + 三连对比内容。
  return <div className="backdoor-workbench">
    <header className="backdoor-workbench-header">
      <div role="tablist" aria-label="后门攻防功能">
        <button role="tab" aria-selected={tab === 'train'} onClick={() => setTab('train')}><ExperimentOutlined /><span>训练实验</span></button>
        <button role="tab" aria-selected={tab === 'compare'} onClick={() => setTab('compare')}><ScanOutlined /><span>对比测试</span></button>
      </div>
    </header>
    <section hidden={tab !== 'train'}>
      <BackdoorTrainingPanel onCompleted={() => setRefresh(value => value + 1)} onOpenCompare={() => setTab('compare')} />
    </section>
    <section hidden={tab !== 'compare'} className="backdoor-lab">
      {!testset ? (
        <div className="studio-loading"><Spin size="large" /></div>
      ) : !testset.exported ? (
        <div className="studio-empty-state">
          <SafetyCertificateOutlined />
          {testset.runs.attack ? <>
            <h2>尚未上传测试集</h2>
            <p>对比测试将在你上传的测试集上进行，而不是后端自动划分的数据。</p>
            <p className="platform-muted">请上传按 类别/图片 组织的图片文件夹，类别名需与训练集一致；不上传则无法进行测试。</p>
            <DatasetUpload kind="test" onSaved={dataset => void applyTestset(dataset)} />
            {error && <Alert className="backdoor-training-error" type="error" showIcon title={error} />}
          </> : <>
            <h2>还没有可测试的实验</h2>
            <p>{testset.message || '还没有攻防实验结果'}</p>
            <p className="platform-muted">在「训练实验」里选择数据集并启动训练，三个攻防实验完成后上传测试集即可开始测试。</p>
          </>}
        </div>
      ) : (
        <>
          <Card className="platform-panel backdoor-picker" title={<span><ScanOutlined /> 选择测试图片</span>}
            extra={<Space>
              {testset.testset && <Tag color="blue" title={`测试集 ${testset.testset.name}`}>{testset.testset.name} · {testset.testset.count} 张{testset.testset.skipped ? ` · 跳过 ${testset.testset.skipped}` : ''}{testset.testset.unlabelled ? ` · 未标注 ${testset.testset.unlabelled}` : ''}</Tag>}
              <DatasetUpload kind="test" onSaved={dataset => void applyTestset(dataset)} />
              <Select aria-label="图片数量" value={count} disabled={busy} onChange={setCount} options={[5, 10, 15, 20].map(value => ({ value, label: value + ' 张' }))} />
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
                  const url = figureUrl(imageKey);
                  const missing = result.triggerless && imageKey === 'triggered' ? '该攻击无触发器' : '未生成';
                  return <figure key={imageKey} className={'backdoor-figure ' + meta[imageKey].tone}>
                    <div className="backdoor-figure-head"><h3>{meta[imageKey].title}</h3>{url && <Tooltip title="查看大图"><Button type="text" aria-label={'放大' + meta[imageKey].title} icon={<ExpandOutlined />} onClick={() => setExpanded(imageKey)} /></Tooltip>}</div>
                    <div className="backdoor-figure-frame">{url ? <img src={url} alt={meta[imageKey].title + '预测结果'} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={missing} />}</div>
                    <figcaption><span>{meta[imageKey].caption}</span></figcaption>
                  </figure>;
                })}
              </div>
              {result.triggerless && <Alert className="backdoor-triggerless" type="info" showIcon
                title={`${result.attackName} 属于无触发器攻击`}
                description="该攻击通过污染标签/数据统计实现，没有可注入的触发器，因此只给出攻击模型与防御模型在同一批干净测试图上的预测对照。" />}
              <div className="backdoor-submit">
                <Link to={backdoorCompareHref(job.id)}><Button type="primary">查看逐样本对照 <ArrowRightOutlined /></Button></Link>
              </div>
            </>}
          </Card>}

          <Modal open={!!expanded} title={expanded ? meta[expanded].title : ''} onCancel={() => setExpanded(undefined)} footer={null} width={1100} className="design-modal">
            {expanded && figureUrl(expanded) && <div className="backdoor-full-frame"><img src={figureUrl(expanded)} alt={meta[expanded].title} /></div>}
          </Modal>
        </>
      )}
    </section>
  </div>;
}
