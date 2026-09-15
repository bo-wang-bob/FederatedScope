import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, Button, Card, Empty, Modal, Select, Space, Spin, Tag, Tooltip } from 'antd';
import { DownloadOutlined, ExpandOutlined, ReloadOutlined, SafetyCertificateOutlined, ScanOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { api, backdoorImageUrl, key, percent, terminal, type BackdoorJob, type BackdoorPick, type BackdoorStat, type BackdoorTestset } from './api';
import './backdoor.css';

const MAX_IDS = 20;
const IMAGE_KEYS = ['clean', 'triggered', 'defense'] as const;
type ImageKey = typeof IMAGE_KEYS[number];

const imageMeta: Record<ImageKey, { title: string; caption: string; tone: string }> = {
  clean: { title: '干净样本', caption: '无防御模型 · 未注入触发器', tone: 'baseline' },
  triggered: { title: '注入触发器', caption: '无防御模型 · 触发器已生效', tone: 'attack' },
  defense: { title: '防御后', caption: '防御模型 · 同样注入触发器', tone: 'defense' },
};

function readable(name: string) { return name.replaceAll('_', ' '); }

function StatValue({ stat, emphasis }: { stat?: BackdoorStat; emphasis?: 'attack' | 'defense' }) {
  if (!stat) return <b>—</b>;
  return <b className={emphasis ? 'backdoor-emphasis ' + emphasis : undefined}>{percent(stat.asrRate)}</b>;
}

export function BackdoorLab() {
  const [testset, setTestset] = useState<BackdoorTestset>();
  const [loadError, setLoadError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [count, setCount] = useState(MAX_IDS);
  const [domain, setDomain] = useState<string>();
  const [label, setLabel] = useState<number>();
  const [seed, setSeed] = useState(1);
  const [candidates, setCandidates] = useState<{ id: string; label: number }[]>([]);
  const [chosen, setChosen] = useState<string[]>([]);
  const [picking, setPicking] = useState(false);
  const [job, setJob] = useState<BackdoorJob>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState<ImageKey>();
  const pending = useRef<string | undefined>(undefined);
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);

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
    if (!pending.current) pending.current = key();
    try {
      const created = await api<BackdoorJob>('backdoor/jobs', { ids, name: `后门对比 · ${ids.length} 张`, idempotencyKey: pending.current });
      pending.current = undefined;
      if (alive.current) setJob(created);
    } catch (e) { if (alive.current) setError(String((e as Error).message || e)); }
    finally { if (alive.current) setSubmitting(false); }
  };

  if (loadError) return <Alert className="studio-connection-alert" type="error" showIcon title="后门研究接口不可用" description={loadError} action={<Button icon={<ReloadOutlined />} onClick={() => setRefresh(v => v + 1)}>重试</Button>} />;
  if (!testset) return <div className="studio-loading"><Spin size="large" /></div>;
  if (!testset.exported) return <div className="studio-empty-state"><SafetyCertificateOutlined /><h2>测试集尚未导出</h2><p>{testset.message || '后端未找到测试集图片目录'}</p><p className="platform-muted">先在服务器执行一次绘图脚本，导出测试集图片后再回到本页。</p></div>;

  const stats = result?.stats || {};
  return <div className="backdoor-lab">
    <div className="platform-stats backdoor-summary">
      <div className="platform-stat"><span>测试集样本</span><strong>{testset.total.toLocaleString()}</strong><small>{testset.domains.length} 个测试域 · {testset.classNames.length} 类</small></div>
      <div className="platform-stat"><span>本次选择</span><strong>{ids.length}<small> / {MAX_IDS}</small></strong><small>点击缩略图可增删</small></div>
      <div className="platform-stat"><span>攻击 / 防御实验</span><strong style={{ fontSize: 15 }}>{testset.runs.attack || '—'}</strong><small>{testset.runs.defense || '未找到防御实验'}</small></div>
      <div className="platform-stat"><span>目标类别</span><strong style={{ fontSize: 18 }}>{result ? readable(result.targetName) : '—'}</strong><small>{result ? `${result.attackName} 触发器` : '生成后显示'}</small></div>
    </div>

    <Card className="platform-panel backdoor-picker" title={<span><ScanOutlined /> 选择测试图片</span>}
      extra={<Space><Select aria-label="图片数量" value={count} disabled={busy} onChange={setCount} options={[5, 10, 15, 20].map(value => ({ value, label: value + ' 张' }))} />
        <Button icon={<ReloadOutlined />} disabled={busy || picking} onClick={reroll}>换一批</Button></Space>}>
      <div className="backdoor-filters">
        <label><span>测试域</span><Select aria-label="测试域" placeholder="全部域" allowClear value={domain} disabled={busy} onChange={value => { setDomain(value); setLabel(undefined); }} options={testset.domains.map(item => ({ value: item.name, label: `${item.name} (${item.count})` }))} /></label>
        <label><span>类别</span><Select aria-label="类别" placeholder="全部类别" allowClear showSearch optionFilterProp="label" value={label} disabled={busy} onChange={setLabel} options={testset.labels.map(item => ({ value: item.index, label: `${readable(item.name)} (${item.count})` }))} /></label>
        <span className="platform-muted">随机抽样自 {testset.total.toLocaleString()} 张测试图片；编号格式 {testset.domains[0]?.name ?? 'Art'}_00001</span>
      </div>
      <div className="backdoor-thumbnails" aria-busy={picking}>
        {picking ? <div className="backdoor-loading"><Spin /></div> : candidates.map(item => <button key={item.id} className={chosen.includes(item.id) ? 'selected' : ''} disabled={busy}
          aria-pressed={chosen.includes(item.id)} aria-label={`样本 ${item.id} · ${readable(nameOf(item.label))}`} onClick={() => toggle(item.id)}>
          <img src={backdoorImageUrl(item.id)} alt={`测试样本 ${item.id}`} loading="lazy" /><span>{readable(nameOf(item.label))}</span></button>)}
      </div>
      <div className="backdoor-submit">
        <span className="platform-muted">已选 <b>{ids.length}</b> 张，最多 {MAX_IDS} 张；编号将写入后端的 ids 文件并驱动绘图脚本。</span>
        <Button type="primary" size="large" icon={<ThunderboltOutlined />} disabled={busy || !ids.length} loading={busy} onClick={() => void generate()}>{busy ? '正在生成对比' : '生成三连对比'}</Button>
      </div>
      {error && <Alert type="error" showIcon title={error} />}
    </Card>

    {job && <Card className="platform-panel backdoor-output" title={<span><SafetyCertificateOutlined /> 对比结果</span>}
      extra={<Tag color={job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : 'processing'}>{job.stage}</Tag>}>
      {job.status !== 'completed' ? <div className="backdoor-progress">{job.status === 'failed' ? <><h3>生成失败</h3><p role="alert">{job.error || '请查看日志'}</p></> : <><Spin size="large" /><h3>{job.stage}</h3><p className="platform-muted">加载模型并对 {job.ids.length} 张图片做三次前向推理</p></>}</div> : result && <>
        <div className="backdoor-figures">
          {IMAGE_KEYS.map(imageKey => {
            const url = job.images?.[imageKey];
            const stat = imageKey === 'clean' ? stats.clean : imageKey === 'triggered' ? stats.triggered : stats.defense;
            return <figure key={imageKey} className={'backdoor-figure ' + imageMeta[imageKey].tone}>
              <div className="backdoor-figure-head"><h3>{imageMeta[imageKey].title}</h3>{url && <Tooltip title="查看大图"><Button type="text" aria-label={'放大' + imageMeta[imageKey].title} icon={<ExpandOutlined />} onClick={() => setExpanded(imageKey)} /></Tooltip>}</div>
              <div className="backdoor-figure-frame">{url ? <img src={url} alt={imageMeta[imageKey].title + '预测结果'} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="未生成" />}</div>
              <figcaption>
                <span>{imageMeta[imageKey].caption}</span>
                <div>{stat && <><i>攻击成功率</i><StatValue stat={stat} emphasis={imageKey === 'triggered' ? 'attack' : imageKey === 'defense' ? 'defense' : undefined} /></>}</div>
                <div>{stat && <><i>分类准确率</i><b>{percent(stat.accuracy)}</b></>}</div>
                {imageKey === 'triggered' && result && <small>目标类别 {readable(result.targetName)} · {stat?.asr ?? 0}/{stat?.total ?? 0} 张被劫持</small>}
                {imageKey === 'defense' && result && <small>防御后仍有 {stat?.asr ?? 0}/{stat?.total ?? 0} 张被劫持</small>}
              </figcaption>
            </figure>;
          })}
        </div>
        <div className="backdoor-verdict">
          <p>{result.runs.attack?.display || '攻击模型'} 在干净样本上准确率 {percent(stats.clean?.accuracy)}；注入 {result.attackName} 触发器后，{stats.triggered?.asr ?? 0}/{stats.triggered?.total ?? 0} 张被预测为 {readable(result.targetName)}（ASR {percent(stats.triggered?.asrRate)}），自身准确率降至 {percent(stats.triggered?.accuracy)}。</p>
          {stats.defense && <p>启用防御后，同一批触发器样本只有 {stats.defense.asr}/{stats.defense.total} 张被劫持（ASR {percent(stats.defense.asrRate)}），准确率回到 {percent(stats.defense.accuracy)}。</p>}
        </div>
        <div className="backdoor-table-head"><h3>逐样本对照</h3><Space>{job.images && <Button type="link" href={job.images.triggered} target="_blank" icon={<DownloadOutlined />}>下载触发器图</Button>}</Space></div>
        <div className="backdoor-table" role="table">
          <div role="row" className="backdoor-row head"><span>样本</span><span>真实标签</span><span>干净样本</span><span>注入触发器</span><span>防御后</span></div>
          {result.images.map(row => <div role="row" className="backdoor-row" key={row.id}>
            <span><img src={backdoorImageUrl(row.id)} alt={row.id} loading="lazy" /><code>{row.id}</code></span>
            <span>{readable(row.labelName)}</span>
            <span className={row.clean.label === row.label ? 'ok' : 'warn'}>{readable(row.clean.name)}</span>
            <span className={row.triggered.hit ? 'hit' : row.triggered.label === row.label ? 'ok' : 'warn'}>{readable(row.triggered.name)}{row.triggered.hit && <Tag color="error">劫持</Tag>}</span>
            <span className={row.defense ? row.defense.hit ? 'hit' : row.defense.label === row.label ? 'ok' : 'warn' : undefined}>{row.defense ? readable(row.defense.name) : '—'}</span>
          </div>)}
        </div>
        <p className="platform-muted backdoor-ids">编号：{result.ids.join(' ')}</p>
      </>}
    </Card>}

    <Modal open={!!expanded} title={expanded ? imageMeta[expanded].title : ''} onCancel={() => setExpanded(undefined)} footer={null} width={1100} className="design-modal">
      {expanded && job?.images?.[expanded] && <div className="backdoor-full-frame"><img src={job.images[expanded]} alt={imageMeta[expanded].title} /></div>}
    </Modal>
  </div>;
}
