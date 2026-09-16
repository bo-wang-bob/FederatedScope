import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Alert, Button, Card, Empty, Select, Space, Spin, Tag } from 'antd';
import { ReloadOutlined, UnorderedListOutlined } from '@ant-design/icons';
import { api, backdoorImageUrl, statusText, terminal, type BackdoorJob } from './api';
import { readable } from './backdoorShared';
import { viewHref } from './navigation';
import './backdoor.css';

const statusColor = (status: string) => status === 'completed' ? 'success' : status === 'failed' ? 'error' : 'processing';

// 逐样本对照：从后门研究页面迁出，任务可复查、可分享（URL 带 job 编号）。
export function BackdoorCompare() {
  const [query, setQuery] = useSearchParams();
  const requested = query.get('job');
  const [jobs, setJobs] = useState<BackdoorJob[]>([]);
  const [job, setJob] = useState<BackdoorJob>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const selected = requested || jobs.find(item => item.status === 'completed')?.id || jobs[0]?.id;

  useEffect(() => {
    let active = true;
    setLoading(true);
    void api<BackdoorJob[]>('backdoor/jobs').then(list => { if (active) { setJobs(list); setError(''); } })
      .catch(e => { if (active) setError(String((e as Error).message || e)); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [refresh]);

  useEffect(() => {
    let active = true, busy = false, timer: ReturnType<typeof setInterval> | undefined;
    setJob(undefined);
    if (!selected) return;
    const load = async () => {
      if (busy) return;
      busy = true;
      try {
        const next = await api<BackdoorJob>('backdoor/jobs/' + selected);
        if (!active) return;
        setJob(next); setError('');
        if (terminal(next.status) && timer) clearInterval(timer);
      } catch (e) { if (active) setError(String((e as Error).message || e)); }
      finally { busy = false; }
    };
    timer = setInterval(() => { void load(); }, 1500);
    void load();
    return () => { active = false; if (timer) clearInterval(timer); };
  }, [selected, refresh]);

  const result = job?.status === 'completed' ? job.result : undefined;
  return <div className="backdoor-lab">
    <Card className="platform-panel backdoor-compare" title={<span><UnorderedListOutlined /> 逐样本对照</span>}
      extra={<Space>{job && <Tag color={statusColor(job.status)}>{job.stage}</Tag>}
        <Select aria-label="对比任务" placeholder="选择对比任务" style={{ minWidth: 260 }} value={selected} disabled={!jobs.length}
          onChange={(value: string) => setQuery({ view: 'backdoorCompare', job: value })}
          options={jobs.map(item => ({ value: item.id, label: `${item.name || '后门对比'} · ${item.ids.length} 张 · ${statusText[item.status] || item.status}` }))} />
        <Button icon={<ReloadOutlined />} onClick={() => setRefresh(value => value + 1)}>刷新</Button>
      </Space>}>
      {error && <Alert className="backdoor-compare-alert" type="error" showIcon title={error}
        action={<Button icon={<ReloadOutlined />} onClick={() => setRefresh(value => value + 1)}>重试</Button>} />}
      {loading && !job && <div className="backdoor-loading"><Spin /></div>}
      {!loading && !jobs.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<span>还没有对比任务，先到<Link to={viewHref('backdoor')}>后门研究</Link>生成三连对比。</span>} />}
      {job && <>
        {job.status !== 'completed' ? <div className="backdoor-progress">{terminal(job.status) ? <><h3>{job.stage}</h3>{job.error && <p role="alert">{job.error}</p>}</> : <><Spin size="large" /><h3>{job.stage}</h3><p className="platform-muted">正在计算 {job.ids.length} 张图片的攻防预测</p></>}</div> : result && <>
          <h3 className="backdoor-table-title">逐样本预测对照</h3>
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
        </>}
      </>}
    </Card>
  </div>;
}
