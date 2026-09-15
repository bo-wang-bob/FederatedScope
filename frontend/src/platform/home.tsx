import { ArrowRightOutlined, ExperimentOutlined, FileSearchOutlined, LineChartOutlined, ScanOutlined, PlusOutlined } from '@ant-design/icons';
import { Badge, Button, Empty } from 'antd';
import { Link } from 'react-router-dom';
import { methodLabel, percent, terminal, type Catalog, type Job, type Library, type Resource } from './api';
import { Curves } from './charts';
import { FutureModuleEntries } from './extensions';
import { jobHref } from './navigation';
import { State } from './ui';

export function SystemHome({ catalog, jobs, library, resource, stale, overview }: {
  catalog?: Catalog; jobs: Job[]; library: Library; resource?: Resource; stale: boolean; overview?: Job;
}) {
  const active = jobs.find(job => !terminal(job.status));
  const recent = jobs.filter(job => job.action === 'train').slice(0, 4);
  const lastPoint = overview?.metrics?.at(-1);
  return <div className="studio-home">
    <div className="studio-welcome"><div><span className="studio-kicker">YOUR RESEARCH WORKSPACE</span><h1>开始一次新的探索<span>.</span></h1></div><Link className="ant-btn ant-btn-primary studio-new-link" to="/?view=train"><PlusOutlined />新建实验</Link></div>
    <div className="studio-home-stats" aria-label="服务器资产概览">{[
      { label: '已完成训练', value: jobs.filter(job => job.action === 'train' && job.status === 'completed').length, href: '/?view=jobs', icon: <ExperimentOutlined /> },
      { label: '已保存模型', value: library.models.length, href: '/?view=evaluate', icon: <ScanOutlined /> },
      { label: '测试集', value: library.testsets.length, href: '/?view=evaluate', icon: <FileSearchOutlined /> },
      { label: '缓存配置', value: catalog?.groups.filter(group => group.cacheFound).length, href: '/?view=cache', icon: <LineChartOutlined /> },
    ].map(stat => <Link className="studio-home-stat" key={stat.label} to={stat.href}><span className="studio-stat-icon">{stat.icon}</span><span><small>{stat.label}</small><strong>{catalog ? stat.value ?? '—' : '—'}</strong></span><ArrowRightOutlined /></Link>)}</div>
    <div className="studio-home-grid"><section className="studio-surface studio-latest"><div className="studio-section-head"><div><h2>{active ? '进行中的任务' : '最近训练'}</h2><span>{stale ? '离线快照' : active ? active.stage : overview?.request.group || '等待实验'}</span></div>{(active || overview) && <Link to={jobHref((active || overview)!.id)}>查看详情 <ArrowRightOutlined /></Link>}</div>
      {active && active.id !== overview?.id ? <div className="studio-active-notice"><Badge status={stale ? 'default' : 'processing'} /><strong>{active.request.name || active.id.slice(0, 8)}</strong><State value={active.status} /></div> : null}
      {overview ? <><div className="studio-latest-heading"><div><h3>{overview.request.name || overview.id.slice(0, 8)}</h3><span>{methodLabel(overview.request.method)} <i>·</i> {overview.request.group}</span></div><div><strong>{percent(lastPoint?.accuracy)}</strong><small>总体准确率</small></div></div><Curves points={overview.metrics || []} /></> : <div className="studio-home-empty"><Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={!catalog ? stale ? '服务器暂时离线' : '正在读取实验' : '还没有训练记录'} /><Button href="/?view=train">配置第一个实验</Button></div>}
    </section><div className="studio-home-aside"><section className="studio-surface"><div className="studio-section-head"><h2>快捷操作</h2></div><div className="studio-quick-actions">{[
      { href: 'train', label: '新建训练', sub: '算法与超参数', icon: <ExperimentOutlined /> },
      { href: 'experience', label: '单图预测', sub: '选择模型与样本', icon: <ScanOutlined /> },
      { href: 'evaluate', label: '独立评测', sub: '总体与分域指标', icon: <FileSearchOutlined /> },
      { href: 'compare', label: '结果对比', sub: '多实验分析', icon: <LineChartOutlined /> },
    ].map(item => <Link to={`/?view=${item.href}`} key={item.href}><span className="studio-quick-icon">{item.icon}</span><span><strong>{item.label}</strong><small>{item.sub}</small></span><ArrowRightOutlined /></Link>)}</div></section>
      <Link to="/?view=overview" className="studio-resource-preview" aria-label="查看服务器资源"><div><strong>4090lziy</strong><Badge status={stale ? 'error' : resource ? 'success' : 'default'} text={stale ? '离线' : resource ? '在线' : '连接中'} /></div><span>单机联邦 · GPU 工作站</span><div className="studio-resource-meters">{!stale && resource ? <>{resource.gpus.map(gpu => <span key={gpu.index}>GPU {gpu.index}<b>{gpu.utilization.toFixed(0)}%</b></span>)}<span>内存<b>{resource.memoryPercent.toFixed(0)}%</b></span></> : <span>资源数据待刷新</span>}</div></Link>
    </div></div>
    <section className="studio-surface studio-recent"><div className="studio-section-head"><h2>实验记录</h2><Link to="/?view=jobs">全部实验 <ArrowRightOutlined /></Link></div>{recent.length ? <div className="studio-recent-list">{recent.map(job => <Link to={jobHref(job.id)} key={job.id}><span className="studio-recent-icon"><ExperimentOutlined /></span><span className="studio-recent-name"><strong>{job.request.name || job.id.slice(0, 8)}</strong><small>{job.request.group} · {methodLabel(job.request.method)}</small></span><time>{new Date(job.createdAt).toLocaleDateString('zh-CN')}</time><State value={job.status} /><ArrowRightOutlined /></Link>)}</div> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={catalog ? '暂无实验' : '正在读取实验'} />}</section>
    <FutureModuleEntries />
  </div>;
}
