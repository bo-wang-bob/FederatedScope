import { ArrowRightOutlined, AuditOutlined, ExperimentOutlined, GlobalOutlined, LineChartOutlined, PlusOutlined, ScanOutlined } from '@ant-design/icons';
import { Empty } from 'antd';
import { Link } from 'react-router-dom';
import { methodLabel, terminal, type Job } from './api';
import { jobHref } from './navigation';
import { plannedModules } from './extensions';
import { State } from './ui';

const entries = [
  { title: '训练实验', description: '数据配置与联邦训练', href: '/?view=train', icon: <ExperimentOutlined /> },
  { title: '模型验证', description: '测试样本与单图预测', href: '/?view=experience', icon: <ScanOutlined /> },
  { title: '独立评测', description: '总体、分域及分类指标', href: '/?view=evaluate', icon: <AuditOutlined /> },
  { title: '算法对比', description: '实验结果与方法对比', href: '/?view=compare', icon: <LineChartOutlined /> },
];
export function SystemHome({ jobs, loading }: { jobs: Job[]; loading: boolean }) {
  const active = jobs.find(job => !terminal(job.status));
  const recent = jobs.filter(job => job.action === 'train').slice(0, 4);
  return <div className="studio-home studio-page-enter">
    <section className="home-overview">
      <div><span className="studio-kicker">科研仿真实验平台</span><h1>跨域联邦学习</h1><div className="home-research-tags"><span>数据异构</span><span>联邦聚合</span><span>模型效用验证</span></div></div>
      <Link className="studio-button primary" to="/?view=train"><PlusOutlined />新建训练</Link>
    </section>
    {active && <Link className="home-active" to={jobHref(active.id)}><i className="live-dot" /><span>当前任务</span><strong>{active.request.name || active.id.slice(0, 8)}</strong><State value={active.status} /><ArrowRightOutlined /></Link>}
    <section className="home-entry-grid" aria-label="功能导航">{entries.map((entry,index) => <Link className="home-entry" aria-label={entry.title} key={entry.title} to={entry.href}><div className="home-entry-top"><span>{entry.icon}</span><small>0{index+1}</small></div><h2>{entry.title}</h2><p>{entry.description}</p><ArrowRightOutlined className="home-entry-arrow" /></Link>)}</section>
    <div className="home-workbench"><section className="home-recent studio-surface">
      <div className="studio-section-head"><h2>最近实验</h2><Link to="/?view=jobs">全部记录 <ArrowRightOutlined /></Link></div>
      {recent.length ? <div className="studio-recent-list">{recent.map(job => <Link to={jobHref(job.id)} key={job.id}><span className="studio-recent-icon"><ExperimentOutlined /></span><span className="studio-recent-name"><strong>{job.request.name || job.id.slice(0, 8)}</strong><small>{job.request.group.split('_')[0]} <i>·</i> {methodLabel(job.request.method)}</small></span><time>{new Date(job.createdAt).toLocaleDateString('zh-CN')}</time><State value={job.status} /><ArrowRightOutlined /></Link>)}</div> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={loading ? '正在读取实验' : '暂无训练实验'} />}
    </section><section className="home-research studio-surface"><div className="studio-section-head"><h2>仿真与研究扩展</h2></div>
      <a href="/demo"><GlobalOutlined /><span><strong>地图仿真</strong><small>纯前端模拟 · 非真实训练结果</small></span><ArrowRightOutlined /></a>
      {Object.entries(plannedModules).map(([id,module]) => <Link to={'/?view='+id} key={id}>{module.icon}<span><strong>{module.label}</strong><small>功能预留</small></span><em>未接入</em></Link>)}
    </section></div>
  </div>;
}
