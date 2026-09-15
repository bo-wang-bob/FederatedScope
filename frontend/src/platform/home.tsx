import { ArrowRightOutlined, ExperimentOutlined, PlusOutlined, ScanOutlined } from '@ant-design/icons';
import { Empty } from 'antd';
import { Link } from 'react-router-dom';
import { methodLabel, terminal, type Job } from './api';
import { jobHref } from './navigation';
import { State } from './ui';

export function SystemHome({ jobs, loading }: { jobs: Job[]; loading: boolean }) {
  const active = jobs.find(job => !terminal(job.status));
  const recent = jobs.filter(job => job.action === 'train').slice(0, 4);
  return <div className="studio-home studio-page-enter">
    <section className="home-hero">
      <div className="home-hero-copy"><span className="studio-kicker"><i /> FEDERATED LEARNING WORKSPACE</span>
        <h1>让每一次训练，<br />走向更好的<span>模型。</span></h1>
        <p>从联邦训练，到模型验证。</p>
        <div className="home-hero-actions"><Link className="studio-button primary" to="/?view=train"><PlusOutlined />新建训练<ArrowRightOutlined /></Link><Link className="studio-button subtle" to="/?view=experience">验证模型<ArrowRightOutlined /></Link></div>
      </div>
      <div className="home-art" aria-hidden="true">
        <div className="art-grid" /><div className="art-orbit orbit-one" /><div className="art-orbit orbit-two" /><div className="art-orbit orbit-three" />
        <div className="art-core"><div /><span>f</span></div>
        <div className="art-node node-one"><i /><i /><i /></div><div className="art-node node-two"><i /><i /><i /></div><div className="art-node node-three"><i /><i /><i /></div>
        <span className="art-label label-one">DISTRIBUTED IDEAS.</span><span className="art-label label-two">ONE SHARED INTELLIGENCE.</span>
      </div>
      <div className="hero-edition">01 — ACCURACY STUDIO</div>
    </section>
    {active && <Link className="home-active" to={jobHref(active.id)}><i className="live-dot" /><span>继续当前任务</span><strong>{active.request.name || active.id.slice(0, 8)}</strong><State value={active.status} /><ArrowRightOutlined /></Link>}
    <div className="home-workbench"><section className="home-recent">
      <div className="studio-section-head"><div><span className="studio-kicker">RECENT WORK</span><h2>最近实验</h2></div><Link to="/?view=jobs">全部记录 <ArrowRightOutlined /></Link></div>
      {recent.length ? <div className="studio-recent-list">{recent.map(job => <Link to={jobHref(job.id)} key={job.id}><span className="studio-recent-icon"><ExperimentOutlined /></span><span className="studio-recent-name"><strong>{job.request.name || job.id.slice(0, 8)}</strong><small>{job.request.group.split('_')[0]} <i>·</i> {methodLabel(job.request.method)}</small></span><time>{new Date(job.createdAt).toLocaleDateString('zh-CN')}</time><State value={job.status} /><ArrowRightOutlined /></Link>)}</div> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={loading ? '正在读取实验' : '创建你的第一个实验'} />}
    </section><Link className="home-model-card" to="/?view=experience"><div className="home-model-icon"><ScanOutlined /></div><span className="studio-kicker">MODEL PLAYGROUND</span><h2>一个样本，<br />看见模型的判断。</h2><span className="home-card-action">打开模型验证 <ArrowRightOutlined /></span></Link></div>
    <div className="home-bottom-links"><Link to="/?view=evaluate">独立评测 <ArrowRightOutlined /></Link><Link to="/?view=compare">多实验对比 <ArrowRightOutlined /></Link><a href="/demo">地图模拟演示 <ArrowRightOutlined /></a></div>
  </div>;
}
