import { ArrowRightOutlined, CheckCircleOutlined, ClockCircleOutlined, DatabaseOutlined,
  ExperimentOutlined, FileSearchOutlined, GlobalOutlined, ScanOutlined } from '@ant-design/icons';
import { Badge, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { methodLabel, statusText, terminal, type Catalog, type Job, type Library, type Resource } from './api';
import { pages, viewHref, type PlatformView } from './navigation';

const actionLabels: Record<Job['action'], string> = { train: '联邦训练', evaluate: '独立评测', inspect: '缓存预检', predict: '单图预测' };
const jobHref = (id: string) => `/?${new URLSearchParams({ view: 'jobs', id })}`;
const jobName = (job: Job) => job.request.name || `${actionLabels[job.action]} · ${job.id.slice(0, 8)}`;

function Shortcut({ view }: { view: PlatformView }) {
  const page = pages[view];
  return <Link className="home-shortcut" to={viewHref(view)}><span className="home-shortcut-icon">{page.icon}</span>
    <span><strong>{page.label}</strong><small>{page.description}</small></span><ArrowRightOutlined /></Link>;
}

export function SystemHome({ catalog, jobs, library, resource, stale }: {
  catalog?: Catalog; jobs: Job[]; library: Library; resource?: Resource; stale: boolean;
}) {
  const running = jobs.filter(job => !terminal(job.status));
  const recent = [...jobs].filter(job => job.action !== 'inspect' && terminal(job.status))
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, 3);
  const imageModels = library.models.filter(model => /^(officehome|digit3|domainnet)_/.test(model.group));
  const imagePairs = imageModels.filter(model => library.testsets.some(test => test.featureSpace === model.featureSpace));
  const stats = [
    { label: '已保存模型', value: library.models.length, note: '检查点数量 · 含 final / best', view: 'evaluate' },
    { label: '已登记测试集', value: library.testsets.length, note: '可选范围以模型兼容性为准', view: 'evaluate' },
    { label: '已发现缓存配置', value: catalog?.groups.filter(group => group.cacheFound).length, note: '完整性仍需本次预检', view: 'cache' },
    { label: '已完成训练', value: jobs.filter(job => job.action === 'train' && job.status === 'completed').length, note: '查看模型、配置与日志', view: 'jobs' },
  ] as const;

  return <div className={`system-home ${stale ? 'home-stale' : ''}`}>
    <div className="home-inventory" aria-label="服务器资产概览">{stats.map(stat => <Link to={viewHref(stat.view)} className="home-inventory-item" key={stat.label}>
      <span>{stat.label}<ArrowRightOutlined /></span><strong>{catalog ? stat.value?.toLocaleString() ?? '—' : '—'}</strong><small>{stat.note}</small>
    </Link>)}</div>

    <section className="home-launchpad" aria-label="主要功能导航">
      <article className="home-feature home-feature-primary">
        <div className="home-feature-heading"><span className="home-feature-icon"><ScanOutlined /></span><Tag color="cyan">模型可用性验证</Tag></div>
        <div><span className="home-kicker">MODEL EXPERIENCE</span><h2>让模型给出答案</h2><p>从测试集中选一张图片，查看预测类别、分类分数与真实标签。</p></div>
        <ol className="home-model-flow" aria-label="单图体验流程">{['选择已存模型', '浏览测试图片', '核对预测结果'].map((step, index) => <li key={step}><span>{index + 1}</span>{step}</li>)}</ol>
        <div className="home-feature-bottom"><Link className="home-primary-link" to={viewHref('experience')}>进入模型体验台 <ArrowRightOutlined /></Link>
          <span>{!catalog ? '正在读取模型目录' : imagePairs.length ? `${imagePairs.length} 个图像模型检查点匹配到测试集` : '尚无匹配测试集的图像模型'}</span></div>
        <small className="home-feature-note">真实测试图片 · 已存冻结特征推理 · 不重新提取原图特征</small>
      </article>
      <div className="home-secondary-features">
        <article className="home-feature home-training-entry"><span className="home-feature-icon"><ExperimentOutlined /></span><div><h2>新建准确率训练</h2><p>选择算法与增强方式，预检通过后再启动。</p><Link to={viewHref('train')}>配置训练 <ArrowRightOutlined /></Link></div></article>
        <article className="home-feature home-evaluation-entry"><span className="home-feature-icon"><FileSearchOutlined /></span><div><h2>独立验证模型表现</h2><p>选择模型、测试集与范围，查看分域和分类指标。</p><Link to={viewHref('evaluate')}>进入独立评测 <ArrowRightOutlined /></Link></div></article>
      </div>
    </section>

    <div className="home-workspace-grid">
      <section className="home-section" aria-labelledby="home-navigation-title">
        <div className="home-section-heading"><div><h2 id="home-navigation-title">工作区导航</h2><p>从准备数据到结果复核，每一步都有入口。</p></div><span className="home-kicker">WORKSPACE</span></div>
        <div className="home-shortcut-grid">{(['overview', 'jobs', 'cache', 'compare'] as const).map(view => <Shortcut key={view} view={view} />)}</div>
        <a className="home-demo-link" href="/demo"><span className="home-shortcut-icon"><GlobalOutlined /></span><span><strong>地图模拟演示 <Tag>纯前端</Tag></strong><small>保留原地图与联邦流程动画，不会启动真实任务。</small></span><ArrowRightOutlined /></a>
        <div className="home-workflow-note"><DatabaseOutlined /><span>原始特征仍使用已有缓存；选择“本架构”后，可按配置生成增强数据或复用增强缓存。</span></div>
      </section>

      <section className="home-section home-activity" aria-labelledby="home-activity-title">
        <div className="home-section-heading"><div><h2 id="home-activity-title">继续工作</h2><p>{stale ? '连接异常，下方为上次读取的记录。' : '当前任务与最近完成的操作。'}</p></div><Link to={viewHref('jobs')}>全部记录 <ArrowRightOutlined /></Link></div>
        {!catalog ? <div className="home-empty"><ClockCircleOutlined /><div><strong>{stale ? '暂时无法读取任务' : '正在读取任务'}</strong><p>导航入口仍可使用，连接恢复后将自动更新。</p></div></div>
          : running.length ? <div className="home-active-jobs">{running.map(job => <Link className="home-active-job" to={jobHref(job.id)} key={job.id}>
            <div><Badge status={stale ? 'default' : 'processing'} text={stale ? '状态待刷新' : statusText[job.status] || job.status} /><span>{actionLabels[job.action]}</span></div>
            <strong>{jobName(job)}</strong><p>{job.stage}</p><span className="home-resume">查看任务与进度 <ArrowRightOutlined /></span>
          </Link>)}</div>
            : <div className="home-empty"><CheckCircleOutlined /><div><strong>{stale ? '上次读取时无进行中任务' : '当前没有进行中的任务'}</strong><p>可以体验已有模型，或配置下一次训练。</p></div></div>}
        {recent.length > 0 && <div className="home-recent-jobs"><h3>最近记录</h3>{recent.map(job => <Link className="home-recent-job" to={jobHref(job.id)} key={job.id}>
          <span><strong>{jobName(job)}</strong><small>{actionLabels[job.action]}{job.request.method ? ` · ${methodLabel(job.request.method)}` : ''} · {new Date(job.createdAt).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}</small></span>
          <Tag color={job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : 'default'}>{statusText[job.status] || job.status}</Tag>
        </Link>)}</div>}
      </section>
    </div>

    <Link className="home-server-strip" to={viewHref('overview')} aria-label="查看服务器资源与运行总览">
      <span><Badge status={stale ? 'error' : resource ? 'success' : 'default'} /><strong>{catalog?.host || '4090lziy'}</strong><small>单机模拟多客户端</small></span>
      <span className="home-server-metrics">{stale ? '连接异常 · 资源数据待刷新' : resource ? <>{resource.gpus.map(gpu => <span key={gpu.index}>GPU {gpu.index}<b>{gpu.utilization.toFixed(0)}%</b></span>)}<span>内存<b>{resource.memoryPercent.toFixed(0)}%</b></span>{resource.gpuError && <span>GPU 指标不可用</span>}</> : '正在连接服务器'}</span>
      <ArrowRightOutlined />
    </Link>
  </div>;
}
