import { useState, type CSSProperties, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import {
  AimOutlined, ArrowRightOutlined, DatabaseOutlined, ExperimentOutlined,
  FileSearchOutlined, HistoryOutlined, LineChartOutlined, LockOutlined,
  MinusOutlined, PlusOutlined, ScanOutlined, SecurityScanOutlined,
} from '@ant-design/icons';
import { methodLabel, terminal, type Catalog, type Job, type Library } from '../platform/api';
import { jobHref, viewHref, type PlatformView } from '../platform/navigation';
import { State } from '../platform/ui';
import { landscape } from './assets';

const modules = [
  { id: 'train', title: '训练实验', icon: <ExperimentOutlined />, x: 23, y: 25 },
  { id: 'experience', title: '模型验证', icon: <ScanOutlined />, x: 77, y: 25 },
  { id: 'privacy', title: '隐私保护', icon: <LockOutlined />, x: 23, y: 70 },
  { id: 'backdoor', title: '后门防御', icon: <SecurityScanOutlined />, x: 77, y: 70 },
  { id: 'compare', title: '算法对比', icon: <LineChartOutlined />, x: 50, y: 83 },
] satisfies { id: PlatformView; title: string; icon: ReactNode; x: number; y: number }[];

type Props = { jobs?: Job[]; loading?: boolean; library?: Library; catalog?: Catalog; preview?: boolean };

function Metric({ label, value, unit }: { label: string; value?: number; unit: string }) {
  return <div className="command-metric"><span>{label}</span><div><strong>{value == null ? '—' : value.toLocaleString('zh-CN')}</strong><span>{unit}</span></div></div>;
}

export function CommandHome({ jobs = [], loading = false, library, catalog, preview = false }: Props) {
  const [zoom, setZoom] = useState(1), [showLinks, setShowLinks] = useState(true);
  const [highlighted, setHighlighted] = useState<string>();
  const active = jobs.find(job => !terminal(job.status));
  const recent = jobs.filter(job => job.action === 'train').slice().sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, 3);
  const ready = !loading && !preview;
  const datasetCount = catalog ? new Set(catalog.groups.map(group => group.dataset)).size : undefined;
  const trainingCount = ready ? jobs.filter(job => job.action === 'train' && job.status === 'completed').length : undefined;
  const latest = active?.metrics.at(-1);
  const progress = active?.action === 'train' && active.request.rounds > 0 && latest
    ? Math.max(0, Math.min(100, latest.round / active.request.rounds * 100)) : undefined;

  return <section className="command-home design-enter" aria-label="功能导航">
    <div className="command-terrain" aria-hidden="true"><img src={landscape} alt="" fetchPriority="high" /><div className="command-terrain-grid" /></div>
    <div className="command-home-heading">
      <div><span className="command-eyebrow">协同研究工作台</span><h1>跨域协同训练</h1></div>
      <div className="command-home-heading-actions"><span className="command-map-label">功能拓扑</span><Link className="command-primary-action" to={viewHref('train')}><PlusOutlined />新建训练<ArrowRightOutlined /></Link></div>
    </div>
    <div className="command-home-grid">
      <div className="command-home-side">
        <section className="command-glass" aria-label="实验概览">
          <header className="command-panel-title"><ExperimentOutlined /><h2>实验工作台</h2><Link to={viewHref('jobs')} aria-label="查看全部实验"><ArrowRightOutlined /></Link></header>
          <div className="command-stat-grid"><Metric label="完成训练" value={trainingCount} unit="项" /><Metric label="数据集" value={ready ? datasetCount : undefined} unit="组" /></div>
          <div className="command-current-task">
            <div className="command-section-label"><span>当前任务</span>{active && <State value={active.status} />}</div>
            {active ? <>
              <Link className="command-task-name" to={jobHref(active.id)} aria-label={`当前任务 ${active.request.name || active.id.slice(0, 8)}`}>{active.request.name || active.id.slice(0, 8)}<ArrowRightOutlined /></Link>
              <span className="command-task-stage">{active.stage}</span>
              {progress != null && <div className="command-progress" role="progressbar" aria-label="当前任务进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}><span style={{ width: `${progress}%` }} /></div>}
            </> : <div className="command-idle"><span className="command-idle-mark" aria-hidden="true"><ExperimentOutlined /></span>{preview ? '尚未连接实验' : loading ? '正在读取实验' : '暂无运行任务'}</div>}
          </div>
        </section>
        <section className="command-glass command-history" aria-label="最近训练">
          <header className="command-panel-title"><HistoryOutlined /><h2>最近训练</h2></header>
          {recent.length ? <div className="command-history-list">{recent.map(job => <Link to={jobHref(job.id)} key={job.id}>
            <span className="command-history-icon"><ExperimentOutlined /></span><div><strong>{job.request.name || job.id.slice(0, 8)}</strong><span>{methodLabel(job.request.method)}</span></div><State value={job.status} />
          </Link>)}</div> : <div className="command-empty">{loading ? '正在读取记录' : '暂无训练记录'}</div>}
          <Link className="command-panel-link" to={viewHref('jobs')}>实验记录<ArrowRightOutlined /></Link>
        </section>
      </div>
      <section className="command-map" aria-label="功能拓扑地图">
        <div className="command-map-guide" aria-hidden="true"><span /><i /><span /></div>
        <div className="command-map-viewport">
          <div className="command-map-plane" style={{ transform: `scale(${zoom})` }} data-zoom={zoom.toFixed(1)}>
            <svg className={`command-topology-links${showLinks ? '' : ' is-hidden'}`} viewBox="0 0 1000 700" preserveAspectRatio="none" aria-hidden="true">
              <ellipse className="command-orbit" cx="500" cy="350" rx="315" ry="250" />
              <ellipse className="command-orbit inner" cx="500" cy="350" rx="225" ry="166" />
              {modules.map(item => <g key={item.id} className={highlighted === item.id ? 'is-highlighted' : ''}>
                <path className="command-link-shadow" d={`M 500 350 Q ${item.x < 50 ? 470 : 530} ${item.y * 7} ${item.x * 10} ${item.y * 7}`} />
                <path className="command-link" d={`M 500 350 Q ${item.x < 50 ? 470 : 530} ${item.y * 7} ${item.x * 10} ${item.y * 7}`} />
              </g>)}
            </svg>
            <div className="command-hub"><div className="command-hub-orbit" aria-hidden="true" /><span className="command-hub-icon"><SecurityScanOutlined /></span><strong>可信共享</strong><span>研究协同中心</span></div>
            {modules.map(item => <Link key={item.id} className={`command-map-node node-${item.id}`} to={viewHref(item.id)} aria-label={item.title}
              style={{ '--node-x': `${item.x}%`, '--node-y': `${item.y}%` } as CSSProperties}
              onMouseEnter={() => setHighlighted(item.id)} onMouseLeave={() => setHighlighted(undefined)}
              onFocus={() => setHighlighted(item.id)} onBlur={() => setHighlighted(undefined)}>
              <span className="command-node-icon">{item.icon}</span><span>{item.title}</span><ArrowRightOutlined />
            </Link>)}
          </div>
        </div>
        <div className="command-map-footer"><span className="command-map-caption"><i />模块导航</span>
          <div className="command-map-controls" aria-label="地图视图控制">
            <button type="button" aria-label="显示拓扑连线" aria-pressed={showLinks} onClick={() => setShowLinks(value => !value)}><span className="command-layer-icon" aria-hidden="true" /></button>
            <span className="command-control-divider" />
            <button type="button" aria-label="缩小地图" disabled={zoom <= .8} onClick={() => setZoom(value => Math.max(.8, +(value - .1).toFixed(1)))}><MinusOutlined /></button>
            <button type="button" aria-label="重置地图" onClick={() => setZoom(1)}><AimOutlined /></button>
            <button type="button" aria-label="放大地图" disabled={zoom >= 1.2} onClick={() => setZoom(value => Math.min(1.2, +(value + .1).toFixed(1)))}><PlusOutlined /></button>
          </div>
        </div>
      </section>
      <div className="command-home-side">
        <section className="command-glass" aria-label="模型与测试集">
          <header className="command-panel-title"><DatabaseOutlined /><h2>验证中心</h2></header>
          <div className="command-stat-grid"><Metric label="已保存模型" value={ready ? library?.models.length : undefined} unit="个" /><Metric label="测试集" value={ready ? library?.testsets.length : undefined} unit="组" /></div>
          <div className="command-verification-actions">
            <Link to={viewHref('experience')} aria-label="进入单图验证"><ScanOutlined /><div><strong>单图验证</strong><span>模型预测</span></div><ArrowRightOutlined /></Link>
            <Link to={viewHref('evaluate')} aria-label="独立评测"><FileSearchOutlined /><div><strong>独立评测</strong><span>测试集评估</span></div><ArrowRightOutlined /></Link>
          </div>
        </section>
        <section className="command-glass command-research" aria-label="安全验证">
          <header className="command-panel-title"><SecurityScanOutlined /><h2>安全验证</h2></header>
          <Link to={viewHref('privacy')}><span className="command-research-icon"><LockOutlined /></span><span>隐私保护</span><ArrowRightOutlined /></Link>
          <Link to={viewHref('backdoor')}><span className="command-research-icon"><SecurityScanOutlined /></span><span>后门防御</span><ArrowRightOutlined /></Link>
        </section>
      </div>
    </div>
    <div className="command-workflow" aria-label="研究流程">
      {[
        { title: '训练配置', to: 'train', icon: <ExperimentOutlined /> },
        { title: '实验监控', to: 'jobs', icon: <LineChartOutlined /> },
        { title: '模型验证', to: 'experience', icon: <ScanOutlined /> },
        { title: '独立评测', to: 'evaluate', icon: <FileSearchOutlined /> },
        { title: '结果对比', to: 'compare', icon: <LineChartOutlined /> },
      ].map((step, index) => <Link to={viewHref(step.to as PlatformView)} key={step.to} aria-label={`研究流程：${step.title}`}><span className="command-workflow-number">0{index + 1}</span>{step.icon}<strong>{step.title}</strong><ArrowRightOutlined /></Link>)}
    </div>
  </section>;
}
