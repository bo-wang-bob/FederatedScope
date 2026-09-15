import { useEffect, useState, type PropsWithChildren } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button, Checkbox, ConfigProvider, Modal, Select, Segmented, Tooltip } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import {
  ArrowLeftOutlined, ArrowRightOutlined, CheckOutlined, CloseOutlined, ExpandOutlined,
  ExperimentOutlined, FileImageOutlined, GlobalOutlined, InfoCircleOutlined,
  LineChartOutlined, LockOutlined, PictureOutlined, ScanOutlined, SecurityScanOutlined,
} from '@ant-design/icons';
import { mainPages, mainView, pages, resolveView, viewHref, type PlatformView } from '../platform/navigation';
import { researchTheme } from '../platform/theme';
import { TrainingForm } from '../platform/training';
import type { Draft } from '../platform/draft';
import type { RequestConfig } from '../platform/api';
import { designCatalog, designMethods } from './catalog';
import { imageSource, landscape, samples, type DesignSample } from './assets';
import '../platform/studio.css';
import './design.css';

const previewTheme = { ...researchTheme, token: { ...researchTheme.token, fontSizeSM: 14 } };
const domainOptions = [{ value: 'all', label: '全部域' }, { value: 'Art', label: 'Art' }, { value: 'Real_World', label: 'Real World' }];
const classOptions = [{ value: 'all', label: '全部类别' }, ...Array.from(new Set(samples.map(s => s.category))).map(category => ({ value: category, label: category }))];
type Notice = { title: string; detail: string };

function DesignShell({ view, children, showSources }: PropsWithChildren<{ view: PlatformView; showSources: () => void }>) {
  return <div className="studio-shell design-console">
    <a className="studio-skip" href="#design-main">跳到主要内容</a>
    <aside className="design-sidebar">
      <Link to="/" className="design-brand" aria-label="跨域联邦学习 · 返回首页">
        <span className="studio-brand-mark" aria-hidden="true"><i /><i /><i /><i /></span><span>跨域联邦学习</span>
      </Link>
      <nav className="design-nav" aria-label="主要功能">{Object.entries(mainPages).map(([id, page]) =>
        <Link key={id} to={viewHref(id as PlatformView)} aria-label={page.label} aria-current={mainView(view) === id ? 'page' : undefined}>{page.icon}<span>{page.label}</span></Link>)}</nav>
      <nav className="design-nav design-secondary" aria-label="仿真与研究扩展">
        <a href="/demo"><GlobalOutlined /><span>地图仿真</span></a>
        <Link to={viewHref('privacy')} aria-current={view === 'privacy' ? 'page' : undefined}><LockOutlined /><span>隐私研究</span></Link>
        <Link to={viewHref('backdoor')} aria-current={view === 'backdoor' ? 'page' : undefined}><SecurityScanOutlined /><span>后门研究</span></Link>
      </nav>
      <div className="design-sidebar-end" aria-hidden="true"><span /><i /><span /></div>
    </aside>
    <div className="design-workspace">
      <header className="design-topbar"><span>{pages[view].label}</span><div className="design-topbar-actions">
        <span className="design-mode" role="status">设计预览</span>
        <Tooltip title="图片来源"><Button type="text" icon={<InfoCircleOutlined />} aria-label="图片来源" onClick={showSources} /></Tooltip>
      </div></header>
      <main className="design-main" id="design-main">{children}</main>
    </div>
  </div>;
}

function Home() {
  return <section className="design-home design-enter" aria-label="功能导航">
    <div className="design-hero">
      <img src={landscape} alt="纳米布沙漠卫星影像，作为科研仿真场景配图" fetchPriority="high" />
      <div className="design-hero-content"><span className="design-hero-symbol" aria-hidden="true"><ExperimentOutlined /></span>
        <h1>跨域联邦学习</h1>
        <Link className="design-action" to={viewHref('train')}>新建训练 <ArrowRightOutlined /></Link>
      </div>
    </div>
    <div className="design-home-modules">
      <Link to={viewHref('experience')} className="design-module-card design-model-card" aria-label="模型验证">
        <div className="design-card-images"><img src={samples[0].src} alt="Office-Home Art 域无线电设备样本" /><img src={samples[1].src} alt="Office-Home Real World 域无线电设备样本" /></div>
        <div className="design-card-caption"><span><ScanOutlined /><h2>模型验证</h2></span><ArrowRightOutlined /></div>
      </Link>
      <Link to={viewHref('compare')} className="design-module-card design-compare-card" aria-label="算法对比">
        <div className="design-card-images"><img src={samples[2].src} alt="Office-Home Real World 域笔记本电脑样本" /></div>
        <div className="design-card-caption"><span><LineChartOutlined /><h2>算法对比</h2></span><ArrowRightOutlined /></div>
      </Link>
    </div>
  </section>;
}

function PageHeading({ title, children }: PropsWithChildren<{ title: string }>) {
  return <div className="design-heading"><h1>{title}</h1>{children}</div>;
}

function Training({ draft, onDraft, notify }: { draft?: Draft; onDraft: (draft: Draft) => void; notify: (notice: Notice) => void }) {
  const launch = { intent: undefined, busy: false, start: (_request: RequestConfig) => notify({
    title: '训练配置已确认', detail: '当前为设计预览，未预检、未提交训练。',
  }), resume() {}, cancel() {}, edit() {} };
  return <section className="design-training design-enter">
    <PageHeading title="训练实验" />
    <TrainingForm catalog={designCatalog} preview previewDraft={draft} onPreviewDraft={onDraft}
      disconnected={false} launch={launch} open={() => {}} />
  </section>;
}

function SampleImage({ sample, className = '', onFailure, onReady }: { sample: DesignSample; className?: string; onFailure?: () => void; onReady?: () => void }) {
  const [failed, setFailed] = useState(false);
  if (failed) return <div className={'design-image-failed ' + className} role="img" aria-label="图像无法载入"><PictureOutlined /><span>图像无法载入</span></div>;
  return <img className={className} src={sample.src} alt={`${sample.domain} · ${sample.category} · ${sample.filename}`}
    onError={() => { setFailed(true); onFailure?.(); }} onLoad={onReady} />;
}

function SampleWorkbench({ notify }: { notify: (notice: Notice) => void }) {
  const [method, setMethod] = useState('heterogeneous_solution');
  const [domain, setDomain] = useState('all'), [category, setCategory] = useState('all');
  const [selected, setSelected] = useState<string>(samples[1].id);
  const [expanded, setExpanded] = useState(false), [loadedSampleId, setLoadedSampleId] = useState<string>();
  const visible = samples.filter(s => (domain === 'all' || s.domain === domain) && (category === 'all' || s.category === category));
  const sample = visible.find(s => s.id === selected) || visible[0];
  const ready = !!sample && loadedSampleId === sample.id;
  const index = visible.findIndex(s => s.id === sample?.id);
  const selectNext = (direction: number) => {
    const next = visible[index + direction];
    if (next) { setLoadedSampleId(undefined); setSelected(next.id); }
  };
  return <div className="design-verification">
    <section className="design-gallery" aria-label="测试样本">
      <div className="design-gallery-header"><h2>测试样本</h2><span>Office-Home</span></div>
      <div className="design-gallery-filters"><Select aria-label="样本域" value={domain} options={domainOptions} onChange={setDomain} /><Select aria-label="样本类别" value={category} options={classOptions} onChange={setCategory} /></div>
      <div className="design-sample-grid">{visible.map(item => <button key={item.id} className={sample?.id === item.id ? 'selected' : ''}
        aria-label={`选择 ${item.domain} ${item.category}`} aria-pressed={sample?.id === item.id} onClick={() => { if (sample?.id !== item.id) { setLoadedSampleId(undefined); setSelected(item.id); } }}>
        <SampleImage sample={item} /><span>{item.category}</span>{sample?.id === item.id && <i aria-hidden="true"><CheckOutlined /></i>}
      </button>)}</div>
    </section>
    <section className="design-selected" aria-label="当前样本">
      <div className="design-image-toolbar"><span>{sample?.category || '未选择样本'}</span><Tooltip title="查看原图"><Button type="text" icon={<ExpandOutlined />} aria-label="查看原图" disabled={!ready} onClick={() => setExpanded(true)} /></Tooltip></div>
      <div className="design-image-stage">{sample ? <SampleImage key={sample.id} sample={sample} onReady={() => setLoadedSampleId(sample.id)} onFailure={() => setLoadedSampleId(undefined)} /> : <span>没有匹配的样本</span>}</div>
      <div className="design-image-meta"><span>{sample?.domain.replace('_', ' ')}</span><div><Button type="text" aria-label="上一张" icon={<ArrowLeftOutlined />} disabled={index <= 0} onClick={() => selectNext(-1)} /><Button type="text" aria-label="下一张" icon={<ArrowRightOutlined />} disabled={index < 0 || index >= visible.length - 1} onClick={() => selectNext(1)} /></div></div>
    </section>
    <section className="design-prediction" aria-label="预测配置与结果">
      <h2>模型</h2><Select aria-label="模型方案" value={method} onChange={setMethod} options={designMethods} />
      <div className="design-result-empty"><span className="design-empty-symbol"><ScanOutlined /></span><h3>尚未预测</h3></div>
      <Button type="primary" block icon={<ScanOutlined />} disabled={!sample || !ready} onClick={() => notify({ title: '预测预览', detail: '已选择图像与模型方案，尚未加载模型或执行推理。' })}>预测</Button>
    </section>
    <Modal open={expanded} onCancel={() => setExpanded(false)} footer={null} title={sample?.category} width={900} className="design-modal" closeIcon={<CloseOutlined />}>
      {expanded && sample && <SampleImage key={sample.id} sample={sample} className="design-full-image" />}
    </Modal>
  </div>;
}

function Evaluation({ notify }: { notify: (notice: Notice) => void }) {
  const [method, setMethod] = useState('heterogeneous_solution');
  const [domains, setDomains] = useState(['Art', 'Real_World']);
  return <div className="design-evaluation">
    <section className="design-evaluation-config"><h2>评测配置</h2>
      <label htmlFor="design-eval-model">模型方案</label><Select id="design-eval-model" value={method} options={designMethods} onChange={setMethod} />
      <label htmlFor="design-eval-testset">测试集</label><Select id="design-eval-testset" value="officehome-samples" options={[{ value: 'officehome-samples', label: 'Office-Home · 预览样本' }]} />
      <span className="design-field-label">测试域</span><Checkbox.Group value={domains} onChange={values => setDomains(values as string[])} options={domainOptions.slice(1)} />
      <Button type="primary" block disabled={!domains.length} onClick={() => notify({ title: '评测预览', detail: '当前仅预览评测配置，未提交任务或生成指标。' })}>开始评测 <ArrowRightOutlined /></Button>
    </section>
    <section className="design-evaluation-results" aria-label="评测结果"><h2>评测结果</h2>
      <Segmented aria-label="评测维度" options={['总体', '分域', '分类']} />
      <div className="design-results-placeholder"><FileImageOutlined /><h3>尚无评测结果</h3></div>
    </section>
  </div>;
}

function Verification({ evaluate, notify }: { evaluate: boolean; notify: (notice: Notice) => void }) {
  return <section className="design-enter"><PageHeading title="模型验证"><nav className="design-tabs" aria-label="验证方式">
    <Link to={viewHref('experience')} aria-current={!evaluate ? 'page' : undefined}>单图预测</Link>
    <Link to={viewHref('evaluate')} aria-current={evaluate ? 'page' : undefined}>独立评测</Link>
  </nav></PageHeading>{evaluate ? <Evaluation notify={notify} /> : <SampleWorkbench notify={notify} />}</section>;
}

function Comparison() {
  const [methods, setMethods] = useState(['heterogeneous_solution', 'fedavg']);
  const [dimension, setDimension] = useState('总体');
  return <section className="design-enter"><PageHeading title="算法对比" />
    <div className="design-comparison">
      <div className="design-comparison-controls"><label htmlFor="design-methods">对比方案</label><Select id="design-methods" mode="multiple" value={methods} options={designMethods} onChange={setMethods} placeholder="选择算法" /></div>
      <div className="design-comparison-heading"><Segmented aria-label="对比维度" value={dimension} onChange={setDimension} options={['总体', '分域', '分类']} /><Tooltip title="有真实评测结果后可导出"><span><Button disabled>导出结果</Button></span></Tooltip></div>
      <div className="design-comparison-empty"><LineChartOutlined /><h2>{methods.length < 2 ? '选择至少两种方案' : '尚无可对比结果'}</h2></div>
    </div>
  </section>;
}

function Reserved({ view }: { view: 'privacy' | 'backdoor' }) {
  return <section className="design-enter"><PageHeading title={pages[view].label} /><div className="design-reserved">
    {view === 'privacy' ? <LockOutlined /> : <SecurityScanOutlined />}<h2>未接入</h2>
  </div></section>;
}

export default function DesignPreview() {
  const location = useLocation();
  const requested = resolveView(new URLSearchParams(location.search).get('view'), location.pathname);
  const view = requested === 'jobs' ? 'train' : requested;
  const [sourcesOpen, setSourcesOpen] = useState(false), [notice, setNotice] = useState<Notice>();
  const [draft, setDraft] = useState<Draft>();
  useEffect(() => { document.title = `${pages[view].label} · 设计预览`; window.scrollTo({ top: 0, behavior: 'instant' }); }, [view]);
  return <ConfigProvider theme={previewTheme} locale={zhCN}>
    <DesignShell view={view} showSources={() => setSourcesOpen(true)}>
      {view === 'home' && <Home />}
      {view === 'train' && <Training draft={draft} onDraft={setDraft} notify={setNotice} />}
      {(view === 'experience' || view === 'evaluate') && <Verification evaluate={view === 'evaluate'} notify={setNotice} />}
      {view === 'compare' && <Comparison />}
      {(view === 'privacy' || view === 'backdoor') && <Reserved view={view} />}
    </DesignShell>
    <Modal open={!!notice} title={notice?.title} onCancel={() => setNotice(undefined)} className="design-modal"
      footer={<Button type="primary" onClick={() => setNotice(undefined)}>确定</Button>}><p>{notice?.detail}</p></Modal>
    <Modal open={sourcesOpen} title="图片来源" onCancel={() => setSourcesOpen(false)} footer={null} className="design-modal">
      <div className="design-source-list"><section><h3>场景影像</h3><p>NASA Earth Observatory，Joshua Stevens；Landsat 数据来自 USGS。仅作场景配图，不代表训练数据或实验结果。</p>
        <a href={imageSource} target="_blank" rel="noreferrer">Where the Dunes End <ArrowRightOutlined /></a></section>
        <section><h3>测试样本</h3><p>现有 Office-Home 测试集中的 4 张原始图像，展示原始域与类别标签。历史特征与图像的逐样本关联仍有来源限制。</p></section>
        <section><h3>预览范围</h3><p>配置选项为界面设计样例，未读取后端库存、未加载模型，不产生预测或评测结果。</p></section>
      </div>
    </Modal>
  </ConfigProvider>;
}
