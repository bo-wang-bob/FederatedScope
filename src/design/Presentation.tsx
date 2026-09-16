import { useState, type PropsWithChildren } from 'react';
import { Link } from 'react-router-dom';
import { Button, Modal, Tooltip } from 'antd';
import { ArrowRightOutlined, ExperimentOutlined, InfoCircleOutlined, LineChartOutlined, LockOutlined, ScanOutlined, SecurityScanOutlined } from '@ant-design/icons';
import { mainPages, mainView, pages, viewHref, type PlatformView } from '../platform/navigation';
import { imageSource, landscape, samples } from './assets';
import '../platform/studio.css';
import './design.css';

// Shared approved presentation; live data/commands stay in the platform controllers.
export function ConsoleShell({ view, children, mode = 'preview', connected = false, running = false, openCurrent, showDatasetImages = true }: PropsWithChildren<{
  view: PlatformView; mode?: 'preview' | 'live'; connected?: boolean; running?: boolean; openCurrent?: () => void;
  showDatasetImages?: boolean;
}>) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  return <div className={'studio-shell design-console' + (mode === 'live' ? ' live-console' : '')}>
    <a className="studio-skip" href="#design-main">跳到主要内容</a>
    <aside className="design-sidebar">
      <Link to="/" className="design-brand" aria-label="跨域协同训练 · 返回首页"><span className="studio-brand-mark" aria-hidden="true"><i /><i /><i /><i /></span><span>跨域协同训练</span></Link>
      <nav className="design-nav" aria-label="主要功能">{Object.entries(mainPages).map(([id, page]) =>
        <Link key={id} to={viewHref(id as PlatformView)} aria-label={page.label} aria-current={mainView(view) === id ? 'page' : undefined}>{page.icon}<span>{page.label}</span></Link>)}</nav>
      <nav className="design-nav design-secondary" aria-label="研究扩展">
        <Link to={viewHref('privacy')} aria-current={view === 'privacy' ? 'page' : undefined}><LockOutlined /><span>隐私研究</span></Link>
        <Link to={viewHref('backdoor')} aria-current={view === 'backdoor' ? 'page' : undefined}><SecurityScanOutlined /><span>后门研究</span></Link>
      </nav>
      <div className="design-sidebar-end" aria-hidden="true"><span /><i /><span /></div>
    </aside>
    <div className="design-workspace">
      <header className="design-topbar"><span>{pages[view].label}</span><div className="design-topbar-actions">
        {mode === 'live' && running && <Button type="text" onClick={openCurrent}>当前任务 <ArrowRightOutlined /></Button>}
        <span className={'design-mode' + (mode === 'live' ? connected ? ' connected' : ' disconnected' : '')} role="status">{mode === 'preview' ? '设计预览' : connected ? '服务已连接' : '服务未连接'}</span>
        <Tooltip title="图片来源"><Button type="text" icon={<InfoCircleOutlined />} aria-label="图片来源" onClick={() => setSourcesOpen(true)} /></Tooltip>
      </div></header>
      <main className="design-main" id="design-main">{children}</main>
    </div>
    <Modal open={sourcesOpen} title="图片来源" onCancel={() => setSourcesOpen(false)} footer={null} className="design-modal">
      <div className="design-source-list"><section><h3>场景影像</h3><p>NASA Earth Observatory，Joshua Stevens；Landsat 数据来自 USGS。仅作场景配图，不代表训练数据或实验结果。</p>
        <a href={imageSource} target="_blank" rel="noreferrer">Where the Dunes End <ArrowRightOutlined /></a></section>
        <section><h3>测试样本</h3><p>{showDatasetImages ? '首页素材来自现有 Office-Home 测试集。' : '场景影像不是军机测试样本。'}模型验证页展示所选测试集的原始图像；历史特征与图像的逐样本关联仍有来源限制。</p></section>
        {mode === 'preview' && <section><h3>预览范围</h3><p>配置选项为界面设计样例，未读取后端库存、未加载模型，不产生预测或评测结果。</p></section>}
      </div>
    </Modal>
  </div>;
}

export function PhotoHome({ showDatasetImages = true }: { showDatasetImages?: boolean }) {
  return <section className="design-home design-enter" aria-label="功能导航">
    <div className="design-hero"><img src={landscape} alt="纳米布沙漠卫星影像，作为科研仿真场景配图" fetchPriority="high" />
      <div className="design-hero-content"><span className="design-hero-symbol" aria-hidden="true"><ExperimentOutlined /></span><h1>跨域协同训练</h1>
        <Link className="design-action" to={viewHref('train')}>新建训练 <ArrowRightOutlined /></Link>
      </div>
    </div>
    <div className="design-home-modules">
      <Link to={viewHref('experience')} className="design-module-card design-model-card" aria-label="模型验证">
        {showDatasetImages ? <div className="design-card-images"><img src={samples[0].src} alt="Office-Home Art 域无线电设备样本" /><img src={samples[1].src} alt="Office-Home Real World 域无线电设备样本" /></div> : <div className="demo-module-symbol" aria-hidden="true"><ScanOutlined /></div>}
        <div className="design-card-caption"><span><ScanOutlined /><h2>模型验证</h2></span><ArrowRightOutlined /></div>
      </Link>
      <Link to={viewHref('compare')} className="design-module-card design-compare-card" aria-label="算法对比">
        {showDatasetImages ? <div className="design-card-images"><img src={samples[2].src} alt="Office-Home Real World 域笔记本电脑样本" /></div> : <div className="demo-module-symbol" aria-hidden="true"><LineChartOutlined /></div>}
        <div className="design-card-caption"><span><LineChartOutlined /><h2>算法对比</h2></span><ArrowRightOutlined /></div>
      </Link>
    </div>
  </section>;
}
