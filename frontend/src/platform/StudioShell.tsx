import type { PropsWithChildren } from 'react';
import { Link } from 'react-router-dom';
import { Button } from 'antd';
import { ArrowRightOutlined, GlobalOutlined } from '@ant-design/icons';
import { mainPages, mainView, pages, viewHref, type PlatformView } from './navigation';
import { plannedModules } from './extensions';

export function StudioShell({ view, connected, running, openCurrent, children }: PropsWithChildren<{
  view: PlatformView; connected: boolean; running: boolean; openCurrent: () => void;
}>) {
  const active = mainView(view);
  return <div className="studio-shell">
    <a className="studio-skip" href="#studio-main">跳到主要内容</a>
    <aside className="studio-sidebar">
      <Link to="/" className="studio-brand" aria-label="跨域联邦学习 · 返回首页">
        <span className="studio-brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>
        <span>跨域联邦学习<small>科研仿真平台</small></span>
      </Link>
      <div className="studio-nav-label">实验工作区</div>
      <nav className="studio-nav" aria-label="主要功能">{Object.entries(mainPages).map(([id, page]) =>
        <Link key={id} to={viewHref(id as PlatformView)} aria-label={page.label} aria-current={active === id ? 'page' : undefined} className={active === id ? 'active' : ''}>{page.icon}<span>{page.label}</span></Link>)}</nav>
      <div className="studio-nav-label">仿真与扩展</div>
      <nav className="studio-nav studio-secondary-nav" aria-label="仿真与研究扩展">
        <a href="/demo"><GlobalOutlined /><span>地图仿真</span><small>模拟</small></a>
        {Object.entries(plannedModules).map(([id, module]) =>
          <Link key={id} to={viewHref(id as PlatformView)} aria-current={view === id ? 'page' : undefined} className={view === id ? 'active' : ''}>{module.icon}<span>{module.label}</span><small>未接入</small></Link>)}
      </nav>
      <div className="studio-sidebar-footer"><span>FederatedScope</span><small>单机多客户端 · 联邦实验</small></div>
    </aside>
    <div className="studio-content">
      <header className="studio-topbar"><div className="studio-breadcrumb"><span>科研仿真</span><i>/</i><strong>{pages[view].label}</strong></div>
        <div className="studio-topbar-right">
          {running && <Button className="studio-current" type="text" onClick={openCurrent}><i className={connected ? 'live-dot' : 'offline-dot'} />当前任务 <ArrowRightOutlined /></Button>}
          <span className="studio-connection" role="status"><i className={connected ? 'live-dot' : 'offline-dot'} />{connected ? '服务已连接' : '服务未连接'}</span>
        </div>
      </header>
      <main id="studio-main" className="studio-main">{children}</main>
    </div>
  </div>;
}
