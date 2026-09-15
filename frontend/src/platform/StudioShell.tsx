import { useState, type PropsWithChildren } from 'react';
import { Link } from 'react-router-dom';
import { Badge, Button, Drawer, Tooltip } from 'antd';
import { ArrowRightOutlined, DeploymentUnitOutlined, GlobalOutlined, MenuFoldOutlined, MenuOutlined, MenuUnfoldOutlined } from '@ant-design/icons';
import { mainPages, mainView, pages, viewHref, type PlatformView } from './navigation';
import { plannedModules } from './extensions';

export function StudioShell({ view, connected, running, openCurrent, children }: PropsWithChildren<{
  view: PlatformView; connected: boolean; running: boolean; openCurrent: () => void;
}>) {
  const [collapsed, setCollapsed] = useState(false), [mobileOpen, setMobileOpen] = useState(false);
  const active = mainView(view);
  const navigation = <>
    <Link to="/" className="studio-brand" aria-label="全域智汇 · 返回首页" onClick={() => setMobileOpen(false)}><span className="studio-brand-mark"><DeploymentUnitOutlined /></span><span className="studio-brand-text"><strong>全域智汇</strong><small>FEDERATED STUDIO</small></span></Link>
    <nav className="studio-nav" aria-label="主要功能"><span className="studio-nav-caption">工作空间</span>
      {Object.entries(mainPages).map(([key, page]) => <Tooltip title={collapsed ? page.label : undefined} placement="right" key={key}><Link to={viewHref(key as PlatformView)} aria-label={page.label} aria-current={active === key ? 'page' : undefined} className={active === key ? 'active' : ''} onClick={() => setMobileOpen(false)}>{page.icon}<span>{page.label}</span>{active === key && <i />}</Link></Tooltip>)}
      <span className="studio-nav-caption studio-nav-extension">研究扩展</span>
      {Object.entries(plannedModules).map(([key, page]) => <Tooltip title={collapsed ? `${page.label} · 规划中` : undefined} placement="right" key={key}><Link to={viewHref(key as PlatformView)} aria-label={`${page.label} · 规划中`} aria-current={active === key ? 'page' : undefined} className={active === key ? 'active' : ''} onClick={() => setMobileOpen(false)}>{page.icon}<span>{page.label}</span><small>规划中</small></Link></Tooltip>)}
    </nav>
    <div className="studio-sidebar-bottom"><a href="/demo" className="studio-demo" aria-label="地图模拟演示"><GlobalOutlined /><span>地图演示</span><ArrowRightOutlined /></a>
      <div className="studio-host"><span className="studio-host-icon">G</span><div><strong>4090lziy</strong><small><Badge status={connected ? 'success' : 'default'} />{connected ? '已连接 · 单机联邦' : '连接待恢复'}</small></div></div>
    </div>
  </>;
  return <div className={`studio-shell ${collapsed ? 'studio-collapsed' : ''}`}>
    <a className="studio-skip" href="#studio-main">跳到主要内容</a>
    <aside className="studio-sidebar">{navigation}<Button className="studio-collapse" type="text" aria-label={collapsed ? '展开侧栏' : '收起侧栏'} icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />} onClick={() => setCollapsed(value => !value)} /></aside>
    <Drawer className="studio-mobile-drawer" title="导航" open={mobileOpen} onClose={() => setMobileOpen(false)} placement="left" size={260}>{navigation}</Drawer>
    <div className="studio-content"><header className="studio-topbar"><div><Button className="studio-mobile-trigger" aria-label="打开功能导航" aria-expanded={mobileOpen} type="text" icon={<MenuOutlined />} onClick={() => setMobileOpen(true)} /><span className="studio-topbar-prefix">工作空间</span><span className="studio-topbar-divider">/</span><strong>{pages[view].label}</strong></div><div className="studio-topbar-right">{running && <Button type="text" className="studio-current" onClick={openCurrent}><Badge status={connected ? 'processing' : 'default'} />当前任务 <ArrowRightOutlined /></Button>}<span className="studio-system-badge">ACCURACY LAB</span></div></header>
      <main id="studio-main" className="studio-main">{children}</main>
    </div>
  </div>;
}
