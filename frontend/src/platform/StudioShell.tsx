import type { PropsWithChildren } from 'react';
import { Link } from 'react-router-dom';
import { Button, Dropdown } from 'antd';
import { ArrowRightOutlined, DownOutlined, GlobalOutlined } from '@ant-design/icons';
import { mainPages, mainView, viewHref, type PlatformView } from './navigation';
import { plannedModules } from './extensions';

export function StudioShell({ view, connected, running, openCurrent, children }: PropsWithChildren<{
  view: PlatformView; connected: boolean; running: boolean; openCurrent: () => void;
}>) {
  const active = mainView(view);
  return <div className="studio-shell">
    <a className="studio-skip" href="#studio-main">跳到主要内容</a>
    <header className="studio-topbar">
      <Link to="/" className="studio-brand" aria-label="全域智汇 · 返回首页"><span className="studio-brand-mark"><i /><i /><i /><i /></span><span>全域智汇<small>FEDERATED STUDIO</small></span></Link>
      <nav className="studio-nav" aria-label="主要功能">{Object.entries(mainPages).map(([id, page]) =>
        <Link key={id} to={viewHref(id as PlatformView)} aria-current={active === id ? 'page' : undefined} className={active === id ? 'active' : ''}>{page.label}</Link>)}</nav>
      <div className="studio-topbar-right">
        {running && <Button className="studio-current" type="text" onClick={openCurrent}><i className={connected ? 'live-dot' : 'offline-dot'} />当前任务 <ArrowRightOutlined /></Button>}
        <Dropdown trigger={['click']} menu={{ items: [
          ...Object.entries(plannedModules).map(([id, module]) => ({ key: id, icon: module.icon, label: <Link to={viewHref(id as PlatformView)}>{module.label}<span className="menu-planned">待开放</span></Link> })),
          { key: 'divider', type: 'divider' as const },
          { key: 'demo', icon: <GlobalOutlined />, label: <a href="/demo">地图模拟演示</a> },
        ] }}><Button type="text" className="studio-extension-trigger">研究扩展 <DownOutlined /></Button></Dropdown>
      </div>
    </header>
    <main id="studio-main" className="studio-main">{children}</main>
    <footer className="studio-footer"><span>全域智汇 <i>/</i> FederatedScope</span><span>RESEARCH, REFINED.</span></footer>
  </div>;
}
