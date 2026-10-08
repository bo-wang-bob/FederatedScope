import type { PropsWithChildren } from 'react';
import { Link } from 'react-router-dom';
import { Button } from 'antd';
import { ArrowRightOutlined, LockOutlined, SecurityScanOutlined } from '@ant-design/icons';
import { mainPages, mainView, pages, viewHref, type PlatformView } from '../platform/navigation';
import { PLATFORM_NAME } from '../platform/branding';
import { CommandHome } from './CommandHome';
import '../platform/studio.css';
import './design.css';
import './command.css';

const navigation = [
  ...Object.entries(mainPages).map(([id, page]) => ({ id: id as PlatformView, ...page })),
  { id: 'privacy' as const, label: '隐私保护', icon: <LockOutlined /> },
  { id: 'backdoor' as const, label: '后门防御', icon: <SecurityScanOutlined /> },
];

// Shared approved presentation; live data/commands stay in the platform controllers.
export function ConsoleShell({ view, children, mode = 'preview', connected = false, running = false, openCurrent }: PropsWithChildren<{
  view: PlatformView; mode?: 'preview' | 'live'; connected?: boolean; running?: boolean; openCurrent?: () => void;
  showDatasetImages?: boolean;
}>) {
  const renderLink = (item: typeof navigation[number]) => <Link key={item.id} to={viewHref(item.id)}
    aria-label={item.label} aria-current={mainView(view) === item.id ? 'page' : undefined}>
    {item.icon}<span>{item.label}</span>
  </Link>;
  return <div className={`studio-shell design-console command-console${mode === 'live' ? ' live-console' : ''}${view === 'home' ? ' command-home-view' : ''}`}>
    <a className="studio-skip" href="#design-main">跳到主要内容</a>
    <header className="command-header">
      <nav className="command-navigation" aria-label="主要功能">
        {navigation.slice(0, 3).map(renderLink)}
        <div className="command-brand" aria-label={PLATFORM_NAME}>
          <span className="command-brand-emblem" aria-hidden="true"><SecurityScanOutlined /></span><span>{PLATFORM_NAME}</span>
        </div>
        {navigation.slice(3).map(renderLink)}
      </nav>
      <div className="command-context">
        <div className="command-breadcrumb"><span className="command-context-mark" aria-hidden="true" /><span>科研验证</span><i>/</i><strong>{pages[view].label}</strong></div>
        <div className="design-topbar-actions">
          {mode === 'live' && running && <Button type="text" onClick={openCurrent}>当前任务 <ArrowRightOutlined /></Button>}
          <span className={'design-mode' + (mode === 'live' ? connected ? ' connected' : ' disconnected' : '')} role="status">{mode === 'preview' ? '设计预览' : connected ? '服务已连接' : '服务未连接'}</span>
        </div>
      </div>
    </header>
    <div className="design-workspace">
      <main className="design-main" id="design-main">{children}</main>
    </div>
  </div>;
}

export function PhotoHome(_props: { showDatasetImages?: boolean }) {
  return <CommandHome preview />;
}
