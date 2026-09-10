import {
  ApartmentOutlined,
  ControlOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  MenuOutlined,
} from '@ant-design/icons';
import { Badge, Button, ConfigProvider, Drawer, Layout, Menu, Tag, theme, type MenuProps } from 'antd';
import { useState, type PropsWithChildren } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAppStore } from '../store/useAppStore';

const { Sider, Header, Content } = Layout;

function Brand({ onHome }: { onHome?: () => void }) {
  const content = <><div className="brand-mark" aria-hidden="true"><span /><span /><span /></div><div><div className="brand-title">全域智汇</div><div className="brand-subtitle">FEDERATED INTELLIGENCE</div></div></>;
  return onHome ? <button className="brand brand-home" onClick={onHome} aria-label="全域智汇 · 返回系统首页">{content}</button> : <div className="brand">{content}</div>;
}

export interface PlatformShellState {
  menuItems: MenuProps['items']; selectedKey: string; navigate: (key: string) => void;
  connected: boolean; clientCount?: number; domainCount?: number; openCurrent?: () => void;
  pageLabel?: string; sectionLabel?: string;
}

export function AppShell({ children, platform }: PropsWithChildren<{ platform?: PlatformShellState }>) {
  const navigate = useNavigate();
  const location = useLocation();
  const activeExperimentId = useAppStore((state) => state.activeExperimentId);
  const [navigationOpen, setNavigationOpen] = useState(false);
  const immersive = !platform && location.pathname === '/overview';
  const monitorPath = activeExperimentId ? `/experiments/${activeExperimentId}/live` : '/experiments/current/live';
  const menuItems = platform?.menuItems || [
    { key: '/overview', icon: <DashboardOutlined />, label: '综合态势' },
    { key: '/scenario-analysis', icon: <ApartmentOutlined />, label: '场景与异构分析' },
    { key: '/experiments/new', icon: <ExperimentOutlined />, label: '实验配置' },
    { key: monitorPath, icon: <ControlOutlined />, label: '运行监控', disabled: !activeExperimentId },
    { key: '/reports', icon: <FileTextOutlined />, label: '实验记录' },
  ];
  const selectedKey = platform?.selectedKey || (location.pathname === '/overview'
    ? '/overview'
    : location.pathname === '/scenario-analysis'
      ? '/scenario-analysis'
      : location.pathname === '/experiments/new'
        ? '/experiments/new'
        : location.pathname.endsWith('/live')
          ? monitorPath
          : location.pathname === '/reports'
            ? '/reports'
            : '/overview');

  const selectPage = (key: string) => { (platform?.navigate || navigate)(key); setNavigationOpen(false); };
  const navigation = <>
    <Brand onHome={platform ? () => selectPage('home') : undefined} />
    <nav className={platform ? 'platform-nav-body' : undefined} aria-label="系统功能导航">
    <div className="system-label">{platform ? '训练与模型验证平台' : '全域联邦学习实验平台'}</div>
    <Menu mode="inline" selectedKeys={[selectedKey]} items={menuItems} onClick={({ key }) => selectPage(key)} className="side-menu" />
    </nav>
    <div className="sidebar-status">
      <div className="sidebar-status-head"><span>{platform ? '4090lziy' : '统一实验引擎'}</span><Badge status={platform ? platform.connected ? 'success' : 'error' : 'processing'} /></div>
      {platform && <div className="platform-host-address">10.112.81.135 <span>{platform.connected ? '已连接' : '未连接'}</span></div>}
      <div className="engine-row"><span>执行方式</span><Tag color="cyan">{platform ? '单机联邦' : 'API'}</Tag></div>
      {platform ? <div className="engine-row platform-experiment-context"><span>当前 / 最近训练</span><span>{platform.clientCount ?? '—'} 客户端 · {platform.domainCount ?? '—'} 域</span></div> : <>
        <div className="engine-row"><span>逻辑客户端</span><strong>60</strong></div>
        <div className="engine-row"><span>数据域</span><strong>4</strong></div>
      </>}
    </div>
  </>;

  return <ConfigProvider theme={{ algorithm: theme.darkAlgorithm, token: { colorPrimary: '#39c9ff', colorBgBase: '#07101f', colorBgContainer: '#0e1c2d', colorBorder: 'rgba(125, 177, 220, .16)', borderRadius: 8, fontFamily: "Inter, 'PingFang SC', 'Microsoft YaHei', sans-serif" } }}>
    <Layout className={`app-layout ${immersive ? 'immersive-layout' : ''}`}>
      {!immersive && <Sider width={platform ? 248 : 228} className={`app-sider ${platform ? 'platform-sider' : ''}`}>{navigation}</Sider>}
      {platform && <Drawer className="navigation-drawer platform-navigation-drawer" title="功能导航" placement="left" width={280} open={navigationOpen} onClose={() => setNavigationOpen(false)}>{navigation}</Drawer>}
      {immersive && <><Button className="overview-menu-trigger" type="text" icon={<MenuOutlined />} aria-label="打开导航" onClick={() => setNavigationOpen(true)} /><Drawer className="navigation-drawer" placement="left" width={260} open={navigationOpen} onClose={() => setNavigationOpen(false)} closable={false}>{navigation}</Drawer></>}
      <Layout className={platform ? 'platform-content-layout' : undefined}>
        {!immersive && <Header className="app-header"><div className="header-context">
          {platform ? <><Button className="platform-mobile-menu" type="text" icon={<MenuOutlined />} aria-label="打开功能导航" aria-expanded={navigationOpen} onClick={() => setNavigationOpen(true)} />
            <nav className="platform-breadcrumb" aria-label="当前位置"><button onClick={() => selectPage('home')}>系统首页</button>{platform.selectedKey !== 'home' && <><span aria-hidden="true">/</span><span className="platform-breadcrumb-section">{platform.sectionLabel}</span><span className="platform-breadcrumb-section" aria-hidden="true">/</span><strong aria-current="page">{platform.pageLabel}</strong></>}</nav>
          </> : <><span className="context-dot" /><span className="context-name">跨域联邦学习统一实验平台</span><Tag bordered={false} color="cyan">真实任务控制</Tag></>}
        </div>{platform?.openCurrent ? <Button className="platform-current-task" type="link" onClick={platform.openCurrent}><Badge status={platform.connected ? 'processing' : 'default'} /> 查看当前任务</Button> : !platform && activeExperimentId && <Button type="link" onClick={() => navigate(monitorPath)}>打开当前实验</Button>}</Header>}
        <Content className={`app-content ${immersive ? 'immersive-content' : ''}`}>{children}</Content>
      </Layout>
    </Layout>
  </ConfigProvider>;
}
