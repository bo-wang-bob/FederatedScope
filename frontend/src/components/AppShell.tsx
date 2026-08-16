import {
  ApartmentOutlined,
  ControlOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  MenuOutlined,
} from '@ant-design/icons';
import { Badge, Button, ConfigProvider, Drawer, Layout, Menu, Tag, theme } from 'antd';
import { useState, type PropsWithChildren } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAppStore } from '../store/useAppStore';

const { Sider, Header, Content } = Layout;

function Brand() {
  return <div className="brand"><div className="brand-mark"><span /><span /><span /></div><div><div className="brand-title">全域智汇</div><div className="brand-subtitle">FEDERATED INTELLIGENCE</div></div></div>;
}

export function AppShell({ children }: PropsWithChildren) {
  const navigate = useNavigate();
  const location = useLocation();
  const activeExperimentId = useAppStore((state) => state.activeExperimentId);
  const [navigationOpen, setNavigationOpen] = useState(false);
  const immersive = location.pathname === '/overview';
  const monitorPath = activeExperimentId ? `/experiments/${activeExperimentId}/live` : '/experiments/current/live';
  const menuItems = [
    { key: '/overview', icon: <DashboardOutlined />, label: '综合态势' },
    { key: '/scenario-analysis', icon: <ApartmentOutlined />, label: '场景与异构分析' },
    { key: '/experiments/new', icon: <ExperimentOutlined />, label: '实验配置' },
    { key: monitorPath, icon: <ControlOutlined />, label: '运行监控', disabled: !activeExperimentId },
    { key: '/reports', icon: <FileTextOutlined />, label: '实验记录' },
  ];
  const selectedKey = location.pathname === '/overview'
    ? '/overview'
    : location.pathname === '/scenario-analysis'
      ? '/scenario-analysis'
      : location.pathname === '/experiments/new'
        ? '/experiments/new'
        : location.pathname.endsWith('/live')
          ? monitorPath
          : location.pathname === '/reports'
            ? '/reports'
            : '/overview';

  const navigation = <>
    <Brand />
    <div className="system-label">全域联邦学习实验平台</div>
    <Menu mode="inline" selectedKeys={[selectedKey]} items={menuItems} onClick={({ key }) => { navigate(key); setNavigationOpen(false); }} className="side-menu" />
    <div className="sidebar-status">
      <div className="sidebar-status-head"><span>单机实验引擎</span><Badge status="processing" /></div>
      <div className="engine-row"><span>控制方式</span><Tag color="cyan">API</Tag></div>
      <div className="engine-row"><span>逻辑客户端</span><strong>60</strong></div>
      <div className="engine-row"><span>数据域</span><strong>4</strong></div>
    </div>
  </>;

  return <ConfigProvider theme={{ algorithm: theme.darkAlgorithm, token: { colorPrimary: '#39c9ff', colorBgBase: '#07101f', colorBgContainer: '#0e1c2d', colorBorder: 'rgba(125, 177, 220, .16)', borderRadius: 8, fontFamily: "Inter, 'PingFang SC', 'Microsoft YaHei', sans-serif" } }}>
    <Layout className={`app-layout ${immersive ? 'immersive-layout' : ''}`}>
      {!immersive && <Sider width={228} className="app-sider">{navigation}</Sider>}
      {immersive && <><Button className="overview-menu-trigger" type="text" icon={<MenuOutlined />} aria-label="打开导航" onClick={() => setNavigationOpen(true)} /><Drawer className="navigation-drawer" placement="left" width={260} open={navigationOpen} onClose={() => setNavigationOpen(false)} closable={false}>{navigation}</Drawer></>}
      <Layout>
        {!immersive && <Header className="app-header"><div className="header-context"><span className="context-dot" /><span className="context-name">跨域联邦学习单机实验平台</span><Tag bordered={false} color="cyan">真实任务控制</Tag></div>{activeExperimentId && <Button type="link" onClick={() => navigate(monitorPath)}>打开当前实验</Button>}</Header>}
        <Content className={`app-content ${immersive ? 'immersive-content' : ''}`}>{children}</Content>
      </Layout>
    </Layout>
  </ConfigProvider>;
}
