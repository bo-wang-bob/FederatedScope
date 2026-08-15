import {
  ApartmentOutlined,
  BarChartOutlined,
  ControlOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  MenuOutlined,
  SafetyCertificateOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { Badge, Button, ConfigProvider, Drawer, Layout, Menu, Space, Tag, theme } from 'antd';
import { useEffect, useState, type PropsWithChildren } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { resolveTrainingDataAdapter } from '../api/trainingAdapter';
import { phases, useAppStore } from '../store/useAppStore';

const { Sider, Header, Content } = Layout;

const menuItems = [
  { key: '/overview', icon: <DashboardOutlined />, label: '综合态势' },
  { key: '/scenario-analysis', icon: <ApartmentOutlined />, label: '场景与异构分析' },
  { key: '/experiments/new', icon: <ExperimentOutlined />, label: '实验配置' },
  { key: '/experiments/demo/live', icon: <ControlOutlined />, label: '运行监控' },
  { key: '/experiments/demo/privacy', icon: <SafetyCertificateOutlined />, label: '隐私攻击效果' },
  { key: '/experiments/demo/compare', icon: <BarChartOutlined />, label: '对照分析' },
  { key: '/reports', icon: <FileTextOutlined />, label: '实验报告' },
  { key: '/settings', icon: <SettingOutlined />, label: '系统设置' },
];

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark"><span /><span /><span /></div>
      <div>
        <div className="brand-title">全域智汇</div>
        <div className="brand-subtitle">FEDERATED INTELLIGENCE</div>
      </div>
    </div>
  );
}

export function AppShell({ children }: PropsWithChildren) {
  const navigate = useNavigate();
  const location = useLocation();
  const { phaseIndex, round, running, toggleRunning, applyTrainingEvent, setConnectionState, dataSource, setDataSource } = useAppStore();
  const [navigationOpen, setNavigationOpen] = useState(false);
  const immersive = location.pathname === '/overview';

  useEffect(() => {
    if (!running) return;
    const current = useAppStore.getState();
    const adapter = resolveTrainingDataAdapter({
      experimentId: 'demo',
      sequence: current.lastSequence,
      phaseIndex: current.phaseIndex,
      round: current.round,
      status: 'running',
      clients: {},
      source: 'frontend_simulation',
      updatedAt: new Date().toISOString(),
    });
    setDataSource(adapter.source);
    const subscription = adapter.subscribe(
      'demo',
      current.lastSequence,
      applyTrainingEvent,
      setConnectionState,
    );
    return () => subscription.close();
  }, [running, applyTrainingEvent, setConnectionState, setDataSource]);

  const selectedKey = menuItems
    .map((item) => item.key)
    .find((key) => location.pathname === key) ??
    menuItems.find((item) => location.pathname.startsWith(item.key.replace('/demo', '')))?.key ??
    '/overview';

  const navigation = (
    <>
      <Brand />
      <div className="system-label">全域联邦学习演示平台</div>
      <Menu
        mode="inline"
        selectedKeys={[selectedKey]}
        items={menuItems}
        onClick={({ key }) => {
          navigate(key);
          setNavigationOpen(false);
        }}
        className="side-menu"
      />
      <div className="sidebar-status">
        <div className="sidebar-status-head"><span>单机模拟引擎</span><Badge status="processing" /></div>
        <div className="engine-row"><span>数据源</span><Tag color="cyan">{dataSource === 'backend' ? 'BACKEND' : 'DEMO'}</Tag></div>
        <div className="engine-row"><span>逻辑客户端</span><strong>60</strong></div>
        <div className="engine-row"><span>数据域</span><strong>4</strong></div>
      </div>
    </>
  );

  return (
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#39c9ff',
          colorBgBase: '#07101f',
          colorBgContainer: '#0e1c2d',
          colorBorder: 'rgba(125, 177, 220, .16)',
          borderRadius: 8,
          fontFamily: "Inter, 'PingFang SC', 'Microsoft YaHei', sans-serif",
        },
      }}
    >
      <Layout className={`app-layout ${immersive ? 'immersive-layout' : ''}`}>
        {!immersive && <Sider width={228} className="app-sider">{navigation}</Sider>}
        {immersive && <>
          <Button className="overview-menu-trigger" type="text" icon={<MenuOutlined />} aria-label="打开导航" onClick={() => setNavigationOpen(true)} />
          <Drawer className="navigation-drawer" placement="left" width={260} open={navigationOpen} onClose={() => setNavigationOpen(false)} closable={false}>{navigation}</Drawer>
        </>}
        <Layout>
          {!immersive && <Header className="app-header">
            <div className="header-context">
              <span className="context-dot" />
              <span className="context-name">跨域联合认知演示 / 安全防御场景</span>
              <Tag bordered={false} color="processing">运行中</Tag>
            </div>
            <Space size={18}>
              <div className="header-kv"><span>当前阶段</span><strong>{phases[phaseIndex]}</strong></div>
              <div className="header-divider" />
              <div className="header-kv"><span>全局轮次</span><strong>{round} / 30</strong></div>
              <Button type={running ? 'default' : 'primary'} onClick={toggleRunning}>
                {running ? '暂停演示' : '继续演示'}
              </Button>
            </Space>
          </Header>}
          <Content className={`app-content ${immersive ? 'immersive-content' : ''}`}>{children}</Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}
