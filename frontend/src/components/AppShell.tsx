import {
  ApartmentOutlined,
  BarChartOutlined,
  ControlOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  RadarChartOutlined,
  SafetyCertificateOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { Badge, Button, ConfigProvider, Layout, Menu, Space, Tag, theme } from 'antd';
import { useEffect, type PropsWithChildren } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { phases, useAppStore } from '../store/useAppStore';

const { Sider, Header, Content } = Layout;

const menuItems = [
  { key: '/overview', icon: <DashboardOutlined />, label: '综合态势' },
  { key: '/scenario', icon: <ApartmentOutlined />, label: '场景编排' },
  { key: '/heterogeneity', icon: <RadarChartOutlined />, label: '异构分析' },
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
  const { phaseIndex, round, running, nextPhase, toggleRunning } = useAppStore();

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(nextPhase, 3200);
    return () => window.clearInterval(timer);
  }, [running, nextPhase]);

  const selectedKey = menuItems
    .map((item) => item.key)
    .find((key) => location.pathname === key) ??
    menuItems.find((item) => location.pathname.startsWith(item.key.replace('/demo', '')))?.key ??
    '/overview';

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
      <Layout className="app-layout">
        <Sider width={228} className="app-sider">
          <Brand />
          <div className="system-label">全域联邦学习演示平台</div>
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            items={menuItems}
            onClick={({ key }) => navigate(key)}
            className="side-menu"
          />
          <div className="sidebar-status">
            <div className="sidebar-status-head"><span>单机模拟引擎</span><Badge status="processing" /></div>
            <div className="engine-row"><span>数据源</span><Tag color="cyan">DEMO</Tag></div>
            <div className="engine-row"><span>逻辑节点</span><strong>20</strong></div>
            <div className="engine-row"><span>军事域</span><strong>4</strong></div>
          </div>
        </Sider>
        <Layout>
          <Header className="app-header">
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
          </Header>
          <Content className="app-content">{children}</Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}
