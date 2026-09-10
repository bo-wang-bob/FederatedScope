import { useEffect } from 'react';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { Button, ConfigProvider, Tag, theme } from 'antd';
import { OverviewPage } from './OverviewPage';
import { phases, useAppStore } from '../store/useAppStore';
import '../styles/map-demo.css';

// Deliberately independent of both API clients and all backend event streams.
export default function MapDemoPage() {
  const running = useAppStore(state => state.running);
  useEffect(() => {
    const previousTitle = document.title;
    document.title = '全域智汇 · 地图模拟演示';
    return () => { document.title = previousTitle; };
  }, []);
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => useAppStore.setState(state => ({
      phaseIndex: (state.phaseIndex + 1) % phases.length,
      round: state.phaseIndex === phases.length - 1 ? state.round % 30 + 1 : state.round,
    })), 3000);
    return () => clearInterval(timer);
  }, [running]);

  return <ConfigProvider theme={{ algorithm: theme.darkAlgorithm, token: {
    colorPrimary: '#39c9ff', colorBgBase: '#07101f', colorBgContainer: '#0e1c2d',
    colorBorder: 'rgba(125,177,220,.16)', borderRadius: 8,
  } }}>
    <div className={`map-demo-surface ${running ? '' : 'map-demo-paused'}`}>
      <header className="map-demo-modebar">
        <div><Tag color="gold">模拟演示</Tag><span>纯前端模拟数据，不会启动真实训练</span></div>
        <Button href="/" icon={<ArrowLeftOutlined />}>返回系统首页</Button>
      </header>
      <OverviewPage simulationOnly />
    </div>
  </ConfigProvider>;
}
