import { ApartmentOutlined, BarChartOutlined, DatabaseOutlined, ExperimentOutlined,
  FileSearchOutlined, GlobalOutlined, HomeOutlined, HistoryOutlined, ScanOutlined } from '@ant-design/icons';
import type { MenuProps } from 'antd';

export const pages = {
  home: { label: '系统首页', section: '工作台', description: '从模型体验出发，连接训练、评测与实验记录。', icon: <HomeOutlined /> },
  overview: { label: '运行总览', section: '工作台', description: '查看真实联邦拓扑、训练曲线与服务器资源。', icon: <ApartmentOutlined /> },
  experience: { label: '模型体验台', section: '模型验证', description: '选择已保存模型和测试图片，查看真实预测结果。', icon: <ScanOutlined /> },
  evaluate: { label: '独立评测', section: '模型验证', description: '选择模型与测试范围，输出总体、分域及分类指标。', icon: <FileSearchOutlined /> },
  compare: { label: '结果对比', section: '模型验证', description: '核对实验条件，对比不同算法表现并导出结果。', icon: <BarChartOutlined /> },
  train: { label: '新建训练', section: '训练实验', description: '配置算法与超参数，完成缓存预检后启动训练。', icon: <ExperimentOutlined /> },
  jobs: { label: '任务与记录', section: '训练实验', description: '跟进正在执行的任务，查看历史日志、模型与评测产物。', icon: <HistoryOutlined /> },
  cache: { label: '数据与缓存', section: '训练实验', description: '查看已有特征缓存与预检记录，选择实验配置。', icon: <DatabaseOutlined /> },
} as const;

export type PlatformView = keyof typeof pages;
export const navigationItems: MenuProps['items'] = [
  ...['工作台', '模型验证', '训练实验'].map(section => ({
    type: 'group' as const, key: section, label: section,
    children: Object.entries(pages).filter(([, page]) => page.section === section)
      .map(([key, page]) => ({ key, label: page.label, icon: page.icon })),
  })),
  { type: 'group', key: '模拟演示', label: '模拟演示', children: [
    { key: 'demo', icon: <GlobalOutlined />, label: <span className="platform-demo-menu-label">地图模拟演示<small>演示</small></span> },
  ] },
];

const legacyViews: Record<string, PlatformView> = {
  '/overview': 'overview', '/scenario-analysis': 'cache', '/experiments/new': 'train', '/reports': 'jobs',
};

export function resolveView(query: string | null, pathname: string): PlatformView {
  if (query) return Object.hasOwn(pages, query) ? query as PlatformView : 'home';
  return legacyViews[pathname] || 'home';
}

export function viewHref(view: PlatformView): string {
  return view === 'home' ? '/' : `/?view=${view}`;
}
