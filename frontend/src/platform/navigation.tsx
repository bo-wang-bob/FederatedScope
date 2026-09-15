import { AppstoreOutlined, ExperimentOutlined, LineChartOutlined, ScanOutlined } from '@ant-design/icons';
import { plannedModules } from './extensions';

export const mainPages = {
  home: { label: '首页', icon: <AppstoreOutlined /> },
  train: { label: '训练实验', icon: <ExperimentOutlined /> },
  experience: { label: '模型验证', icon: <ScanOutlined /> },
  compare: { label: '结果分析', icon: <LineChartOutlined /> },
} as const;
export const pages = {
  home: { label: '首页', section: '工作台' },
  train: { label: '训练实验', section: '工作台' },
  jobs: { label: '训练实验', section: '工作台' },
  cache: { label: '训练实验', section: '工作台' },
  overview: { label: '训练实验', section: '工作台' },
  experience: { label: '模型验证', section: '工作台' },
  evaluate: { label: '模型验证', section: '工作台' },
  compare: { label: '结果分析', section: '工作台' },
  ...plannedModules,
} as const;
export type PlatformView = keyof typeof pages;
export type MainView = keyof typeof mainPages;
export function mainView(view: PlatformView): MainView | 'privacy' | 'backdoor' {
  return ['jobs', 'cache', 'overview'].includes(view) ? 'train' : view === 'evaluate' ? 'experience' : view as MainView | 'privacy' | 'backdoor';
}
export const navigationItems = Object.entries(mainPages).map(([key, page]) => ({ key, label: page.label, icon: page.icon }));
const legacyViews: Record<string, PlatformView> = { '/overview': 'overview', '/scenario-analysis': 'cache', '/experiments/new': 'train', '/reports': 'jobs' };
export function resolveView(query: string | null, pathname: string): PlatformView {
  if (query) return Object.hasOwn(pages, query) ? query as PlatformView : 'home';
  return legacyViews[pathname] || 'home';
}
export function viewHref(view: PlatformView): string { return view === 'home' ? '/' : `/?view=${view}`; }
export const jobHref = (id: string) => `/?${new URLSearchParams({ view: 'jobs', id })}`;
