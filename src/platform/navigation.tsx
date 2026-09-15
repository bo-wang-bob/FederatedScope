import { AppstoreOutlined, ExperimentOutlined, LineChartOutlined, ScanOutlined } from '@ant-design/icons';
import { plannedModules } from './extensions';
export const mainPages = {
  home: { label: '系统首页', icon: <AppstoreOutlined /> },
  train: { label: '训练实验', icon: <ExperimentOutlined /> },
  experience: { label: '模型验证', icon: <ScanOutlined /> },
  compare: { label: '算法对比', icon: <LineChartOutlined /> },
} as const;
export const pages = {
  home: { label: '系统首页', section: '工作台' }, train: { label: '训练实验', section: '工作台' },
  jobs: { label: '实验记录', section: '工作台' }, experience: { label: '模型验证', section: '工作台' },
  evaluate: { label: '独立评测', section: '工作台' }, compare: { label: '算法对比', section: '工作台' },
  backdoor: { label: '后门研究', section: '研究扩展' },
  ...plannedModules,
} as const;
export type PlatformView = keyof typeof pages;
export type MainView = keyof typeof mainPages;
export function mainView(view: PlatformView): MainView | 'privacy' | 'backdoor' {
  return view === 'jobs' ? 'train' : view === 'evaluate' ? 'experience' : view as MainView | 'privacy' | 'backdoor';
}
export const navigationItems = Object.entries(mainPages).map(([key, page]) => ({ key, label: page.label, icon: page.icon }));
const legacyViews: Record<string, PlatformView> = { '/overview': 'home', '/scenario-analysis': 'train', '/experiments/new': 'train', '/reports': 'jobs' };
export function resolveView(query: string | null, pathname: string): PlatformView {
  if (query === 'cache') return 'train';
  if (query === 'overview') return 'home';
  if (query) return Object.hasOwn(pages, query) ? query as PlatformView : 'home';
  return legacyViews[pathname] || 'home';
}
export function viewHref(view: PlatformView): string { return view === 'home' ? '/' : '/?view=' + view; }
export const jobHref = (id: string) => '/?' + new URLSearchParams({ view: 'jobs', id });
export const modelHref = (id: string, evaluate = false, testset?: string) => '/?' + new URLSearchParams({ view: evaluate ? 'evaluate' : 'experience', model: id, ...(testset ? {testset} : {}) });
