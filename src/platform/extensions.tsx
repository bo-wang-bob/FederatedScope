import { ArrowRightOutlined, LockOutlined } from '@ant-design/icons';
import { Tag } from 'antd';
import { Link } from 'react-router-dom';

// UI reservations only. Real execution requires a separate backend capability.
// 后门研究已接入（src/platform/backdoor.tsx），保留在此会把它当未接入模块处理。
export const plannedModules = {
  privacy: { label: '隐私研究', section: '研究扩展', status: 'planned', icon: <LockOutlined />,
    areas: ['保护策略', '隐私评测', '效用对比'] },
} as const;
export type PlannedModuleId = keyof typeof plannedModules;
export function isPlannedView(view: string): view is PlannedModuleId { return Object.hasOwn(plannedModules, view); }
export function FutureModuleEntries() {
  return <section className="studio-extensions" aria-label="研究扩展">{Object.entries(plannedModules).map(([id, module]) => <Link to={`/?view=${id}`} key={id}><span>{module.icon}</span><strong>{module.label}</strong><Tag>未接入</Tag><ArrowRightOutlined /></Link>)}</section>;
}
export function PlannedModule({ moduleId }: { moduleId: PlannedModuleId }) {
  const module = plannedModules[moduleId];
  return <section className="design-reserved" aria-label={`${module.label}规划说明`}>{module.icon}<h2>未接入</h2></section>;
}
