import { ArrowRightOutlined, LockOutlined, SecurityScanOutlined } from '@ant-design/icons';
import { Tag } from 'antd';
import { Link } from 'react-router-dom';

// UI reservations only. Real execution requires a separate backend capability.
export const plannedModules = {
  privacy: { label: '隐私保护', section: '研究扩展', status: 'planned', icon: <LockOutlined />,
    areas: ['保护策略', '隐私评测', '效用对比'] },
  backdoor: { label: '后门攻防', section: '研究扩展', status: 'planned', icon: <SecurityScanOutlined />,
    areas: ['攻防配置', '鲁棒性评测', '效果对比'] },
} as const;
export type PlannedModuleId = keyof typeof plannedModules;
export function isPlannedView(view: string): view is PlannedModuleId { return Object.hasOwn(plannedModules, view); }
export function FutureModuleEntries() {
  return <section className="studio-extensions" aria-label="研究扩展">{Object.entries(plannedModules).map(([id, module]) => <Link to={`/?view=${id}`} key={id}><span>{module.icon}</span><strong>{module.label}</strong><Tag>规划中</Tag><ArrowRightOutlined /></Link>)}</section>;
}
export function PlannedModule({ moduleId }: { moduleId: PlannedModuleId }) {
  const module = plannedModules[moduleId];
  return <section className="studio-planned" aria-label={`${module.label}规划说明`}><span className="studio-planned-icon">{module.icon}</span><Tag>规划中</Tag><h2>{module.label}</h2><p>功能预留，尚未接入真实任务。</p><div className="studio-planned-areas">{module.areas.map(area => <span key={area}>{area}<small>待接入</small></span>)}</div><Link to="/">返回首页 <ArrowRightOutlined /></Link></section>;
}
