import { ArrowRightOutlined, LockOutlined, SecurityScanOutlined } from '@ant-design/icons';
import { Alert, Tag } from 'antd';
import { Link } from 'react-router-dom';

// Presentation-only reservations. No task action, API endpoint or metric is enabled here.
export const plannedModules = {
  privacy: {
    label: '隐私保护', section: '隐私与安全', status: 'planned', icon: <LockOutlined />,
    description: '预留隐私保护实验、效果评测与结果对比空间。',
    areas: [
      { title: '实验配置', description: '保护策略及相关参数，具体方法后续接入。' },
      { title: '训练与评测', description: '后续复用训练任务、模型和测试集选择流程。' },
      { title: '结果分析', description: '隐私指标、准确率影响与运行开销的展示位置。' },
    ],
  },
  backdoor: {
    label: '后门攻防', section: '隐私与安全', status: 'planned', icon: <SecurityScanOutlined />,
    description: '预留后门场景、攻防实验与鲁棒性评测空间。',
    areas: [
      { title: '实验配置', description: '攻击场景与防御策略，具体方案后续接入。' },
      { title: '训练与评测', description: '隔离实验数据，关联模型、配置与测试记录。' },
      { title: '结果分析', description: '干净准确率、攻击成功率及防御效果的展示位置。' },
    ],
  },
} as const;

export type PlannedModuleId = keyof typeof plannedModules;
export function isPlannedView(view: string): view is PlannedModuleId {
  return Object.hasOwn(plannedModules, view);
}

export function FutureModuleEntries() {
  return <section className="home-extensions" aria-labelledby="home-extensions-title">
    <div className="home-extensions-heading"><h2 id="home-extensions-title">隐私与安全</h2><span>后续扩展 · 尚未接入</span></div>
    <div className="home-extension-grid">{Object.entries(plannedModules).map(([id, module]) => <Link className={`home-extension-entry extension-${id}`} to={`/?view=${id}`} key={id}>
      <span className="home-extension-icon">{module.icon}</span><span><strong>{module.label}<Tag>规划中</Tag></strong><small>{module.description}</small></span><ArrowRightOutlined />
    </Link>)}</div>
  </section>;
}

export function PlannedModule({ moduleId }: { moduleId: PlannedModuleId }) {
  const module = plannedModules[moduleId];
  return <section className={`planned-module extension-${moduleId}`} aria-label={`${module.label}规划说明`}>
    <Alert type="info" showIcon title="规划中 · 尚未接入真实任务" description="当前仅预留页面与流程结构，不会启动训练、修改数据或生成指标；现有准确率实验照常使用。" />
    <div className="planned-module-heading"><span className="home-extension-icon">{module.icon}</span><div><h2>计划接入范围</h2><p>以下为后续模块位置，不代表相关能力已经实现。</p></div></div>
    <div className="planned-module-areas">{module.areas.map(area => <article key={area.title}><Tag>待接入</Tag><h3>{area.title}</h3><p>{area.description}</p></article>)}</div>
    <div className="planned-module-footer"><p>后续接入时沿用现有任务记录与资源清理机制，实验配置和结果按模块隔离。</p><Link to="/">返回系统首页 <ArrowRightOutlined /></Link></div>
  </section>;
}
