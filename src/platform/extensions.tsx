import { ArrowRightOutlined, BarChartOutlined, CheckCircleFilled, ExperimentOutlined, InfoCircleOutlined,
  LockOutlined, SafetyCertificateOutlined } from '@ant-design/icons';
import { Alert, Popover, Select, Skeleton, Tag } from 'antd';
import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from './api';
import { PrivacyLab } from './privacyLab';

// UI reservations only. Real execution requires a separate backend capability.
// 后门研究已接入（src/platform/backdoor.tsx），保留在此会把它当未接入模块处理。
export const plannedModules = {
  privacy: { label: '隐私保护', section: '研究扩展', status: 'planned', icon: <LockOutlined />,
    areas: ['保护策略', '隐私评测', '效用对比'] },
} as const;
export type PlannedModuleId = keyof typeof plannedModules;
export function isPlannedView(view: string): view is PlannedModuleId { return Object.hasOwn(plannedModules, view); }
export function FutureModuleEntries() {
  return <section className="studio-extensions" aria-label="研究扩展">{Object.entries(plannedModules).map(([id, module]) => <Link to={`/?view=${id}`} key={id}><span>{module.icon}</span><strong>{module.label}</strong><Tag>{id === 'privacy' ? '成员推理' : '未接入'}</Tag><ArrowRightOutlined /></Link>)}</section>;
}

type MembershipRecord = {
  id: string;
  truth: 'member' | 'nonmember';
  domain: string;
  className: string;
  filename: string;
  imageUrl: string;
  noDefenseScore: number;
  defenseScore: number;
  noDefensePrediction: 'member' | 'nonmember';
  defensePrediction: 'member' | 'nonmember';
};
type MembershipMetrics = {
  auc: number;
  tprAt1Fpr: number;
  fprAtThreshold: number;
};
type MembershipDistribution = {
  bins: number[];
  member: number[];
  nonmember: number[];
  memberMean: number;
  nonmemberMean: number;
  meanGap: number;
  memberSamples: number;
  nonmemberSamples: number;
};
type MembershipPayload = {
  configured: boolean;
  dataset?: string;
  clientId?: number;
  clients?: number[];
  group?: 'member' | 'nonmember';
  source: string | null;
  message?: string;
  expectedPath?: string;
  alignment?: { message: string; memberSamples: number; nonmemberSamples: number; metricsMode?: string } | null;
  metrics?: {
    noDefense: MembershipMetrics;
    defense: MembershipMetrics;
  };
  distributions?: {
    noDefense: MembershipDistribution;
    defense: MembershipDistribution;
  };
  items: MembershipRecord[];
};
const membershipStatus = {
  member: { label: '成员', tone: 'member' },
  nonmember: { label: '非成员', tone: 'nonmember' },
} as const;
const formatScore = (value?: number) => Number.isFinite(value) ? value!.toFixed(4) : '—';
const formatPercent = (value?: number) => Number.isFinite(value) ? `${(value! * 100).toFixed(2)}%` : '—';

function PrivacyDistributionPlot({ distribution, label, mode }: {
  distribution?: MembershipDistribution;
  label: string;
  mode: 'noDefense' | 'defense';
}) {
  if (!distribution || distribution.member.length === 0 || distribution.nonmember.length === 0) {
    return <article className={`privacy-distribution-plot ${mode}`}>
      <header><strong>{label}</strong><span>暂无分布数据</span></header>
      <div className="privacy-chart-empty">暂无分布数据</div>
    </article>;
  }
  const width = 340;
  const height = 190;
  const left = 36;
  const right = 12;
  const top = 17;
  const bottom = 34;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const count = Math.min(distribution.member.length, distribution.nonmember.length);
  const maxY = Math.max(...distribution.member, ...distribution.nonmember, 0.01);
  const barWidth = plotWidth / Math.max(count, 1);
  const id = mode === 'defense' ? 'defense' : 'plain';
  return <article className={`privacy-distribution-plot ${mode}`}>
    <header><strong>{label}</strong><span>分数分布对照</span></header>
    <svg viewBox={`0 0 ${width} ${height}`} aria-label={`${label}训练样本与非训练样本分数分布`}>
      <defs>
        <linearGradient id={`privacy-chart-bg-${id}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#142425" /><stop offset="1" stopColor="#0d181a" /></linearGradient>
        <linearGradient id={`privacy-member-bar-${id}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#67b9ed" /><stop offset="1" stopColor="#438ec0" /></linearGradient>
        <linearGradient id={`privacy-nonmember-bar-${id}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#73d5b7" /><stop offset="1" stopColor="#419c83" /></linearGradient>
      </defs>
      <rect width={width} height={height} rx="10" fill={`url(#privacy-chart-bg-${id})`} />
      {[0, 0.5, 1].map(tick => {
        const x = left + tick * plotWidth;
        return <g key={`x-${tick}`}>
          <line x1={x} y1={top} x2={x} y2={top + plotHeight} stroke="#29413f" strokeDasharray="3 5" />
          <text x={x} y={height - 14} textAnchor="middle">{tick.toFixed(1)}</text>
        </g>;
      })}
      {[0, 0.5, 1].map(tick => {
        const y = top + plotHeight - tick * plotHeight;
        return <line key={`y-${tick}`} x1={left} y1={y} x2={left + plotWidth} y2={y} stroke="#29413f" strokeDasharray="3 5" />;
      })}
      {Array.from({ length: count }, (_, index) => {
        const x = left + index * barWidth;
        const memberHeight = distribution.member[index] / maxY * plotHeight;
        const nonmemberHeight = distribution.nonmember[index] / maxY * plotHeight;
        return <g key={index}>
          <rect x={x} y={top + plotHeight - memberHeight} width={barWidth * 0.9} height={memberHeight}
            fill={`url(#privacy-member-bar-${id})`} opacity="0.72" />
          <rect x={x} y={top + plotHeight - nonmemberHeight} width={barWidth * 0.9} height={nonmemberHeight}
            fill={`url(#privacy-nonmember-bar-${id})`} opacity="0.64" />
        </g>;
      })}
      <line x1={left} y1={top + plotHeight} x2={left + plotWidth} y2={top + plotHeight} stroke="#56706b" />
      <line x1={left} y1={top} x2={left} y2={top + plotHeight} stroke="#56706b" />
      <text x={left + plotWidth} y={height - 3} textAnchor="end" className="axis-title">攻击分数</text>
    </svg>
    <div className="privacy-distribution-meta">
      <div><span>训练样本均值</span><strong>{formatScore(distribution.memberMean)}</strong></div>
      <div><span>非训练样本均值</span><strong>{formatScore(distribution.nonmemberMean)}</strong></div>
      <div className="gap"><span>均值差值</span><strong>{formatScore(distribution.meanGap)}</strong><small>越大越容易区分</small></div>
    </div>
  </article>;
}

function PrivacyDistributionChart({ distributions }: { distributions?: MembershipPayload['distributions'] }) {
  return <section className="privacy-distribution-card">
    <div className="privacy-card-heading"><span className="privacy-heading-icon"><BarChartOutlined /></span><div><h3>攻击分数分布</h3><p>防御前后分数可分性对照</p></div><em>Client 对照</em></div>
    <div className="privacy-chart-legend"><span><i className="training" />训练样本</span><span><i className="nontraining" />非训练样本</span></div>
    <div className="privacy-distribution-plots" role="region" aria-label="攻击分数分布图表" tabIndex={0}>
      <PrivacyDistributionPlot distribution={distributions?.noDefense} label="无防御" mode="noDefense" />
      <PrivacyDistributionPlot distribution={distributions?.defense} label="有防御" mode="defense" />
    </div>
  </section>;
}

export function PrivacyMembershipPanel() {
  const [group, setGroup] = useState<'member' | 'nonmember'>('member');
  const [clientId, setClientId] = useState<number>();
  const [payload, setPayload] = useState<MembershipPayload>();
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState('');
  const payloadCache = useRef(new Map<string, MembershipPayload>());
  const records = payload?.items || [];
  const selected = records.find(record => record.id === selectedId) || records[0];

  useEffect(() => {
    let alive = true;
    const cacheKey = `${clientId ?? 'auto'}:${group}`;
    const cached = payloadCache.current.get(cacheKey);
    if (cached) {
      setPayload(cached);
      setError('');
      setSelectedId(cached.items[0]?.id || '');
      setLoading(false);
      return () => { alive = false; };
    }
    const query = new URLSearchParams({ group, limit: '20', seed: '2026' });
    if (clientId != null) query.set('clientId', String(clientId));
    setLoading(true);
    api<MembershipPayload>(`privacy/membership?${query.toString()}`)
      .then(data => {
        if (!alive) return;
        payloadCache.current.set(cacheKey, data);
        if (data.clientId != null) {
          payloadCache.current.set(`${data.clientId}:${data.group || group}`, data);
        }
        setPayload(data);
        setError('');
        if (clientId == null && data.clientId != null) setClientId(data.clientId);
        setSelectedId(data.items[0]?.id || '');
      })
      .catch(reason => alive && setError(String(reason?.message || reason)))
      .finally(() => alive && setLoading(false));
    return () => { alive = false; };
  }, [clientId, group]);

  const changeGroup = (nextGroup: 'member' | 'nonmember') => {
    setGroup(nextGroup);
    setSelectedId('');
  };

  if (error) return <section className="privacy-lab-shell studio-page-enter">
    <Alert type="error" showIcon message="成员推理结果读取失败" description={error} />
  </section>;

  if (!payload || loading) return <section className="privacy-lab-shell studio-page-enter">
    <Skeleton active paragraph={{ rows: 8 }} />
  </section>;

  if (!payload.configured || !records.length) return <section className="privacy-lab-shell studio-page-enter">
    <Alert type="warning" showIcon message={payload.message || '隐私模块资源未就绪'}
      description={payload.expectedPath ? <details><summary>资源配置</summary>
        <p>需补充 FedMIA 脚本、攻防特征、配置和对应数据集图片。</p><code>{payload.expectedPath}</code>
      </details> : undefined} />
  </section>;
  if (!selected) return null;
  const groupLabel = group === 'member' ? '客户端训练样本' : '非训练样本';
  const comparisons = [
    { key: 'noDefense', label: '无防御', metrics: payload.metrics?.noDefense,
      prediction: selected.noDefensePrediction, score: selected.noDefenseScore },
    { key: 'defense', label: '有防御', metrics: payload.metrics?.defense,
      prediction: selected.defensePrediction, score: selected.defenseScore },
  ] as const;

  return <section className="privacy-lab-shell studio-page-enter">
    <div className="privacy-control-bar">
      <div className="privacy-control-group">
        <span className="privacy-control-label">样本集合</span>
        <div className="privacy-mode-tabs" role="tablist" aria-label="成员状态">
          <button className={group === 'member' ? 'active' : ''} onClick={() => changeGroup('member')}>客户端训练样本</button>
          <button className={group === 'nonmember' ? 'active' : ''} onClick={() => changeGroup('nonmember')}>非训练样本</button>
        </div>
      </div>
      <label className="privacy-client-select">
        <span className="privacy-control-label">目标客户端</span>
        <Select value={payload.clientId} onChange={value => setClientId(value)}
          options={(payload.clients || []).map(id => ({ value: id, label: `Client ${id}` }))} />
      </label>
    </div>
    <div className="privacy-lab">
      <section className="privacy-picker" aria-label="成员推理样本">
        <div className="privacy-section-title"><div><h2>样本选择</h2><p>{groupLabel}</p></div><span>{payload.dataset || '数据集'} · {records.length} 个</span></div>
        <div className="privacy-sample-list">
          {records.map(record => <button key={record.id} className={record.id === selected.id ? 'selected' : ''}
            onClick={() => setSelectedId(record.id)} aria-pressed={record.id === selected.id}>
            <img src={record.imageUrl} alt={`${record.className || record.filename} ${membershipStatus[record.truth].label}`} />
            <span><strong>{record.className || record.filename || record.id}</strong><small>{record.domain || payload.dataset || '数据集'}</small></span>
          </button>)}
        </div>
      </section>
      <PrivacyDistributionChart distributions={payload.distributions} />
      <section className="privacy-result" aria-label="成员推理攻击结果">
        <div className="privacy-card-heading privacy-result-heading"><span className="privacy-heading-icon"><ExperimentOutlined /></span><div><h3 className="privacy-result-title">攻击结果对照{payload.alignment && <Popover title="结果口径" content={<p className="privacy-result-note">{payload.alignment.message}</p>} trigger="click"><button className="privacy-info-button" type="button" aria-label="查看结果口径"><InfoCircleOutlined /></button></Popover>}</h3><p>{payload.alignment?.metricsMode === 'mix' ? '整体指标与当前样本预测' : payload.alignment ? '共同样本指标与当前样本预测' : '整体指标与当前样本预测'}</p></div><em>Client {payload.clientId}</em></div>
        <div className="privacy-selected-summary">
          <img src={selected.imageUrl} alt={selected.className || selected.filename} />
          <div>
            <span className="privacy-eyebrow">当前样本</span>
            <strong>{selected.className || selected.filename || selected.id}</strong>
            <p><b className={membershipStatus[selected.truth].tone}>{selected.truth === 'member' ? '客户端训练样本' : '非训练样本'}</b><i>{selected.domain || payload.dataset || '数据集'}</i></p>
          </div>
        </div>
        <div className="privacy-comparison-list" role="region" aria-label="攻防指标对照" tabIndex={0}>
          {comparisons.map(item => {
            const correct = item.prediction === selected.truth;
            return <article className={`privacy-comparison-card ${item.key}`} key={item.key}>
              <header><span className="privacy-heading-icon"><SafetyCertificateOutlined /></span><div><strong>{item.label}</strong><small>FedMIA 攻击</small></div></header>
              <div className="privacy-comparison-metrics">
                <div><span>AUC</span><strong>{formatPercent(item.metrics?.auc)}</strong></div>
                <div><span>TPR@FPR≤1%</span><strong>{formatPercent(item.metrics?.tprAt1Fpr)}</strong></div>
              </div>
              <div className={`privacy-comparison-prediction ${correct ? 'attack-correct' : 'attack-wrong'}`}>
                <div><span>攻击预测结果</span><strong>{membershipStatus[item.prediction].label}</strong></div>
                <div><span>攻击分数</span><strong>{formatScore(item.score)}</strong></div>
              </div>
              <div className={`privacy-result-verdict ${correct ? 'correct' : 'wrong'}`}>
                <CheckCircleFilled /><span>{correct ? '攻击判断与真实状态一致' : '攻击判断与真实状态不一致'}</span>
              </div>
            </article>;
          })}
        </div>
      </section>
    </div>
  </section>;
}

export function PlannedModule({ moduleId }: { moduleId: PlannedModuleId }) {
  const module = plannedModules[moduleId];
  if (moduleId === 'privacy') return <PrivacyLab />;
  return <section className="design-reserved" aria-label={`${module.label}规划说明`}>{module.icon}<h2>未接入</h2></section>;
}
