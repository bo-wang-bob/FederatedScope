import { ArrowRightOutlined, LockOutlined, SecurityScanOutlined } from '@ant-design/icons';
import { Alert, Select, Skeleton, Tag } from 'antd';
import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from './api';

// UI reservations only. Real execution requires a separate backend capability.
export const plannedModules = {
  privacy: { label: '隐私研究', section: '研究扩展', status: 'planned', icon: <LockOutlined />,
    areas: ['保护策略', '隐私评测', '效用对比'] },
  backdoor: { label: '后门研究', section: '研究扩展', status: 'planned', icon: <SecurityScanOutlined />,
    areas: ['攻防配置', '鲁棒性评测', '效果对比'] },
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
  clientId?: number;
  clients?: number[];
  group?: 'member' | 'nonmember';
  source: string | null;
  message?: string;
  expectedPath?: string;
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
type DefenseMode = 'noDefense' | 'defense';

const membershipStatus = {
  member: { label: '成员', tone: 'member' },
  nonmember: { label: '非成员', tone: 'nonmember' },
} as const;
const formatScore = (value?: number) => Number.isFinite(value) ? value!.toFixed(4) : '—';
const formatPercent = (value?: number) => Number.isFinite(value) ? `${(value! * 100).toFixed(2)}%` : '—';

function PrivacyDistributionChart({ distribution, title }: { distribution?: MembershipDistribution; title: string }) {
  if (!distribution || distribution.member.length === 0 || distribution.nonmember.length === 0) return <section className="privacy-distribution-card">
    <div className="privacy-block-title"><h3>当前客户端分数分布</h3><span>{title}</span></div>
    <div className="privacy-chart-empty">暂无分布数据</div>
  </section>;
  const width = 360;
  const height = 230;
  const left = 34;
  const right = 12;
  const top = 18;
  const bottom = 32;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const count = Math.min(distribution.member.length, distribution.nonmember.length);
  const maxY = Math.max(...distribution.member, ...distribution.nonmember, 0.01);
  const barWidth = plotWidth / Math.max(count, 1);
  const bars = Array.from({ length: count }, (_, index) => {
    const x = left + index * barWidth;
    const memberHeight = distribution.member[index] / maxY * plotHeight;
    const nonmemberHeight = distribution.nonmember[index] / maxY * plotHeight;
    return <g key={index}>
      <rect x={x} y={top + plotHeight - memberHeight} width={barWidth * 0.9}
        height={memberHeight} fill="#5bb7e5" opacity="0.64" />
      <rect x={x} y={top + plotHeight - nonmemberHeight} width={barWidth * 0.9}
        height={nonmemberHeight} fill="#65c8ae" opacity="0.64" />
    </g>;
  });
  return <section className="privacy-distribution-card">
    <div className="privacy-block-title"><h3>当前客户端分数分布</h3><span>{title}</span></div>
    <svg viewBox={`0 0 ${width} ${height}`} aria-label="当前客户端训练样本与非训练样本分数分布">
      <rect x="0" y="0" width={width} height={height} rx="8" fill="#e9eaf2" />
      {[0, 0.5, 1].map(tick => {
        const x = left + tick * plotWidth;
        return <g key={`x-${tick}`}>
          <line x1={x} y1={top} x2={x} y2={top + plotHeight} stroke="#fff" />
          <text x={x} y={height - 11} textAnchor="middle">{tick.toFixed(1)}</text>
        </g>;
      })}
      {[0, 0.5, 1].map(tick => {
        const y = top + plotHeight - tick * plotHeight;
        return <g key={`y-${tick}`}>
          <line x1={left} y1={y} x2={left + plotWidth} y2={y} stroke="#fff" />
          <text x={left - 6} y={y + 4} textAnchor="end">{tick === 1 ? 'max' : tick.toFixed(1)}</text>
        </g>;
      })}
      {bars}
      <line x1={left} y1={top + plotHeight} x2={left + plotWidth} y2={top + plotHeight} stroke="#26313a" />
      <line x1={left} y1={top} x2={left} y2={top + plotHeight} stroke="#26313a" />
      <rect x={width - 112} y={18} width="14" height="8" fill="#5bb7e5" opacity="0.76" />
      <text x={width - 92} y={26}>训练样本</text>
      <rect x={width - 112} y={36} width="14" height="8" fill="#65c8ae" opacity="0.76" />
      <text x={width - 92} y={44}>非训练样本</text>
      <text x={left + plotWidth - 2} y={top + plotHeight - 7} textAnchor="end">攻击分数</text>
    </svg>
    <div className="privacy-distribution-meta">
      <span>μ训练 - μ非训练</span>
      <strong>{formatScore(distribution.meanGap)}</strong>
    </div>
  </section>;
}

function PrivacyMembershipPanel() {
  const [group, setGroup] = useState<'member' | 'nonmember'>('member');
  const [defenseMode, setDefenseMode] = useState<DefenseMode>('noDefense');
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
    <Alert type="warning" showIcon message={payload.message || '尚未放置成员推理展示包'}
      description={payload.expectedPath ? `请把服务器导出的 membership_examples.json 和 images 目录放到：${payload.expectedPath}` : undefined} />
  </section>;
  if (!selected) return null;
  const isDefenseMode = defenseMode === 'defense';
  const attackLabel = isDefenseMode ? '有防御攻击' : '无防御攻击';
  const selectedPrediction = isDefenseMode ? selected.defensePrediction : selected.noDefensePrediction;
  const selectedScore = isDefenseMode ? selected.defenseScore : selected.noDefenseScore;
  const selectedMetrics = isDefenseMode ? payload.metrics?.defense : payload.metrics?.noDefense;
  const selectedDistribution = isDefenseMode ? payload.distributions?.defense : payload.distributions?.noDefense;
  const selectedCorrect = selectedPrediction === selected.truth;

  return <section className="privacy-lab-shell studio-page-enter">
    <div className="privacy-control-bar">
      <div className="privacy-mode-tabs" role="tablist" aria-label="成员状态">
        <button className={group === 'member' ? 'active' : ''} onClick={() => changeGroup('member')}>客户端训练样本</button>
        <button className={group === 'nonmember' ? 'active' : ''} onClick={() => changeGroup('nonmember')}>非训练样本</button>
      </div>
      <div className="privacy-control-actions">
        <div className="privacy-mode-tabs" role="tablist" aria-label="防御状态">
          <button className={defenseMode === 'noDefense' ? 'active' : ''} onClick={() => setDefenseMode('noDefense')}>无防御</button>
          <button className={defenseMode === 'defense' ? 'active' : ''} onClick={() => setDefenseMode('defense')}>有防御</button>
        </div>
        <label>
          <span>目标客户端</span>
          <Select value={payload.clientId} onChange={value => setClientId(value)}
            options={(payload.clients || []).map(id => ({ value: id, label: `Client ${id}` }))} />
        </label>
      </div>
    </div>
    <div className="privacy-lab">
      <section className="privacy-picker" aria-label="成员推理样本">
        <div className="privacy-section-title"><h2>样本选择</h2><span>Office-Home</span></div>
        <div className="privacy-sample-list">
          {records.map(record => <button key={record.id} className={record.id === selected.id ? 'selected' : ''}
            onClick={() => setSelectedId(record.id)} aria-pressed={record.id === selected.id}>
            <img src={record.imageUrl} alt={`${record.className || record.filename} ${membershipStatus[record.truth].label}`} />
            <span>{record.className || record.filename || record.id}</span>
          </button>)}
        </div>
      </section>
      <PrivacyDistributionChart distribution={selectedDistribution} title={attackLabel} />
      <section className="privacy-result" aria-label="成员推理攻击结果">
        <div className="privacy-client-metrics">
          <div className="privacy-result-header"><h2>客户端整体攻击指标</h2><span>Client {payload.clientId}</span></div>
          <article className={isDefenseMode ? 'defense' : ''}>
            <span>{attackLabel}</span>
            <strong>AUC {formatPercent(selectedMetrics?.auc)}</strong>
            <b>TPR@1%FPR {formatPercent(selectedMetrics?.tprAt1Fpr)}</b>
          </article>
        </div>
        <div className="privacy-selected-summary">
          <img src={selected.imageUrl} alt={selected.className || selected.filename} />
          <div>
            <span>当前样本</span>
            <strong>{selected.className || selected.filename || selected.id}</strong>
            <p>真实状态：<b className={membershipStatus[selected.truth].tone}>{membershipStatus[selected.truth].label}</b></p>
            <p>样本域：{selected.domain || '—'}</p>
          </div>
        </div>
        <div className="privacy-panel-block">
          <div className="privacy-block-title"><h3>当前样本攻击结果</h3><span>{attackLabel}</span></div>
          <article className={`privacy-single-result-card ${selectedCorrect ? 'attack-correct' : 'attack-wrong'} ${isDefenseMode ? 'defense' : ''}`}>
            <div>
              <span>攻击预测结果</span>
              <strong>{membershipStatus[selectedPrediction].label}</strong>
            </div>
            <div>
              <span>攻击分数</span>
              <strong>{formatScore(selectedScore)}</strong>
            </div>
          </article>
        </div>
      </section>
    </div>
  </section>;
}

export function PlannedModule({ moduleId }: { moduleId: PlannedModuleId }) {
  const module = plannedModules[moduleId];
  if (moduleId === 'privacy') return <PrivacyMembershipPanel />;
  return <section className="design-reserved" aria-label={`${module.label}规划说明`}>{module.icon}<h2>未接入</h2></section>;
}
