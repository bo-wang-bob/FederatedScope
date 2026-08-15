import { memo, useMemo, type CSSProperties } from 'react';
import { Progress } from 'antd';
import mapImage from '../../../resource/4o28b0625501ad13015501ad2bfc0135.jpg';
import { domains } from '../../mock/data';
import { phases, useAppStore } from '../../store/useAppStore';
import { isCentralLinkActive, isClientLinkActive, isUpperDirection } from '../../utils/hierarchyProjection';
import type { MilitaryDomain, MilitaryNode } from '../../types';

type MapPoint = { x: number; y: number };

type RegionLayout = {
  name: string;
  terrain: string;
  code: string;
  center: MapPoint;
  label: MapPoint;
  radius: { x: number; y: number };
  server: MapPoint;
};

const centralPoint: MapPoint = { x: 51, y: 45 };

const regionLayouts: RegionLayout[] = [
  {
    name: '西北沙漠区', terrain: '沙漠 / 高原', code: 'REGION-A',
    center: { x: 23, y: 34 }, label: { x: 24, y: 18 }, radius: { x: 14, y: 14 }, server: { x: 24, y: 34 },
  },
  {
    name: '北部山地区', terrain: '山地 / 林区', code: 'REGION-B',
    center: { x: 68, y: 24 }, label: { x: 68, y: 8 }, radius: { x: 13, y: 13 }, server: { x: 67, y: 26 },
  },
  {
    name: '中部城镇区', terrain: '城镇 / 交通网', code: 'REGION-C',
    center: { x: 48, y: 64 }, label: { x: 38, y: 51 }, radius: { x: 14, y: 13 }, server: { x: 49, y: 61 },
  },
  {
    name: '东部沿海区', terrain: '沿海 / 岛链', code: 'REGION-D',
    center: { x: 78, y: 62 }, label: { x: 77, y: 42 }, radius: { x: 13, y: 17 }, server: { x: 76, y: 59 },
  },
];

function createClientPoints(layout: RegionLayout, count: number): MapPoint[] {
  return Array.from({ length: count }, (_, index) => {
    const ring = index < 7 ? 0.58 : 0.9;
    const ringIndex = index < 7 ? index : index - 7;
    const ringCount = index < 7 ? 7 : count - 7;
    const angle = (Math.PI * 2 * ringIndex) / ringCount - Math.PI / 2 + (index < 7 ? 0.18 : 0);
    return {
      x: layout.center.x + Math.cos(angle) * layout.radius.x * ring,
      y: layout.center.y + Math.sin(angle) * layout.radius.y * ring,
    };
  });
}

function linePath(from: MapPoint, to: MapPoint): string {
  const middleX = (from.x + to.x) / 2;
  const middleY = ((from.y + to.y) / 2) * 0.7;
  return `M ${from.x} ${from.y * 0.7} L ${middleX} ${middleY} L ${to.x} ${to.y * 0.7}`;
}

function centralPath(from: MapPoint, to: MapPoint, index: number): string {
  const bend = index % 2 === 0 ? -2.4 : 2.4;
  const middleX = (from.x + to.x) / 2 + bend;
  const middleY = ((from.y + to.y) / 2) * 0.7;
  const firstControlX = (from.x + middleX) / 2 + bend;
  const firstControlY = (from.y * 0.7 + middleY) / 2;
  const secondControlX = (middleX + to.x) / 2 + bend;
  const secondControlY = (middleY + to.y * 0.7) / 2;
  return `M ${from.x} ${from.y * 0.7} Q ${firstControlX} ${firstControlY} ${middleX} ${middleY} Q ${secondControlX} ${secondControlY} ${to.x} ${to.y * 0.7}`;
}

function pointStyle(point: MapPoint, color?: string): CSSProperties {
  return { left: `${point.x}%`, top: `${point.y}%`, '--domain-color': color } as CSSProperties;
}

const DomainServerMarker = memo(({ domain, layout }: { domain: MilitaryDomain; layout: RegionLayout }) => (
  <div className="map-domain-server" style={pointStyle(layout.server, domain.color)}>
    <div className="map-server-beacon"><span>▣</span><i /></div>
    <div className="map-server-card">
      <div><b>{domain.serverId}</b><em>域子服务器</em></div>
      <strong>{domain.name}</strong>
      <Progress percent={68 + domains.indexOf(domain) * 6} size="small" showInfo={false} strokeColor={domain.color} />
    </div>
  </div>
));

DomainServerMarker.displayName = 'DomainServerMarker';

const ClientMarker = memo(({ node, point, color, revealTruth, onSelect }: {
  node: MilitaryNode;
  point: MapPoint;
  color: string;
  revealTruth: boolean;
  onSelect: () => void;
}) => {
  const maliciousVisible = revealTruth && node.malicious;
  return (
    <button
      className={`map-client-marker ${node.status === '已过滤' ? 'filtered' : ''} ${node.assessment === '疑似' ? 'suspected' : ''} ${maliciousVisible ? 'malicious' : ''}`}
      style={pointStyle(point, color)}
      onClick={onSelect}
      aria-label={`${node.id}，${node.status}${maliciousVisible ? '，恶意节点' : ''}`}
      title={`${node.id} · ${node.status} · 风险 ${node.risk.toFixed(2)}`}
    >
      <span>{node.id.split('-').at(-1)}</span>
      <small>{node.id}</small>
      <i />
      {maliciousVisible && <em>恶意</em>}
      {!maliciousVisible && node.assessment === '疑似' && <em className="suspect">疑似</em>}
    </button>
  );
});

ClientMarker.displayName = 'ClientMarker';

function FlowLines({ phaseIndex }: { phaseIndex: number }) {
  const phase = phases[phaseIndex];
  const upperDirection = isUpperDirection(phase);
  const centralActive = isCentralLinkActive(phase);
  const clientActive = isClientLinkActive(phase);
  return (
    <svg className="map-flow-layer" viewBox="0 0 100 70" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <marker id="map-arrow-cyan" viewBox="0 0 12 12" refX="10" refY="6" markerUnits="userSpaceOnUse" markerWidth="1.6" markerHeight="1.6" orient="auto"><path d="M 1 1 L 11 6 L 1 11 L 3.5 6 z" fill="#44d8ff" stroke="#03101b" strokeWidth="1" /></marker>
        {domains.map((domain) => <marker key={domain.id} id={`map-arrow-${domain.id}`} viewBox="0 0 12 12" refX="10" refY="6" markerUnits="userSpaceOnUse" markerWidth="1.05" markerHeight="1.05" orient="auto"><path d="M 1 1 L 11 6 L 1 11 L 3.5 6 z" fill={domain.color} stroke="#03101b" strokeWidth="1.2" /></marker>)}
      </defs>
      {domains.map((domain, domainIndex) => {
        const layout = regionLayouts[domainIndex];
        const from = upperDirection ? layout.server : centralPoint;
        const to = upperDirection ? centralPoint : layout.server;
        const path = centralPath(from, to, domainIndex);
        return <g key={`central-${domain.id}`} data-link="central" data-direction={upperDirection ? 'uplink' : 'downlink'}>
          <path d={path} className="map-link-backbone map-central-backbone" />
          <path d={path} className={`map-central-link ${centralActive ? 'active' : ''}`} markerMid={centralActive ? 'url(#map-arrow-cyan)' : undefined} markerEnd={centralActive ? 'url(#map-arrow-cyan)' : undefined} />
        </g>;
      })}
      {domains.flatMap((domain, domainIndex) => {
        const layout = regionLayouts[domainIndex];
        const clientPoints = createClientPoints(layout, domain.nodes.length);
        return domain.nodes.map((node, nodeIndex) => {
          const client = clientPoints[nodeIndex];
          const from = upperDirection ? client : layout.server;
          const to = upperDirection ? layout.server : client;
          const path = linePath(from, to);
          const filtered = node.status === '已过滤';
          return <g key={`${domain.id}-${node.id}`} data-link="client" data-direction={upperDirection ? 'uplink' : 'downlink'}>
            <path d={path} className="map-link-backbone map-client-backbone" />
            <path
              d={path}
              className={`map-client-link ${clientActive ? 'active' : ''} ${filtered ? 'filtered' : ''}`}
              style={{ '--line-color': domain.color, animationDelay: `${nodeIndex * 70}ms` } as CSSProperties}
              markerMid={clientActive && !filtered ? `url(#map-arrow-${domain.id})` : undefined}
              markerEnd={clientActive && !filtered ? `url(#map-arrow-${domain.id})` : undefined}
            />
            {filtered && clientActive && <circle className="map-link-blocked" cx={(from.x + to.x) / 2} cy={((from.y + to.y) / 2) * 0.7} r="0.6" />}
          </g>;
        });
      })}
    </svg>
  );
}

export function FederationTopology({ compact = false }: { compact?: boolean }) {
  const phaseIndex = useAppStore((state) => state.phaseIndex);
  const revealTruth = useAppStore((state) => state.revealTruth);
  const selectNode = useAppStore((state) => state.selectNode);
  const phase = phases[phaseIndex];
  const regionData = useMemo(() => domains.map((domain, index) => ({ domain, layout: regionLayouts[index], clientPoints: createClientPoints(regionLayouts[index], domain.nodes.length) })), []);

  return (
    <div className={`topology-canvas map-topology ${compact ? 'compact' : ''}`}>
      <img className="map-background" src={mapImage} alt="中国行政区域演示底图" />
      <div className="map-tone" />
      <FlowLines phaseIndex={phaseIndex} />
      {regionData.map(({ domain, layout, clientPoints }) => (
        <div key={domain.id} className="map-region-layer">
          <div className="map-region-boundary" style={{ left: `${layout.center.x}%`, top: `${layout.center.y}%`, width: `${layout.radius.x * 2}%`, height: `${layout.radius.y * 2}%`, '--domain-color': domain.color } as CSSProperties} />
          <div className="map-region-label" style={pointStyle(layout.label, domain.color)}>
            <span>{layout.code}</span><b>{domain.name}</b><small>{layout.name} · {layout.terrain}</small>
          </div>
          <DomainServerMarker domain={domain} layout={layout} />
          {domain.nodes.map((node, nodeIndex) => <ClientMarker key={node.id} node={node} point={clientPoints[nodeIndex]} color={domain.color} revealTruth={revealTruth} onSelect={() => selectNode(node.id)} />)}
        </div>
      ))}
      <div className="map-central-server" style={pointStyle(centralPoint)}>
        <div className="map-central-radar"><i /><i /><i /></div>
        <div className="map-central-core">⌾</div>
        <div className="map-central-copy"><span>MASTER NODE</span><b>中央主服务器</b><small>CS-00 · {phase}</small></div>
      </div>
      <div className="map-compass"><b>N</b><i /><span>全域态势底图</span></div>
      <div className="simulation-badge"><span /> 地图与三级链路均为前端模拟</div>
      <div className="map-legend">
        <span><i className="legend-central" />主服务器</span>
        <span><i className="legend-domain" />域子服务器</span>
        <span><i className="legend-client" />逻辑节点</span>
        <span><i className="legend-risk" />恶意真值</span>
        <em>{phase}</em>
      </div>
    </div>
  );
}
