import { memo, useMemo } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Progress } from 'antd';
import { domains } from '../../mock/data';
import { phases, useAppStore } from '../../store/useAppStore';
import { isCentralLinkActive, isClientLinkActive, isUpperDirection } from '../../utils/hierarchyProjection';

type TopologyData = {
  kind: 'central' | 'domain' | 'client';
  label: string;
  subtitle: string;
  status: string;
  progress?: number;
  color?: string;
  malicious?: boolean;
  assessment?: string;
  revealTruth?: boolean;
};

type TopologyNode = Node<TopologyData, 'topology'>;

const TopologyNodeView = memo(({ data, selected }: NodeProps<TopologyNode>) => {
  if (data.kind === 'central') {
    return (
      <div className={`topology-central ${selected ? 'selected' : ''}`}>
        <div className="central-orbit"><span /><span /><span /></div>
        <div className="central-core">⌾</div>
        <strong>{data.label}</strong>
        <small>{data.subtitle}</small>
        <div className="topology-status"><i />{data.status}</div>
      </div>
    );
  }
  if (data.kind === 'domain') {
    return (
      <div className={`topology-domain ${selected ? 'selected' : ''}`} style={{ '--domain-color': data.color } as React.CSSProperties}>
        <div className="domain-node-head"><span className="domain-server-icon">▣</span><b>{data.label}</b></div>
        <small>{data.subtitle}</small>
        <Progress percent={data.progress} size="small" showInfo={false} strokeColor={data.color} />
        <div className="domain-node-foot"><span>{data.status}</span><em>前端模拟</em></div>
      </div>
    );
  }
  const maliciousVisible = data.revealTruth && data.malicious;
  return (
    <div
      className={`topology-client ${selected ? 'selected' : ''} ${data.assessment === '过滤' ? 'filtered' : ''} ${maliciousVisible ? 'malicious' : ''}`}
      style={{ '--domain-color': data.color } as React.CSSProperties}
    >
      <div className="client-ring"><span>{data.label.split('-N')[1]}</span></div>
      <div className="client-copy"><b>{data.label}</b><small>{data.status}</small></div>
      {maliciousVisible && <span className="truth-flag">恶意</span>}
      {data.assessment === '疑似' && <span className="suspect-flag">疑似</span>}
    </div>
  );
});

TopologyNodeView.displayName = 'TopologyNodeView';

const nodeTypes = { topology: TopologyNodeView };

export function FederationTopology({ compact = false }: { compact?: boolean }) {
  const phaseIndex = useAppStore((state) => state.phaseIndex);
  const revealTruth = useAppStore((state) => state.revealTruth);
  const selectNode = useAppStore((state) => state.selectNode);
  const phase = phases[phaseIndex];

  const { nodes, edges } = useMemo(() => {
    const resultNodes: TopologyNode[] = [{
      id: 'central',
      type: 'topology',
      position: { x: 520, y: 12 },
      data: { kind: 'central', label: '中央服务器', subtitle: 'CS-00 · 全域协调', status: phase },
    }];
    const resultEdges: Edge[] = [];
    const domainX = [25, 345, 665, 985];
    const upperDirection = isUpperDirection(phase);
    const centralActive = isCentralLinkActive(phase);
    const clientActive = isClientLinkActive(phase);

    domains.forEach((domain, domainIndex) => {
      const domainId = `domain-${domain.id}`;
      resultNodes.push({
        id: domainId,
        type: 'topology',
        position: { x: domainX[domainIndex], y: 190 },
        data: {
          kind: 'domain',
          label: domain.name,
          subtitle: `${domain.serverId} · ${domain.nodes.length} 个节点`,
          status: domain.status,
          progress: 62 + domainIndex * 7,
          color: domain.color,
        },
      });
      resultEdges.push({
        id: `central-${domainId}`,
        source: upperDirection ? domainId : 'central',
        target: upperDirection ? 'central' : domainId,
        animated: centralActive,
        markerEnd: { type: MarkerType.ArrowClosed, color: centralActive ? '#44d8ff' : '#36546c' },
        style: { stroke: centralActive ? '#44d8ff' : '#284258', strokeWidth: centralActive ? 2 : 1 },
      });

      if (!compact) {
        domain.nodes.forEach((node, nodeIndex) => {
          const nodeId = `client-${node.id}`;
          resultNodes.push({
            id: nodeId,
            type: 'topology',
            position: { x: domainX[domainIndex] - 6 + nodeIndex * 55, y: 365 + (nodeIndex % 2) * 72 },
            data: {
              kind: 'client', label: node.id, subtitle: node.name, status: node.status,
              color: domain.color, malicious: node.malicious,
              assessment: node.assessment, revealTruth,
            },
          });
          resultEdges.push({
            id: `${domainId}-${nodeId}`,
            source: upperDirection ? nodeId : domainId,
            target: upperDirection ? domainId : nodeId,
            animated: clientActive && (nodeIndex % 2 === phaseIndex % 2 || phase === '域内聚合'),
            markerEnd: { type: MarkerType.ArrowClosed, color: clientActive ? domain.color : '#30465b' },
            style: { stroke: clientActive ? domain.color : '#253d50', strokeWidth: clientActive ? 1.5 : 1 },
          });
        });
      }
    });
    return { nodes: resultNodes, edges: resultEdges };
  }, [compact, phase, phaseIndex, revealTruth]);

  return (
    <div className={`topology-canvas ${compact ? 'compact' : ''}`}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => {
          if (node.id.startsWith('client-')) selectNode(node.id.replace('client-', ''));
        }}
        fitView
        fitViewOptions={{ padding: compact ? 0.15 : 0.08 }}
        minZoom={0.55}
        maxZoom={1.5}
        nodesDraggable={false}
        nodesConnectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#26465e" />
        <Controls showInteractive={false} />
        {!compact && <MiniMap pannable zoomable nodeColor="#23455e" maskColor="rgba(3, 10, 20, .72)" />}
      </ReactFlow>
      <div className="simulation-badge"><span /> 三级链路展示模拟</div>
    </div>
  );
}
