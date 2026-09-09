import {
  PauseCircleOutlined,
  PlayCircleOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import { Button, Drawer, Progress, Segmented, Space, Switch, Tag } from 'antd';
import { FederationTopology } from '../components/topology/FederationTopology';
import { domains, nodeRows } from '../mock/data';
import { phases, useAppStore } from '../store/useAppStore';
import type { MilitaryNode } from '../types';
import { officeHomeClassNames } from '../utils/partition';

const phaseDescriptions = [
  ['中央下发', '中央主服务器', '4 / 4 域', '全局模型下发到四个域子服务器'],
  ['域内广播', '域子服务器', '60 / 60 客户端', '四个域分别向域内客户端广播模型'],
  ['节点处理', '逻辑客户端', '37 / 60 处理中', '各客户端执行本地数据处理与训练'],
  ['节点上传', '逻辑客户端', '24 / 60 已上传', '客户端参数沿箭头上传到所属域子服务器'],
  ['域内聚合', '域子服务器', '2 / 4 域', '各域对子节点状态进行前端汇聚投影'],
  ['域级上传', '域子服务器', '2 / 4 已上传', '域级状态沿箭头上传到中央主服务器'],
  ['全域聚合', '中央主服务器', '第 18 轮', '汇聚四域信息并形成新一轮统一状态'],
] as const;

const statusColor: Record<string, string> = {
  本地训练: 'processing', 上传中: 'cyan', 等待聚合: 'default',
  接收中: 'geekblue', 已过滤: 'error', 异常: 'warning', 已完成: 'success',
};

export function OverviewPage({ simulationOnly = false }: { simulationOnly?: boolean }) {
  const {
    phaseIndex,
    setPhaseIndex,
    round,
    running,
    toggleRunning,
    revealTruth,
    toggleTruth,
    selectedNodeId,
    selectNode,
    dataSource,
    connectionState,
  } = useAppStore();
  const selectedNode = nodeRows.find((node) => node.id === selectedNodeId);
  const currentPhase = phaseDescriptions[phaseIndex];

  return (
    <div className="immersive-overview-page">
      <FederationTopology />

      <div className="map-title-overlay">
        <span>GLOBAL FEDERATED INTELLIGENCE</span>
        <h1>全域信息汇聚 · 跨域智能协同</h1>
        <p>四域特征异构 · 域内分布不均 · 60 个逻辑客户端</p>
      </div>

      <div className="map-runtime-toolbar">
        {simulationOnly ? <Tag color="gold">本地动画 · {running ? '播放中' : '已暂停'}</Tag> : <Tag color={connectionState === 'connected' ? 'processing' : 'warning'}>{dataSource === 'backend' ? '训练数据' : '演示数据'} · {connectionState === 'connected' ? '已连接' : '恢复中'}</Tag>}
        <div><small>{simulationOnly ? '模拟轮次' : '全局轮次'}</small><b>{round} / 30</b></div>
        <div><small>当前阶段</small><b>{currentPhase[0]}</b></div>
        <Space size={8}>
          <span className="toolbar-switch-label">恶意真值</span>
          <Switch size="small" checked={revealTruth} onChange={toggleTruth} />
          <Button size="small" icon={running ? <PauseCircleOutlined /> : <PlayCircleOutlined />} onClick={toggleRunning}>{running ? '暂停' : '继续'}</Button>
        </Space>
      </div>

      <section className="training-process-board" aria-label="训练流程">
        <div className="process-current">
          <span>当前执行流程</span>
          <strong>{currentPhase[0]}</strong>
          <b>{currentPhase[2]}</b>
          <small>{currentPhase[3]}</small>
        </div>
        <div className="process-steps">
          {phaseDescriptions.map(([name, owner], index) => (
            <div className="process-step-wrap" key={name}>
              <button
                className={`process-step ${index === phaseIndex ? 'active' : ''} ${index < phaseIndex ? 'completed' : ''}`}
                onClick={() => setPhaseIndex(index)}
                aria-current={index === phaseIndex ? 'step' : undefined}
              >
                <i>{String(index + 1).padStart(2, '0')}</i>
                <span><b>{name}</b><small>{owner}</small></span>
              </button>
              {index < phaseDescriptions.length - 1 && <span className={`process-step-arrow ${index < phaseIndex ? 'completed' : ''}`}>→</span>}
            </div>
          ))}
        </div>
        <div className="process-direction-note"><i /><span>地图中的高亮粗线与箭头表示当前数据流方向</span><em>点击任一阶段可查看对应流程</em></div>
      </section>

      <div className="map-phase-selector">
        <Segmented size="small" options={phases} value={phases[phaseIndex]} onChange={(value) => setPhaseIndex(phases.indexOf(value as typeof phases[number]))} />
      </div>

      <NodeDrawer node={selectedNode} open={Boolean(selectedNode)} onClose={() => selectNode(undefined)} revealTruth={revealTruth} simulationOnly={simulationOnly} />
    </div>
  );
}

function NodeDrawer({ node, open, onClose, revealTruth, simulationOnly }: {
  node?: MilitaryNode;
  open: boolean;
  onClose: () => void;
  revealTruth: boolean;
  simulationOnly?: boolean;
}) {
  const domain = domains.find((item) => item.id === node?.domainId);
  const topClasses = node?.classHistogram
    .map((count, index) => ({ count, index, ratio: node.classProportions[index] }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8) ?? [];
  return (
    <Drawer title={`客户端详情 · ${node?.id ?? ''}`} width={460} open={open} onClose={onClose}>
      {node && <div className="node-inspector">
        <div className="node-identity"><div className="node-avatar" style={{ borderColor: domain?.color }}>{domain?.icon}</div><div><h3>{node.name}</h3><span>{domain?.name} · 单机逻辑客户端</span></div></div>
        <div className="inspector-state"><Tag color={statusColor[node.status]}>{node.status}</Tag><span>延迟 {node.latency} ms</span><span>域键 {node.domainId}</span></div>
        <Progress percent={node.progress} strokeColor={domain?.color} />
        <div className="detail-grid"><div><span>本地样本</span><b>{node.sampleCount.toLocaleString()}</b></div><div><span>类别覆盖</span><b>{node.coveredClassCount} / 65</b></div><div><span>风险分数</span><b className={node.risk > .7 ? 'text-danger' : ''}>{node.risk.toFixed(2)}</b></div><div><span>防御判断</span><b>{node.assessment}</b></div></div>
        {revealTruth && <div className={`truth-panel ${node.malicious ? 'is-malicious' : ''}`}><span>模拟角色真值</span><b>{node.malicious ? '恶意客户端' : '正常客户端'}</b><small>真值仅用于后门实验复盘，不参与系统判断</small></div>}
        <div className="label-bars"><h4>主要类别分布</h4>{topClasses.map(({ count, index, ratio }) => <div key={index}><span>{officeHomeClassNames[index]}</span><Progress percent={ratio * 100} showInfo={false} strokeColor={domain?.color} /><em>{count}</em></div>)}</div>
        <div className="drawer-security-note"><SafetyCertificateOutlined /><span>{simulationOnly ? '节点、风险、标签分布和角色真值均为原前端模拟数据，不代表真实训练结果。' : '完整 65 类数量和比例请在“场景与异构分析”页面查看。'}</span></div>
      </div>}
    </Drawer>
  );
}
