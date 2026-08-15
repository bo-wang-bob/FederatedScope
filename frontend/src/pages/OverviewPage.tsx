import {
  AlertOutlined,
  ApartmentOutlined,
  CloudServerOutlined,
  NodeIndexOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import { Button, Drawer, Progress, Segmented, Space, Switch, Table, Tag } from 'antd';
import { useMemo } from 'react';
import { Chart, chartGrid, chartText, Panel } from '../components/ChartPanel';
import { MetricCard } from '../components/MetricCard';
import { PageHeader } from '../components/PageHeader';
import { FederationTopology } from '../components/topology/FederationTopology';
import { domains, nodeRows, roundMetrics } from '../mock/data';
import { phases, useAppStore } from '../store/useAppStore';
import type { MilitaryNode } from '../types';

const statusColor: Record<string, string> = {
  '本地训练': 'processing', '上传中': 'cyan', '等待聚合': 'default',
  '接收中': 'geekblue', '已过滤': 'error', '异常': 'warning', '已完成': 'success',
};

export function OverviewPage() {
  const { phaseIndex, setPhaseIndex, revealTruth, toggleTruth, selectedNodeId, selectNode } = useAppStore();
  const selectedNode = nodeRows.find((node) => node.id === selectedNodeId);

  const trendOption = useMemo(() => ({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { data: ['全局准确率', '最差域准确率', '攻击成功率'], textStyle: { color: chartText }, top: 0 },
    grid: { left: 42, right: 18, top: 40, bottom: 28 },
    xAxis: { type: 'category', data: roundMetrics.map((item) => item.round), axisLabel: { color: chartText }, axisLine: { lineStyle: { color: chartGrid } } },
    yAxis: { type: 'value', min: 0, max: 100, axisLabel: { color: chartText, formatter: '{value}%' }, splitLine: { lineStyle: { color: chartGrid } } },
    series: [
      { name: '全局准确率', type: 'line', smooth: true, symbol: 'none', data: roundMetrics.map((item) => item.accuracy), lineStyle: { color: '#44d8ff', width: 3 }, areaStyle: { color: 'rgba(68,216,255,.08)' } },
      { name: '最差域准确率', type: 'line', smooth: true, symbol: 'none', data: roundMetrics.map((item) => item.worstDomain), lineStyle: { color: '#29e3ae', width: 2 } },
      { name: '攻击成功率', type: 'line', smooth: true, symbol: 'none', data: roundMetrics.map((item) => item.attackSuccess), lineStyle: { color: '#ffb84c', width: 2, type: 'dashed' } },
    ],
  }), []);

  const columns = [
    { title: '节点编号', dataIndex: 'id', key: 'id', render: (value: string) => <button className="link-button" onClick={() => selectNode(value)}>{value}</button> },
    { title: '所属域', dataIndex: 'domainId', key: 'domain', render: (value: string) => domains.find((domain) => domain.id === value)?.shortName },
    { title: '状态', dataIndex: 'status', key: 'status', render: (value: string) => <Tag color={statusColor[value]}>{value}</Tag> },
    { title: '通信', dataIndex: 'progress', key: 'progress', render: (value: number) => <Progress percent={value} size="small" /> },
    { title: '样本量', dataIndex: 'sampleCount', key: 'samples', render: (value: number) => value.toLocaleString() },
    { title: '风险', dataIndex: 'risk', key: 'risk', render: (value: number) => <span className={value > .7 ? 'text-danger' : value > .4 ? 'text-warning' : 'text-success'}>{value.toFixed(2)}</span> },
    { title: '防御判断', dataIndex: 'assessment', key: 'assessment', render: (value: string) => <Tag color={value === '过滤' ? 'error' : value === '疑似' ? 'warning' : 'success'}>{value}</Tag> },
    ...(revealTruth ? [{ title: '模拟真值', dataIndex: 'malicious', key: 'truth', render: (value: boolean) => value ? <Tag color="error">恶意</Tag> : <span className="muted">正常</span> }] : []),
  ];

  return (
    <div className="page overview-page">
      <PageHeader
        eyebrow="GLOBAL FEDERATED SITUATION"
        title="全域信息汇聚 · 跨域智能协同"
        description="在原始数据不出域的前提下，统一呈现多域训练状态、异构态势、隐私风险与攻击防御结果。"
        actions={<Space><span className="switch-label">显示模拟真值</span><Switch checked={revealTruth} onChange={toggleTruth} /><Button type="primary">新建实验</Button></Space>}
      />

      <div className="intelligence-flow">
        <div className="flow-stage active"><i>01</i><strong>全域感知</strong><span>4 域 · 20 节点</span></div>
        <div className="flow-arrow"><span>•••</span>›</div>
        <div className="flow-stage"><i>02</i><strong>分层汇聚</strong><span>域内 → 全域</span></div>
        <div className="flow-arrow"><span>•••</span>›</div>
        <div className="flow-stage"><i>03</i><strong>统一认知</strong><span>模型 · 风险 · 态势</span></div>
        <div className="flow-arrow"><span>•••</span>›</div>
        <div className="flow-stage"><i>04</i><strong>智能协同</strong><span>模型逐级下发</span></div>
        <div className="cognition-summary"><small>统一认知摘要</small><b>全域覆盖完整，当前异构风险中等</b><span>检测到 2 个高风险节点更新，已进入防御流程</span></div>
      </div>

      <div className="metrics-grid six">
        <MetricCard label="已接入军事域" value="4" suffix=" / 4" delta="全域覆盖率 100%" icon={<ApartmentOutlined />} />
        <MetricCard label="在线逻辑节点" value="19" suffix=" / 20" delta="1 个节点降级" icon={<NodeIndexOutlined />} tone="green" />
        <MetricCard label="全局模型精度" value="85.7" suffix="%" delta="较基线 +12.4%" icon={<CloudServerOutlined />} tone="violet" />
        <MetricCard label="域间异构指数" value="0.69" delta="处理后下降 31%" icon={<span>≋</span>} tone="amber" />
        <MetricCard label="隐私风险" value="低" delta="本地保护已启用" icon={<SafetyCertificateOutlined />} tone="green" />
        <MetricCard label="风险节点" value="2" suffix=" 个" delta="1 过滤 · 1 复核" icon={<AlertOutlined />} tone="red" />
      </div>

      <div className="overview-main-grid">
        <Panel
          title="全域地理部署与三级协同"
          subtitle="中心主服务器 → 地区域子服务器 → 区域内逻辑节点"
          extra={<Segmented size="small" options={phases} value={phases[phaseIndex]} onChange={(value) => setPhaseIndex(phases.indexOf(value as typeof phases[number]))} />}
          className="topology-panel"
        >
          <FederationTopology />
        </Panel>
        <div className="overview-side">
          <Panel title="域级汇聚状态" subtitle="各域子服务器的模拟进度">
            <div className="domain-progress-list">
              {domains.map((domain, index) => (
                <div className="domain-progress" key={domain.id}>
                  <div className="domain-progress-head"><span><i style={{ background: domain.color }} />{domain.name}</span><em>{domain.serverId}</em></div>
                  <Progress percent={62 + index * 7} strokeColor={domain.color} size="small" />
                  <div className="domain-progress-meta"><span>{domain.nodes.filter((node) => node.status !== '已过滤').length} / {domain.nodes.length} 节点有效</span><span>{domain.status}</span></div>
                </div>
              ))}
            </div>
          </Panel>
          <Panel title="两层异构态势" subtitle="域间特征差异 / 域内分布不均">
            <div className="hetero-rings">
              <div className="ring-gauge"><Progress type="circle" percent={69} size={84} strokeColor="#44d8ff" format={() => '0.69'} /><span>域间特征偏移</span></div>
              <div className="ring-gauge"><Progress type="circle" percent={76} size={84} strokeColor="#ffbd52" format={() => '0.76'} /><span>域内分布不均</span></div>
            </div>
            <div className="mini-insight"><b>主要差异来源</b><span>指挥决策域与态势感知域的特征距离最高</span></div>
          </Panel>
        </div>
      </div>

      <div className="bottom-grid">
        <Panel title="全局性能与安全趋势" subtitle="最近 30 轮训练变化"><Chart option={trendOption} height={285} /></Panel>
        <Panel title="节点状态明细" subtitle="真值与系统判断分栏呈现" className="node-table-panel">
          <Table<MilitaryNode> rowKey="id" columns={columns} dataSource={nodeRows} pagination={{ pageSize: 5, size: 'small' }} size="small" />
        </Panel>
      </div>

      <NodeDrawer node={selectedNode} open={Boolean(selectedNode)} onClose={() => selectNode(undefined)} revealTruth={revealTruth} />
    </div>
  );
}

function NodeDrawer({ node, open, onClose, revealTruth }: { node?: MilitaryNode; open: boolean; onClose: () => void; revealTruth: boolean }) {
  const domain = domains.find((item) => item.id === node?.domainId);
  return (
    <Drawer title={`节点详情 · ${node?.id ?? ''}`} width={440} open={open} onClose={onClose}>
      {node && <div className="node-inspector">
        <div className="node-identity"><div className="node-avatar" style={{ borderColor: domain?.color }}>{domain?.icon}</div><div><h3>{node.name}</h3><span>{domain?.name} · 逻辑模拟节点</span></div></div>
        <div className="inspector-state"><Tag color={statusColor[node.status]}>{node.status}</Tag><span>延迟 {node.latency} ms</span><span>第 18 轮</span></div>
        <Progress percent={node.progress} strokeColor={domain?.color} />
        <div className="detail-grid"><div><span>本地样本</span><b>{node.sampleCount.toLocaleString()}</b></div><div><span>数据质量</span><b>{node.quality} / 100</b></div><div><span>风险分数</span><b className={node.risk > .7 ? 'text-danger' : ''}>{node.risk.toFixed(2)}</b></div><div><span>防御判断</span><b>{node.assessment}</b></div></div>
        {revealTruth && <div className={`truth-panel ${node.malicious ? 'is-malicious' : ''}`}><span>模拟角色真值</span><b>{node.malicious ? '恶意节点' : '正常节点'}</b><small>真值仅用于演示复盘，不参与系统判断</small></div>}
        <div className="label-bars"><h4>标签分布</h4>{node.labels.map((value, index) => <div key={index}><span>类别 {index + 1}</span><Progress percent={value} showInfo={false} strokeColor={domain?.color} /><em>{value}</em></div>)}</div>
      </div>}
    </Drawer>
  );
}
