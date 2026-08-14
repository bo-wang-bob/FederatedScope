import { CopyOutlined, PlusOutlined, SaveOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { Button, InputNumber, Progress, Select, Slider, Space, Tag } from 'antd';
import { useState } from 'react';
import { Panel } from '../components/ChartPanel';
import { PageHeader } from '../components/PageHeader';
import { domains } from '../mock/data';

export function ScenarioPage() {
  const [selected, setSelected] = useState(0);
  const [featureShift, setFeatureShift] = useState(72);
  const [distributionShift, setDistributionShift] = useState(76);
  const domain = domains[selected];
  return (
    <div className="page">
      <PageHeader eyebrow="SCENARIO ORCHESTRATION" title="跨域场景编排" description="配置域间特征差异、域内数据分布与逻辑节点状态，快速生成可复现的单机模拟场景。" actions={<Space><Button icon={<CopyOutlined />}>复制场景</Button><Button type="primary" icon={<SaveOutlined />}>保存场景</Button></Space>} />
      <div className="scenario-layout">
        <Panel title="军事域" subtitle="每个域自动生成一个子服务器" className="domain-list-panel">
          <div className="domain-selector-list">
            {domains.map((item, index) => <button key={item.id} className={selected === index ? 'active' : ''} onClick={() => setSelected(index)} style={{ '--domain-color': item.color } as React.CSSProperties}>
              <i>{item.icon}</i><span><b>{item.name}</b><small>{item.serverId} · {item.nodes.length} 个节点</small></span><em>{item.modality}</em>
            </button>)}
            <Button block type="dashed" icon={<PlusOutlined />}>增加军事域</Button>
          </div>
        </Panel>
        <div className="scenario-center">
          <Panel title={`${domain.name} · 域内结构`} subtitle="拖动参数即时更新场景预览" extra={<Tag color="cyan">{domain.serverId}</Tag>}>
            <div className="domain-preview" style={{ '--domain-color': domain.color } as React.CSSProperties}>
              <div className="preview-server"><span>▣</span><b>{domain.serverId}</b><small>域子服务器 · 前端模拟</small></div>
              <div className="preview-links" />
              <div className="preview-clients">{domain.nodes.map((node, index) => <div key={node.id} className="preview-client"><i>{index + 1}</i><b>{node.id}</b><span>{node.sampleCount.toLocaleString()} 样本</span><Progress percent={node.labels[index % 5]} showInfo={false} strokeColor={domain.color} /></div>)}</div>
            </div>
          </Panel>
          <Panel title="场景预设" subtitle="使用固定种子保证演示可复现">
            <div className="preset-grid">{[
              ['轻度异构', '特征差异小，域内分布相对均衡', '0.25'],
              ['中度异构', '明显特征偏移，样本量呈长尾', '0.52'],
              ['重度异构', '多模态差异，节点缺类严重', '0.81'],
              ['攻防演示', '包含模拟恶意节点与两阶段防御', '安全'],
            ].map(([title, copy, tag]) => <button key={title}><ThunderboltOutlined /><b>{title}</b><span>{copy}</span><em>{tag}</em></button>)}</div>
          </Panel>
        </div>
        <Panel title="属性配置" subtitle={domain.name} className="property-panel">
          <div className="form-section"><h4>域级特征</h4><label>数据模态<Select value={domain.modality} options={[{ value: domain.modality, label: domain.modality }]} /></label><label>原始特征维度<InputNumber value={domain.featureDimension} min={64} max={4096} /></label><label>统一表示维度<InputNumber value={domain.unifiedDimension} min={32} max={1024} /></label></div>
          <div className="form-section"><h4>异构强度</h4><label>域间特征偏移 <b>{(featureShift / 100).toFixed(2)}</b><Slider value={featureShift} onChange={setFeatureShift} /></label><label>域内分布不均 <b>{(distributionShift / 100).toFixed(2)}</b><Slider value={distributionShift} onChange={setDistributionShift} /></label></div>
          <div className="form-section"><h4>节点模拟</h4><label>节点数量<InputNumber value={5} min={2} max={60} /></label><label>编号前缀<Select value={`D0${selected + 1}-N`} options={[{ value: `D0${selected + 1}-N`, label: `D0${selected + 1}-N` }]} /></label><label>随机种子<InputNumber value={20260815} /></label></div>
        </Panel>
      </div>
    </div>
  );
}
