import {
  CheckCircleFilled,
  CloudUploadOutlined,
  ExperimentOutlined,
  LockOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import { Alert, Button, Checkbox, InputNumber, Radio, Select, Slider, Space, Steps, Switch, Tag } from 'antd';
import { useState } from 'react';
import { Panel } from '../components/ChartPanel';
import { PageHeader } from '../components/PageHeader';
import { useAppStore } from '../store/useAppStore';
import type { ExperimentMode } from '../types';

const modeCards: { key: ExperimentMode; icon: React.ReactNode; title: string; description: string; tag: string }[] = [
  { key: 'baseline', icon: <ExperimentOutlined />, title: '异构协同基线', description: '只运行两阶段异构解决流程，不启用安全实验。', tag: '普通训练' },
  { key: 'privacy', icon: <LockOutlined />, title: '隐私风险与保护', description: '独立评估隐私攻击效果，支持本地自适应保护对照。', tag: '与后门互斥' },
  { key: 'backdoor', icon: <SafetyCertificateOutlined />, title: '后门攻防模拟', description: '配置模拟恶意节点，对比无防御与两阶段防御。', tag: '与隐私互斥' },
];

export function ExperimentPage() {
  const { mode, setMode } = useAppStore();
  const [step, setStep] = useState(1);
  const [protection, setProtection] = useState(true);
  const [featureDefense, setFeatureDefense] = useState(true);
  const [trainingDefense, setTrainingDefense] = useState(true);
  return <div className="page">
    <PageHeader eyebrow="EXPERIMENT CONFIGURATION" title="实验配置与模式编排" description="隐私实验与后门实验分别运行；所有客户端与服务端角色均由单机进程模拟。" actions={<Tag color="cyan">配置草稿自动保存</Tag>} />
    <Steps current={step} className="experiment-steps" items={['选择场景', '实验模式', '异构方案', '安全配置', '训练参数', '确认启动'].map((title) => ({ title }))} onChange={setStep} />
    <div className="experiment-layout">
      <div className="experiment-main">
        <Panel title="选择实验模式" subtitle="三种模式的运行记录相互独立">
          <div className="mode-card-grid">{modeCards.map((item) => <button className={mode === item.key ? 'active' : ''} key={item.key} onClick={() => setMode(item.key)}><div className="mode-icon">{item.icon}</div><Tag>{item.tag}</Tag><h3>{item.title}</h3><p>{item.description}</p>{mode === item.key && <CheckCircleFilled className="mode-check" />}</button>)}</div>
        </Panel>

        <Panel title="异构解决方案" subtitle="所有实验均基于两阶段跨域协同底座">
          <div className="two-stage-config"><div><span className="stage-number">01</span><div><b>特征统计与本地扩充</b><p>节点交换非敏感特征摘要，在本地扩充缺失类别表示。</p></div><Switch defaultChecked /></div><div className="stage-connector" /><div><span className="stage-number">02</span><div><b>正常联邦训练</b><p>基于统一表示空间执行多轮本地训练与服务端聚合。</p></div><Switch defaultChecked disabled /></div></div>
        </Panel>

        {mode === 'privacy' && <Panel title="隐私风险与保护设置" subtitle="仅作用于隐私实验，不与后门攻击同时执行">
          <Alert type="info" showIcon message="隐私攻击分别评估" description="成员关系推断、属性推断和数据重建将生成三份独立结果；保护前后由两个独立实验组成对照。" />
          <div className="config-grid">
            <label><span>评估类型</span><Checkbox.Group defaultValue={['membership', 'property', 'reconstruction']} options={[{ label: '成员关系推断', value: 'membership' }, { label: '属性推断', value: 'property' }, { label: '数据重建', value: 'reconstruction' }]} /></label>
            <label><span>启用本地隐私保护</span><Switch checked={protection} onChange={setProtection} /></label>
            <label><span>初始裁剪阈值</span><InputNumber defaultValue={1.0} step={.1} /></label>
            <label><span>自适应目标分位数</span><Slider defaultValue={80} tooltip={{ formatter: (value) => `${value}%` }} /></label>
            <label><span>噪声强度</span><InputNumber defaultValue={0.65} step={.05} /></label>
            <label><span>保护预算</span><InputNumber defaultValue={4.0} step={.5} /></label>
          </div>
        </Panel>}

        {mode === 'backdoor' && <Panel title="后门攻击与防御设置" subtitle="防御判断与模拟恶意真值相互独立">
          <Alert type="warning" showIcon message="后门模式已自动关闭隐私攻击与隐私保护" />
          <div className="config-grid">
            <label><span>攻击类别</span><Select defaultValue="后门注入" options={['后门注入', '标签污染', '模型更新污染'].map((value) => ({ value }))} /></label>
            <label><span>模拟恶意节点</span><Select mode="multiple" defaultValue={['D01-N004', 'D03-N002']} options={['D01-N004', 'D03-N002', 'D04-N005'].map((value) => ({ value }))} /></label>
            <label><span>注入起始轮次</span><InputNumber defaultValue={4} min={1} /></label>
            <label><span>攻击强度</span><Slider defaultValue={65} /></label>
          </div>
          <div className="defense-stage-grid"><label className={featureDefense ? 'active' : ''}><Switch checked={featureDefense} onChange={setFeatureDefense} /><span><b>第一阶段防御</b><small>对特征统计摘要执行异常识别与过滤</small></span></label><label className={trainingDefense ? 'active' : ''}><Switch checked={trainingDefense} onChange={setTrainingDefense} /><span><b>第二阶段防御</b><small>对训练更新执行多维风险判断</small></span></label></div>
        </Panel>}

        {mode === 'baseline' && <Panel title="训练参数" subtitle="单机模拟 20 个逻辑客户端"><div className="config-grid"><label><span>全局轮次</span><InputNumber defaultValue={30} min={1} /></label><label><span>每轮采样节点</span><InputNumber defaultValue={16} min={2} max={20} /></label><label><span>本地训练轮次</span><InputNumber defaultValue={2} min={1} /></label><label><span>聚合策略</span><Select value="样本量加权聚合" options={[{ value: '样本量加权聚合' }]} /></label></div></Panel>}
      </div>
      <aside className="experiment-summary">
        <Panel title="运行摘要" subtitle="提交前校验">
          <div className="summary-scene"><span>跨域联合认知演示</span><Tag color="cyan">4 域 / 20 节点</Tag></div>
          <div className="summary-list"><div><span>实验模式</span><b>{modeCards.find((item) => item.key === mode)?.title}</b></div><div><span>异构解决</span><b>两阶段启用</b></div><div><span>隐私保护</span><b>{mode === 'privacy' && protection ? '启用' : '关闭'}</b></div><div><span>后门攻击</span><b>{mode === 'backdoor' ? '启用' : '关闭'}</b></div><div><span>攻击防御</span><b>{mode === 'backdoor' && (featureDefense || trainingDefense) ? '启用' : '关闭'}</b></div><div><span>运行方式</span><b>单机逻辑模拟</b></div></div>
          <div className="validation-pass"><CheckCircleFilled />模式互斥校验通过</div>
          <Button type="primary" size="large" block icon={<CloudUploadOutlined />}>创建并启动实验</Button>
          <p className="summary-note">三级分层聚合只作为前端态势模拟，不改变后端训练逻辑。</p>
        </Panel>
      </aside>
    </div>
  </div>;
}
