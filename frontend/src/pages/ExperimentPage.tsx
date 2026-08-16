import { CheckCircleFilled, DownloadOutlined, ExperimentOutlined, LockOutlined, RocketOutlined, SafetyCertificateOutlined } from '@ant-design/icons';
import { Alert, Button, Input, InputNumber, Select, Slider, Space, Switch, Tag, message } from 'antd';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { ApiError, experimentApi } from '../api/experimentApi';
import { Panel } from '../components/ChartPanel';
import { PageHeader } from '../components/PageHeader';
import { exportExperimentConfig } from '../features/experiment/configExport';
import { nodeRows } from '../mock/data';
import { useAppStore } from '../store/useAppStore';
import type { BackdoorConfig, Capabilities, ExperimentConfig, ExperimentMethod, ExperimentMode, PreflightResult, PrivacyConfig } from '../types';

const modeCards: Array<{ key: ExperimentMode; icon: ReactNode; title: string; description: string }> = [
  { key: 'heterogeneity', icon: <ExperimentOutlined />, title: '异构协同实验', description: '运行 FedAvg、FedProx 或异构解决方案。' },
  { key: 'privacy', icon: <LockOutlined />, title: '隐私保护实验', description: '分别启动有保护或无保护的隐私攻击评估。' },
  { key: 'backdoor', icon: <SafetyCertificateOutlined />, title: '后门攻防实验', description: '分别启动有防御或无防御的后门攻击实验。' },
];

const methodOptions = [
  { value: 'fedavg', label: 'FedAvg' },
  { value: 'fedprox', label: 'FedProx' },
  { value: 'heterogeneous_solution', label: '异构解决方案' },
];

function newIdempotencyKey() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function ExperimentPage() {
  const navigate = useNavigate();
  const { mode, setMode, scenarioId, scenarioVersion, setActiveExperiment } = useAppStore();
  const [name, setName] = useState(`跨域实验-${new Date().toISOString().slice(0, 10)}`);
  const [method, setMethod] = useState<ExperimentMethod>('heterogeneous_solution');
  const [rounds, setRounds] = useState(30);
  const [localEpochs, setLocalEpochs] = useState(1);
  const [participationRate, setParticipationRate] = useState(100);
  const [batchSize, setBatchSize] = useState(8);
  const [learningRate, setLearningRate] = useState(0.001);
  const [seed, setSeed] = useState(20260816);
  const [device, setDevice] = useState<'cpu' | 'cuda'>('cpu');
  const [fedproxMu, setFedproxMu] = useState(0.01);
  const [expansionTarget, setExpansionTarget] = useState(50);
  const [featureBatchSize, setFeatureBatchSize] = useState(64);
  const [privacyAttack, setPrivacyAttack] = useState<PrivacyConfig['attack']>('membership');
  const [privacyDefense, setPrivacyDefense] = useState(true);
  const [initialClip, setInitialClip] = useState(1);
  const [targetQuantile, setTargetQuantile] = useState(0.7);
  const [noiseMultiplier, setNoiseMultiplier] = useState(0.05);
  const [epsilon, setEpsilon] = useState(6);
  const [backdoorAttack, setBackdoorAttack] = useState<BackdoorConfig['attack']>('trigger_injection');
  const [backdoorDefense, setBackdoorDefense] = useState(true);
  const [maliciousRatio, setMaliciousRatio] = useState(5);
  const [maliciousClients, setMaliciousClients] = useState<string[]>([]);
  const [startRound, setStartRound] = useState(1);
  const [poisonRatio, setPoisonRatio] = useState(20);
  const [targetLabel, setTargetLabel] = useState(0);
  const [featureDefense, setFeatureDefense] = useState(true);
  const [trainingDefense, setTrainingDefense] = useState(true);
  const [capabilities, setCapabilities] = useState<Capabilities>();
  const [capabilityError, setCapabilityError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [preflightResult, setPreflightResult] = useState<PreflightResult>();
  const [preflighting, setPreflighting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    experimentApi.capabilities(controller.signal)
      .then((result) => {
        setCapabilities(result);
        setCapabilityError('');
      })
      .catch((error: Error) => setCapabilityError(error.message));
    return () => controller.abort();
  }, []);

  const config = useMemo<ExperimentConfig>(() => {
    const common = {
      method, rounds, localEpochs, participationRate: participationRate / 100,
      batchSize, learningRate, seed, device,
      ...(method === 'fedprox' ? { fedproxMu } : {}),
    };
    const base = {
      schemaVersion: '1.0' as const,
      idempotencyKey: newIdempotencyKey(),
      name: name.trim(), scenarioId: scenarioId || '', common,
    };
    if (mode === 'privacy') return {
      ...base, type: 'privacy', heterogeneity: null,
      privacy: {
        attack: privacyAttack, defenseEnabled: privacyDefense,
        ...(privacyDefense ? { initialClip, targetQuantile, noiseMultiplier, epsilon } : {}),
      },
      backdoor: null,
    };
    if (mode === 'backdoor') return {
      ...base, type: 'backdoor', heterogeneity: null, privacy: null,
      backdoor: {
        attack: backdoorAttack, defenseEnabled: backdoorDefense,
        maliciousRatio: maliciousRatio / 100, maliciousClients, startRound,
        poisonRatio: poisonRatio / 100, targetLabel,
        featureStageDefense: backdoorDefense && featureDefense,
        trainingStageDefense: backdoorDefense && trainingDefense,
      },
    };
    return {
      ...base, type: 'heterogeneity',
      heterogeneity: { expansionTarget, featureBatchSize },
      privacy: null, backdoor: null,
    };
  }, [backdoorAttack, backdoorDefense, batchSize, device, epsilon, expansionTarget, featureBatchSize, featureDefense, fedproxMu, initialClip, learningRate, localEpochs, maliciousClients, maliciousRatio, method, mode, name, noiseMultiplier, participationRate, poisonRatio, privacyAttack, privacyDefense, rounds, scenarioId, seed, startRound, targetLabel, targetQuantile, trainingDefense]);

  useEffect(() => setPreflightResult(undefined), [config]);

  const runPreflight = async () => {
    if (!scenarioId) {
      message.error('请先在场景与异构分析页应用基础异构环境');
      return undefined;
    }
    setPreflighting(true);
    setFieldErrors({});
    try {
      const result = await experimentApi.preflight(config);
      setPreflightResult(result);
      message.success('运行预检通过');
      return result;
    } catch (error) {
      if (error instanceof ApiError) setFieldErrors(error.fieldErrors);
      message.error(error instanceof Error ? error.message : '运行预检失败');
      return undefined;
    } finally {
      setPreflighting(false);
    }
  };

  const startExperiment = async () => {
    if (!scenarioId) {
      message.error('请先在场景与异构分析页应用基础异构环境');
      return;
    }
    setSubmitting(true);
    setFieldErrors({});
    try {
      const request = { ...config, idempotencyKey: newIdempotencyKey() } as ExperimentConfig;
      const checked = await experimentApi.preflight(request);
      setPreflightResult(checked);
      const record = await experimentApi.create(request);
      setActiveExperiment(record.experimentId);
      navigate(`/experiments/${encodeURIComponent(record.experimentId)}/live`);
    } catch (error) {
      if (error instanceof ApiError) setFieldErrors(error.fieldErrors);
      message.error(error instanceof Error ? error.message : '实验启动失败');
    } finally {
      setSubmitting(false);
    }
  };

  return <div className="page experiment-page">
    <PageHeader eyebrow="EXPERIMENT CONFIGURATION" title="实验配置" description="配置实验参数、导出配置文件并启动后端单机任务。" actions={<Space><Tag color={capabilities?.runner.ready ? 'success' : 'warning'}>{capabilities?.runner.ready ? '运行环境就绪' : '运行环境待检查'}</Tag><Button icon={<DownloadOutlined />} onClick={() => exportExperimentConfig(config)}>导出配置</Button></Space>} />

    {capabilityError && <Alert type="error" showIcon message="后端控制服务不可用" description={capabilityError} />}
    {capabilities && !capabilities.runner.ready && <Alert type="warning" showIcon message="训练资源尚未就绪" description="请在后端配置 OfficeHome 数据目录后再启动实验；配置编辑与导出仍可使用。" />}
    {Object.keys(fieldErrors).length > 0 && <Alert type="error" showIcon message="配置校验未通过" description={Object.entries(fieldErrors).map(([field, text]) => <div key={field}><code>{field}</code>：{text}</div>)} />}

    <div className="experiment-layout">
      <div className="experiment-main">
        <Panel title="实验类型" subtitle="每次任务只运行一种实验类型">
          <div className="mode-card-grid">{modeCards.map((item) => <button className={mode === item.key ? 'active' : ''} key={item.key} onClick={() => setMode(item.key)}><div className="mode-icon">{item.icon}</div><h3>{item.title}</h3><p>{item.description}</p>{mode === item.key && <CheckCircleFilled className="mode-check" />}</button>)}</div>
          <div className="experiment-presets"><span>快速预设</span><Button size="small" onClick={() => { setMode('heterogeneity'); setMethod('fedavg'); }}>FedAvg 基线</Button><Button size="small" onClick={() => { setMode('heterogeneity'); setMethod('fedprox'); }}>FedProx 基线</Button><Button size="small" onClick={() => { setMode('heterogeneity'); setMethod('heterogeneous_solution'); }}>异构协同</Button><Button size="small" onClick={() => { setMode('privacy'); setPrivacyDefense(false); }}>隐私无保护</Button><Button size="small" onClick={() => { setMode('privacy'); setPrivacyDefense(true); }}>隐私有保护</Button><Button size="small" onClick={() => { setMode('backdoor'); setBackdoorDefense(false); }}>后门无防御</Button><Button size="small" onClick={() => { setMode('backdoor'); setBackdoorDefense(true); setFeatureDefense(true); setTrainingDefense(true); }}>后门双阶段防御</Button></div>
        </Panel>

        <Panel title="基础参数" subtitle="适用于当前实验任务">
          <div className="config-grid">
            <label><span>实验名称</span><Input value={name} maxLength={80} onChange={(event) => setName(event.target.value)} status={fieldErrors.name ? 'error' : undefined} /></label>
            <label><span>运行方案</span><Select value={method} options={methodOptions} onChange={setMethod} /></label>
            <label><span>全局轮次</span><InputNumber value={rounds} min={1} max={500} onChange={(value) => setRounds(value ?? 30)} /></label>
            <label><span>本地训练轮次</span><InputNumber value={localEpochs} min={1} max={50} onChange={(value) => setLocalEpochs(value ?? 1)} /></label>
            <label><span>客户端参与比例</span><Slider value={participationRate} min={1} max={100} onChange={setParticipationRate} tooltip={{ formatter: (value) => `${value}%` }} /></label>
            <label><span>批大小</span><InputNumber value={batchSize} min={1} max={1024} onChange={(value) => setBatchSize(value ?? 8)} /></label>
            <label><span>学习率</span><InputNumber value={learningRate} min={0.0000001} max={1} step={0.0001} onChange={(value) => setLearningRate(value ?? 0.001)} /></label>
            <label><span>随机种子</span><InputNumber value={seed} min={0} precision={0} onChange={(value) => setSeed(value ?? 20260816)} /></label>
            <label><span>运行设备</span><Select value={device} options={(capabilities?.devices || ['cpu', 'cuda']).map((value) => ({ value, label: value.toUpperCase() }))} onChange={setDevice} /></label>
            {method === 'fedprox' && <label><span>近端项系数 μ</span><InputNumber value={fedproxMu} min={0} step={0.01} onChange={(value) => setFedproxMu(value ?? 0.01)} /></label>}
          </div>
        </Panel>

        {mode === 'heterogeneity' && method === 'heterogeneous_solution' && <Panel title="异构解决方案参数" subtitle="第一阶段交换特征摘要，第二阶段执行正常联邦训练"><div className="config-grid"><label><span>特征提取批大小</span><InputNumber value={featureBatchSize} min={1} max={1024} onChange={(value) => setFeatureBatchSize(value ?? 64)} /></label><label><span>每类本地扩充目标</span><InputNumber value={expansionTarget} min={0} max={10000} onChange={(value) => setExpansionTarget(value ?? 50)} /></label></div></Panel>}

          {mode === 'privacy' && <Panel title="隐私攻击与保护" subtitle="选择一种攻击，并决定是否启用本地自适应保护"><div className="config-grid"><label><span>攻击类型</span><Select value={privacyAttack} onChange={setPrivacyAttack} options={[{ value: 'membership', label: '成员关系推断' }, { value: 'property', label: '属性推断' }, { value: 'reconstruction', label: '数据重建' }]} /></label><label><span>启用本地自适应保护</span><Switch checked={privacyDefense} onChange={setPrivacyDefense} /></label>{privacyDefense && <><label><span>初始裁剪阈值</span><InputNumber value={initialClip} min={0.001} step={0.1} onChange={(value) => setInitialClip(value ?? 1)} /></label><label><span>目标分位数</span><Slider value={targetQuantile * 100} min={1} max={99} onChange={(value) => setTargetQuantile(value / 100)} tooltip={{ formatter: (value) => `${value}%` }} /></label><label><span>噪声强度</span><InputNumber value={noiseMultiplier} min={0} step={0.01} onChange={(value) => setNoiseMultiplier(value ?? 0.05)} /></label><label><span>经验预算参数（非严格保证）</span><InputNumber value={epsilon} min={0.01} step={0.5} onChange={(value) => setEpsilon(value ?? 6)} /></label></>}</div></Panel>}

        {mode === 'backdoor' && <Panel title="后门攻击与防御" subtitle="恶意真值与防御判断分别记录"><div className="config-grid"><label><span>攻击类型</span><Select value={backdoorAttack} onChange={setBackdoorAttack} options={[{ value: 'trigger_injection', label: '触发器注入' }, { value: 'label_poisoning', label: '标签污染' }, { value: 'model_update_poisoning', label: '模型更新污染' }]} /></label><label><span>启用攻击防御</span><Switch checked={backdoorDefense} onChange={setBackdoorDefense} /></label><label><span>默认恶意客户端比例</span><Slider value={maliciousRatio} min={1} max={50} onChange={setMaliciousRatio} tooltip={{ formatter: (value) => `${value}%` }} /></label><label><span>指定恶意客户端（可选）</span><Select mode="multiple" value={maliciousClients} placeholder="留空则按默认比例选择" onChange={setMaliciousClients} options={nodeRows.map((node) => ({ value: node.id, label: node.id }))} maxTagCount="responsive" /></label><label><span>攻击起始轮次</span><InputNumber value={startRound} min={0} max={rounds} onChange={(value) => setStartRound(value ?? 1)} /></label><label><span>注入比例</span><Slider value={poisonRatio} min={1} max={100} onChange={setPoisonRatio} tooltip={{ formatter: (value) => `${value}%` }} /></label><label><span>目标类别</span><InputNumber value={targetLabel} min={0} max={64} onChange={(value) => setTargetLabel(value ?? 0)} /></label></div>{backdoorDefense && <div className="defense-stage-grid"><label className={featureDefense ? 'active' : ''}><Switch checked={featureDefense} onChange={setFeatureDefense} /><span><b>特征统计阶段防御</b><small>识别并过滤异常特征摘要</small></span></label><label className={trainingDefense ? 'active' : ''}><Switch checked={trainingDefense} onChange={setTrainingDefense} /><span><b>正常训练阶段防御</b><small>识别并过滤异常模型更新</small></span></label></div>}</Panel>}
      </div>

      <aside className="experiment-summary"><Panel title="运行摘要" subtitle="后端提交前校验"><div className="summary-scene"><span>基础异构环境</span><Tag color={scenarioId ? 'cyan' : 'warning'}>{scenarioId ? scenarioVersion || scenarioId : '尚未应用'}</Tag></div><div className="summary-list"><div><span>实验类型</span><b>{modeCards.find((item) => item.key === mode)?.title}</b></div><div><span>运行方案</span><b>{methodOptions.find((item) => item.value === method)?.label}</b></div><div><span>客户端参与</span><b>{participationRate}%</b></div><div><span>训练轮次</span><b>{rounds}</b></div><div><span>保护/防御</span><b>{mode === 'privacy' ? (privacyDefense ? '启用' : '关闭') : mode === 'backdoor' ? (backdoorDefense ? '启用' : '关闭') : '不适用'}</b></div></div>{scenarioId ? <div className="validation-pass"><CheckCircleFilled />场景快照已绑定</div> : <Alert type="warning" showIcon message="请先设置基础异构环境" action={<Button size="small" onClick={() => navigate('/scenario-analysis')}>前往设置</Button>} />}{preflightResult && <div className="preflight-checks">{preflightResult.checks.map((check) => <div key={check.name} className={check.ready ? 'ready' : 'failed'}><i>{check.ready ? '✓' : '!'}</i><span>{check.message}</span></div>)}</div>}<Space direction="vertical" className="experiment-actions" style={{ width: '100%' }}><Button block icon={<DownloadOutlined />} onClick={() => exportExperimentConfig(config)}>导出当前配置</Button><Button block loading={preflighting} disabled={!scenarioId || Boolean(capabilityError)} onClick={runPreflight}>运行预检</Button><Button type="primary" size="large" block icon={<RocketOutlined />} loading={submitting} disabled={!scenarioId || Boolean(capabilityError)} onClick={startExperiment}>创建并启动实验</Button></Space></Panel></aside>
    </div>
  </div>;
}
