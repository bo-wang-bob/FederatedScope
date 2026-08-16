import {
  CheckOutlined,
  DatabaseOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Drawer,
  InputNumber,
  Segmented,
  Slider,
  Space,
  Table,
  Tabs,
  Tag,
  Tooltip,
} from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { resolveScenarioDataAdapter } from '../api/scenarioAdapter';
import { Chart, chartGrid, chartText, Panel } from '../components/ChartPanel';
import { MetricCard } from '../components/MetricCard';
import { PageHeader } from '../components/PageHeader';
import { domains } from '../mock/data';
import { useAppStore } from '../store/useAppStore';
import type {
  ClientPartitionPreview,
  DomainKey,
  ScenarioPartitionPreview,
  ScenarioPreviewRequest,
} from '../types';
import {
  distributionSummary,
  generateScenarioPartition,
  officeHomeClassNames,
} from '../utils/partition';

const domainKeys: DomainKey[] = ['Art', 'Clipart', 'Product', 'Real_World'];
const alphaPresets = [0.1, 0.3, 0.5, 1, 10];
const scenarioAdapter = resolveScenarioDataAdapter();

function alphaToSlider(alpha: number) {
  return Math.log10(alpha);
}

function sliderToAlpha(value: number) {
  return Number(Math.pow(10, value).toFixed(2));
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function makeRequest(alpha: number, seed: number): ScenarioPreviewRequest {
  return {
    dataset: 'office-home',
    domains: domainKeys,
    clientsPerDomain: 15,
    partition: { strategy: 'dirichlet', alpha, seed },
  };
}

export function ScenarioAnalysisPage() {
  const [draftAlpha, setDraftAlpha] = useState(0.3);
  const [appliedAlpha, setAppliedAlpha] = useState(0.3);
  const [seed, setSeed] = useState(20_260_815);
  const [appliedSeed, setAppliedSeed] = useState(20_260_815);
  const [selectedDomainKey, setSelectedDomainKey] = useState<DomainKey>('Art');
  const [distributionMode, setDistributionMode] = useState<'样本数量' | '类别比例'>('样本数量');
  const [preview, setPreview] = useState<ScenarioPartitionPreview>(() => generateScenarioPartition());
  const [appliedPreview, setAppliedPreview] = useState<ScenarioPartitionPreview>(() => generateScenarioPartition());
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [selectedClient, setSelectedClient] = useState<ClientPartitionPreview>();
  const { scenarioId, setScenario } = useAppStore();

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    scenarioAdapter.preview(makeRequest(draftAlpha, seed), controller.signal)
      .then((result) => {
        setPreview(result);
        setPreviewError('');
      })
      .catch((error: Error) => {
        if (error.name !== 'AbortError') setPreviewError(error.message);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [draftAlpha, seed]);

  const currentSummary = useMemo(() => distributionSummary(appliedPreview), [appliedPreview]);
  const draftSummary = useMemo(() => distributionSummary(preview), [preview]);
  const selectedDomain = preview.domains.find((domain) => domain.domainKey === selectedDomainKey)!;
  const domainDefinition = domains.find((domain) => domain.id === selectedDomainKey)!;

  const applyScenario = async () => {
    setApplying(true);
    try {
      const record = await scenarioAdapter.create(makeRequest(draftAlpha, seed));
      setAppliedAlpha(draftAlpha);
      setAppliedSeed(seed);
      setPreview(record.preview);
      setAppliedPreview(record.preview);
      setScenario(record.scenarioId, record.preview.partitionVersion);
      setPreviewError('');
    } catch (error) {
      setPreviewError(error instanceof Error ? error.message : '场景应用失败');
    } finally {
      setApplying(false);
    }
  };
  const sampleOption = {
    tooltip: { trigger: 'axis', formatter: (params: Array<{ name: string; value: number }>) => `${params[0].name}<br/>样本数量 ${params[0].value.toLocaleString()}` },
    grid: { left: 58, right: 18, top: 18, bottom: 46 },
    xAxis: { type: 'category', data: selectedDomain.clients.map((client) => client.clientId.split('-').at(-1)), axisLabel: { color: chartText, rotate: 35 }, axisLine: { lineStyle: { color: chartGrid } } },
    yAxis: { type: 'value', name: '样本数', nameTextStyle: { color: chartText }, axisLabel: { color: chartText }, splitLine: { lineStyle: { color: chartGrid } } },
    series: [{ type: 'bar', data: selectedDomain.clients.map((client) => ({ value: client.sampleCount, itemStyle: { color: domainDefinition.color } })), barMaxWidth: 22, itemStyle: { borderRadius: [3, 3, 0, 0] } }],
  };
  const classHeatmapData = selectedDomain.clients.flatMap((client, clientIndex) =>
    client.classHistogram.map((count, classIndex) => [
      classIndex,
      clientIndex,
      distributionMode === '样本数量' ? count : Number((client.classProportions[classIndex] * 100).toFixed(2)),
    ]));
  const heatmapMaximum = Math.max(1, ...classHeatmapData.map((item) => Number(item[2])));
  const classHeatmapOption = {
    tooltip: { formatter: (params: { value: number[] }) => {
      const client = selectedDomain.clients[params.value[1]];
      return `${client.clientId}<br/>${officeHomeClassNames[params.value[0]]}<br/>${distributionMode}：${params.value[2]}${distributionMode === '类别比例' ? '%' : ''}`;
    } },
    grid: { left: 92, right: 28, top: 16, bottom: 78 },
    xAxis: { type: 'category', data: officeHomeClassNames, axisLabel: { color: chartText, interval: 4, rotate: 45, fontSize: 8 }, axisLine: { lineStyle: { color: chartGrid } } },
    yAxis: { type: 'category', data: selectedDomain.clients.map((client) => client.clientId.split('-').at(-1)), axisLabel: { color: chartText, fontSize: 9 }, axisLine: { lineStyle: { color: chartGrid } } },
    dataZoom: [{ type: 'inside', xAxisIndex: 0 }, { type: 'slider', xAxisIndex: 0, height: 12, bottom: 22, borderColor: chartGrid, textStyle: { color: chartText } }],
    visualMap: { min: 0, max: heatmapMaximum, calculable: false, orient: 'vertical', right: 0, top: 'middle', textStyle: { color: chartText }, inRange: { color: ['#0c2233', '#176383', domainDefinition.color, '#fff3c4'] } },
    series: [{ type: 'heatmap', data: classHeatmapData, progressive: 500, itemStyle: { borderColor: 'rgba(7,16,31,.45)', borderWidth: 1 } }],
  };

  const columns = [
    { title: '客户端编号', dataIndex: 'clientId', fixed: 'left' as const, width: 128, render: (value: string, row: ClientPartitionPreview) => <button className="link-button" onClick={() => setSelectedClient(row)}>{value}</button> },
    { title: '样本数', dataIndex: 'sampleCount', width: 90, sorter: (a: ClientPartitionPreview, b: ClientPartitionPreview) => a.sampleCount - b.sampleCount, render: (value: number) => value.toLocaleString() },
    { title: '域内占比', dataIndex: 'domainSampleRatio', width: 88, render: percent },
    { title: '类别覆盖', dataIndex: 'coveredClassCount', width: 84, render: (value: number) => `${value} / 65` },
    { title: '缺失类别', dataIndex: 'missingClassCount', width: 78 },
    { title: '最大类别及占比', key: 'dominant', width: 150, render: (_: unknown, row: ClientPartitionPreview) => `${officeHomeClassNames[row.dominantClassIndex]} · ${percent(row.dominantClassRatio)}` },
    { title: '标签熵', dataIndex: 'labelEntropy', width: 76, render: (value: number) => value.toFixed(2) },
  ];

  return (
    <div className="page scenario-analysis-page">
      <PageHeader
        eyebrow="SCENARIO & HETEROGENEITY"
        title="场景与异构分析"
        description="设置基础异构环境，并查看四个域中每个客户端的样本数量和类别分布。"
        actions={<Space><Tag color="cyan">OfficeHome · 4 域 × 15 客户端</Tag><Tag color={preview.basis === 'actual_dataset' ? 'green' : 'gold'}>{preview.basis === 'actual_dataset' ? '数据目录实测' : '内置数据规模仿真'}</Tag><Tag color={scenarioId ? 'success' : 'default'}>{scenarioId ? '场景已应用' : '场景未应用'}</Tag></Space>}
      />

      {previewError && <Alert type="error" showIcon message="场景服务请求失败" description={previewError} />}

      <div className="scenario-analysis-top">
        <Panel title="基础异构环境" subtitle="设置域内数据分布" className="alpha-panel" extra={<Tag color={draftAlpha === appliedAlpha && seed === appliedSeed && scenarioId ? 'success' : 'warning'}>{draftAlpha === appliedAlpha && seed === appliedSeed && scenarioId ? '已应用' : '待应用'}</Tag>}>
          <div className="alpha-value"><small>狄利克雷参数 α</small><strong>{draftAlpha.toFixed(2)}</strong><span>{draftAlpha <= 0.2 ? '高度不均衡' : draftAlpha < 1 ? '中度不均衡' : '接近均衡'}</span></div>
          <Slider
            min={Math.log10(0.05)}
            max={1}
            step={0.01}
            value={alphaToSlider(draftAlpha)}
            onChange={(value) => setDraftAlpha(sliderToAlpha(value))}
            tooltip={{ formatter: (value) => value === undefined ? '' : `α = ${sliderToAlpha(value).toFixed(2)}` }}
            marks={{ [Math.log10(0.05)]: '0.05', [-1]: '0.1', [Math.log10(0.3)]: '0.3', [0]: '1', [1]: '10' }}
          />
          <div className="alpha-presets">{alphaPresets.map((value) => <Button key={value} size="small" type={draftAlpha === value ? 'primary' : 'default'} onClick={() => setDraftAlpha(value)}>α {value.toFixed(value < 1 ? 1 : 0)}</Button>)}</div>
          <label className="seed-control"><span>随机种子<small>仅用于复现，不改变异构定义</small></span><InputNumber value={seed} min={1} precision={0} onChange={(value) => setSeed(value ?? 20_260_815)} /></label>
          <Space className="alpha-actions"><Button icon={<ReloadOutlined />} onClick={() => { setDraftAlpha(0.3); setSeed(20_260_815); }}>重置</Button><Button type="primary" loading={loading || applying} icon={<CheckOutlined />} disabled={draftAlpha === appliedAlpha && seed === appliedSeed && Boolean(scenarioId)} onClick={applyScenario}>应用到实验配置</Button></Space>
        </Panel>

        <Panel title="当前方案 → 待应用方案" subtitle={`划分版本 ${preview.partitionVersion}`} className="partition-diff-panel">
          <div className="partition-flow">
            <div><small>当前 α</small><b>{appliedAlpha.toFixed(2)}</b><span>{currentSummary.minSamples}～{currentSummary.maxSamples} 样本/客户端</span></div>
            <i>→</i>
            <div className={draftAlpha !== appliedAlpha || seed !== appliedSeed ? 'pending' : ''}><small>预览 α</small><b>{draftAlpha.toFixed(2)}</b><span>{draftSummary.minSamples}～{draftSummary.maxSamples} 样本/客户端</span></div>
          </div>
          <div className="preview-metric-grid">
            <MetricCard label="平均类别覆盖" value={draftSummary.meanCoverage.toFixed(1)} suffix=" / 65" delta={`当前 ${currentSummary.meanCoverage.toFixed(1)}`} tone="green" />
            <MetricCard label="最大类别占比" value={(draftSummary.maxDominantRatio * 100).toFixed(1)} suffix="%" delta={`当前 ${(currentSummary.maxDominantRatio * 100).toFixed(1)}%`} tone="amber" />
            <MetricCard label="平均标签熵" value={draftSummary.meanEntropy.toFixed(2)} delta={`当前 ${currentSummary.meanEntropy.toFixed(2)}`} tone="violet" />
          </div>
        </Panel>
      </div>

      <Panel title="四域客户端分布明细" subtitle="逐域核对全部 60 个客户端的样本数量、类别计数和类别比例" extra={<Space><DatabaseOutlined /><span className="muted">{preview.domains.reduce((sum, domain) => sum + domain.totalSamples, 0).toLocaleString()} 条训练样本</span></Space>}>
        <Tabs
          activeKey={selectedDomainKey}
          onChange={(value) => setSelectedDomainKey(value as DomainKey)}
          items={domains.map((domain) => ({
            key: domain.id,
            label: <span className="domain-tab-label"><i style={{ background: domain.color }} />{domain.name}<small>{preview.domains.find((item) => item.domainKey === domain.id)?.totalSamples.toLocaleString()}</small></span>,
          }))}
        />
        <div className="domain-distribution-summary">
          <span><small>后端域键</small><b>{selectedDomain.domainKey}</b></span>
          <span><small>域内客户端</small><b>15</b></span>
          <span><small>域总样本</small><b>{selectedDomain.totalSamples.toLocaleString()}</b></span>
          <span><small>类别数量</small><b>{selectedDomain.classCount}</b></span>
        </div>
        <div className="distribution-chart-grid">
          <Panel title={`${domainDefinition.name} · 客户端样本数量`} subtitle="每根柱对应一个客户端"><Chart option={sampleOption} height={310} /></Panel>
          <Panel title="客户端 × 类别分布热力图" subtitle="连续色阶表示值，支持横向缩放" extra={<Segmented size="small" options={['样本数量', '类别比例']} value={distributionMode} onChange={(value) => setDistributionMode(value as '样本数量' | '类别比例')} />}><Chart option={classHeatmapOption} height={390} /></Panel>
        </div>
        <div className="distribution-table-head"><div><b>客户端分布表</b><span>点击编号查看全部 65 个类别</span></div><Tooltip title="计数来自预览数据，比例由计数校验后派生"><Tag color="cyan">数量与比例已校验</Tag></Tooltip></div>
        <Table<ClientPartitionPreview>
          rowKey="clientId"
          columns={columns}
          dataSource={selectedDomain.clients}
          pagination={false}
          size="small"
          scroll={{ x: 760 }}
          rowClassName={(row) => row.clientId === selectedClient?.clientId ? 'selected-distribution-row' : ''}
          onRow={(row) => ({ onClick: () => setSelectedClient(row) })}
        />
      </Panel>

      <Drawer title={`客户端分布详情 · ${selectedClient?.clientId ?? ''}`} width={520} open={Boolean(selectedClient)} onClose={() => setSelectedClient(undefined)}>
        {selectedClient && <div className="distribution-drawer">
          <div className="detail-grid">
            <div><span>样本数量</span><b>{selectedClient.sampleCount.toLocaleString()}</b></div>
            <div><span>域内占比</span><b>{percent(selectedClient.domainSampleRatio)}</b></div>
            <div><span>类别覆盖</span><b>{selectedClient.coveredClassCount} / 65</b></div>
            <div><span>标签熵</span><b>{selectedClient.labelEntropy.toFixed(2)}</b></div>
          </div>
          <div className="all-class-list"><h4>全部类别数量与比例</h4>{selectedClient.classHistogram.map((count, index) => <div key={officeHomeClassNames[index]}><span>{officeHomeClassNames[index]}</span><b>{count.toLocaleString()}</b><em>{percent(selectedClient.classProportions[index])}</em><i><u style={{ width: `${Math.min(100, selectedClient.classProportions[index] * 400)}%`, background: domainDefinition.color }} /></i></div>)}</div>
        </div>}
      </Drawer>
    </div>
  );
}
