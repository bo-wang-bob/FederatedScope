import { lazy, Suspense } from 'react';
const Chart = lazy(() => import('../components/ChartPanel').then(module => ({default:module.Chart})));
import { Empty } from 'antd';
import type { Client, Job, Point, Resource } from './api';
const colors = ['#8abfc7', '#a5b88e', '#859fcd', '#cfb380', '#b399c5', '#cc9898'];
const axisStyle = { axisLine: { lineStyle: { color: '#435a68' } }, axisTick: { lineStyle: { color: '#435a68' } },
  axisLabel: { color: '#a6becd', fontSize: 12, hideOverlap: true }, nameTextStyle: { color: '#a6becd', fontSize: 12 },
  splitLine: { lineStyle: { color: '#2a3a45' } } };
function Plot({ option, style }: { option: { xAxis?: object; yAxis?: object; legend?: object; visualMap?: object; [key: string]: unknown }; style: { height: number } }) {
  return <Suspense fallback={<div className="chart-loading">正在加载图表…</div>}><Chart option={{ ...option,
    ...(option.xAxis && { xAxis: { ...axisStyle, ...option.xAxis } }),
    ...(option.yAxis && { yAxis: { ...axisStyle, ...option.yAxis } }),
    ...(option.legend && { legend: { textStyle: { color: '#afc3cf', fontSize: 12 }, pageTextStyle: { color: '#afc3cf' }, pageIconColor: '#8abfc7', ...option.legend } }),
    ...(option.visualMap && { visualMap: { textStyle: { color: '#afc3cf', fontSize: 12 }, ...option.visualMap } }),
    animation: !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches, animationDuration: 200, animationDurationUpdate: 160 }} height={style.height} /></Suspense>;
}
const common = { backgroundColor: 'transparent', color: colors, textStyle: { color: '#afc3cf', fontFamily: 'Segoe UI, Microsoft YaHei, sans-serif', fontSize: 12 },
  tooltip: { trigger: 'axis', backgroundColor: '#20313d', borderColor: '#47606f', textStyle: { color: '#d5e4ed', fontSize: 14 } }, grid: { left: 50, right: 44, top: 45, bottom: 38 } };
export function Curves({ points }: { points: Point[] }) {
  if (!points.length) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="等待训练指标" />;
  const domains = Object.keys(points[0].domains);
  return <Plot style={{ height: 330 }} option={{ ...common, legend: { type: 'scroll', top: 0 },
    xAxis: { type: 'category', name: '轮次', data: points.map(p => p.round === 0 ? '初始化' : p.round) },
    yAxis: { type: 'value', name: '准确率 %', min: 0, max: 100 },
    series: [{ name: '总体', type: 'line', symbolSize: 5, lineStyle: { width: 2.5 }, areaStyle: { opacity: .04 }, data: points.map(p => p.accuracy * 100) },
      ...domains.map(name => ({ name, type: 'line', symbolSize: 4, data: points.map(p => p.domains[name] * 100) }))] }} />;
}
export function LossChart({ points }: { points: Point[] }) {
  const values = points.filter(p => p.trainLoss != null);
  if (!values.length) return <Empty description="尚无本地训练损失" />;
  return <Plot style={{ height: 250 }} option={{ ...common,
    xAxis: { type: 'category', data: values.map(p => p.round), name: '轮次' },
    yAxis: { type: 'value', name: '样本加权训练损失', scale: true },
    series: [{ type: 'line', data: values.map(p => p.trainLoss), areaStyle: { opacity: .08 } }] }} />;
}
export function Topology({ clients }: { clients: Client[] }) {
  if (!clients.length) return <Empty description="等待客户端数据" />;
  const domains = [...new Set(clients.map(c => c.domain))];
  const nodes: object[] = [{ id: 'server', name: '联邦\n聚合', x: 350, y: 190, symbolSize: 80,
    itemStyle: { color: '#263e4b', borderWidth: 2, borderColor: '#87b5c3' }, label: { show: true, color: '#c1e1ea', fontSize: 14 } }];
  const links: object[] = [];
  for (const [i, domain] of domains.entries()) {
    const a = (Math.PI * 2 * i / domains.length) - Math.PI / 2;
    const center = { x: 350 + Math.cos(a) * 210, y: 190 + Math.sin(a) * 130 };
    const group = clients.filter(c => c.domain === domain);
    nodes.push({ id: domain, name: `${domain}\n${group.length} 客户端`, ...center, symbolSize: 57,
      itemStyle: { color: '#263a45', borderColor: colors[i % colors.length], borderWidth: 1.5 }, label: { show: true, position: 'bottom', color: '#bbcfdb', fontSize: 12 } });
    links.push({ source: 'server', target: domain });
    group.forEach((c, j) => {
      const angle = Math.PI * 2 * j / group.length;
      const id = `client-${c.id}`;
      nodes.push({ id, name: `客户端 ${c.id} · ${c.domain}\n${c.samples} 样本 · ${c.stage}`,
        x: center.x + Math.cos(angle) * 70, y: center.y + Math.sin(angle) * 50,
        symbolSize: c.stage === '本地训练' ? 15 : 7, itemStyle: { color: colors[i % colors.length] } });
      links.push({ source: domain, target: id, lineStyle: { opacity: .2 } });
    });
  }
  return <Plot style={{ height: 355 }} option={{ ...common, tooltip: { ...common.tooltip, trigger: 'item' },
    series: [{ type: 'graph', layout: 'none', roam: false, data: nodes, links,
      lineStyle: { color: '#9fb2c5', width: 1.2 }, emphasis: { focus: 'adjacency' } }] }} />;
}
export function Distribution({ clients, classes }: { clients: Client[]; classes: string[] }) {
  if (!clients.length) return <Empty description="暂无真实划分数据" />;
  const data = clients.flatMap((c, i) => c.histogram.map((n, k) => [k, i, n]));
  return <Plot style={{ height: 320 }} option={{ ...common, tooltip: { ...common.tooltip, trigger: 'item', position: 'top' },
    grid: { left: 65, right: 20, top: 15, bottom: 80 },
    xAxis: { type: 'category', data: classes }, yAxis: { type: 'category', data: clients.map(c => `C${c.id}`) },
    dataZoom: [{ type: 'inside', yAxisIndex: 0 }, { type: 'slider', xAxisIndex: 0, bottom: 22 }],
    visualMap: { min: 0, max: data.reduce((max, row) => Math.max(max, row[2]), 1), calculable: true, orient: 'horizontal', bottom: 0,
      inRange: { color: ['#1c2d38', '#496f7c', '#a5cbbf'] } }, series: [{ type: 'heatmap', data }] }} />;
}
export function ResourcesChart({ history }: { history: Resource[] }) {
  return <Plot style={{ height: 225 }} option={{ ...common, legend: { top: 0 },
    xAxis: { type: 'category', data: history.map(r => new Date(r.at).toLocaleTimeString()), axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', min: 0, max: 100, name: '%' },
    series: [{ name: 'CPU', type: 'line', showSymbol: false, data: history.map(r => r.cpuPercent) },
      ...(history.at(-1)?.gpus || []).map(g => ({ name: `GPU ${g.index}`, type: 'line', showSymbol: false,
        data: history.map(r => r.gpus.find(x => x.index === g.index)?.utilization ?? null) }))] }} />;
}
export function DomainBars({ domains }: { domains: Record<string, { accuracy: number }> }) {
  return <Plot style={{ height: 280 }} option={{ ...common,
    xAxis: { type: 'category', data: Object.keys(domains) }, yAxis: { type: 'value', min: 0, max: 100, name: '准确率 %' },
    series: [{ type: 'bar', barMaxWidth: 44, itemStyle: { borderRadius: [5, 5, 0, 0] }, data: Object.values(domains).map(d => d.accuracy * 100) }] }} />;
}
export function Confusion({ matrix, classes }: { matrix: number[][]; classes: string[] }) {
  const data = matrix.flatMap((row, y) => row.map((v, x) => [x, y, v]));
  return <Plot style={{ height: 360 }} option={{ ...common, tooltip: { ...common.tooltip, trigger: 'item', position: 'top' },
    grid: { top: 20, right: 25, bottom: 70, left: 90 },
    xAxis: { type: 'category', name: '预测', data: classes }, yAxis: { type: 'category', name: '真实', data: classes },
    dataZoom: [{ type: 'inside', xAxisIndex: 0 }, { type: 'inside', yAxisIndex: 0 }],
    visualMap: { min: 0, max: Math.max(1, ...matrix.map(r => Math.max(...r))), orient: 'horizontal', bottom: 0,
      inRange: { color: ['#1c2d38', '#496f7c', '#a5cbbf'] } }, series: [{ type: 'heatmap', data }] }} />;
}
export function CompareCurves({ jobs }: { jobs: Job[] }) {
  if (!jobs.some(job => job.metrics.length)) return <Empty description="没有训练曲线" />;
  return <Plot style={{height:390}} option={{...common,legend:{type:'scroll',top:0},
    xAxis:{type:'value',min:0,minInterval:1,name:'通信轮次'},yAxis:{type:'value',min:0,max:100,name:'准确率 %'},
    series:jobs.map(job=>({name:(job.request.name || job.id.slice(0,8)).slice(0,40),type:'line',symbolSize:6,lineStyle:{width:2.5},
      data:job.metrics.map(point=>[point.round,point.accuracy*100])})),
  }}/>;
}
