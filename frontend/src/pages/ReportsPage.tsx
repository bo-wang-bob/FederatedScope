import { DownloadOutlined, FileExcelOutlined, FileImageOutlined, FileTextOutlined, SearchOutlined } from '@ant-design/icons';
import { Button, Input, Select, Space, Table, Tag } from 'antd';
import { Panel } from '../components/ChartPanel';
import { MetricCard } from '../components/MetricCard';
import { PageHeader } from '../components/PageHeader';

const reportRows = [
  ['RUN-0821-B','跨域联合认知演示','后门防御模拟','已完成','2026-08-15 14:18','30 轮'],
  ['RUN-0821-A','跨域联合认知演示','后门攻击模拟','已完成','2026-08-15 13:42','30 轮'],
  ['RUN-0819-P','隐私保护对照','隐私保护评估','已完成','2026-08-15 11:06','24 轮'],
  ['RUN-0819-R','隐私风险基线','隐私风险评估','已完成','2026-08-15 10:24','24 轮'],
  ['RUN-0818-H','重度异构场景','异构基线','已完成','2026-08-14 22:16','30 轮'],
  ['RUN-0817-M','中度异构场景','异构基线','已完成','2026-08-14 20:31','30 轮'],
].map(([id,scene,mode,status,time,rounds]) => ({ id,scene,mode,status,time,rounds }));

export function ReportsPage() {
  return <div className="page">
    <PageHeader eyebrow="EXPERIMENT REPORTS" title="实验档案与报告" description="检索历史实验，导出经过脱敏的配置、指标、图表和结论摘要。" actions={<Button type="primary" icon={<FileTextOutlined />}>生成对照报告</Button>} />
    <div className="metrics-grid four"><MetricCard label="实验总数" value="26" delta="本周新增 12" /><MetricCard label="已完成" value="23" delta="完成率 88.5%" tone="green" /><MetricCard label="安全实验" value="16" delta="隐私 8 · 后门 8" tone="violet" /><MetricCard label="报告归档" value="9" delta="最近更新 14:35" tone="amber" /></div>
    <Panel title="实验记录" subtitle="所有节点标识均为模拟逻辑编号" extra={<Space><Input prefix={<SearchOutlined />} placeholder="搜索实验或场景" /><Select value="全部模式" options={[{ value:'全部模式'},{ value:'异构基线'},{ value:'隐私评估'},{ value:'后门防御' }]} /></Space>}>
      <Table rowKey="id" dataSource={reportRows} columns={[{ title:'实验编号',dataIndex:'id',render:(v:string)=><button className="link-button">{v}</button>},{title:'场景',dataIndex:'scene'},{title:'模式',dataIndex:'mode',render:(v:string)=><Tag color={v.includes('后门')?'orange':v.includes('隐私')?'purple':'cyan'}>{v}</Tag>},{title:'状态',dataIndex:'status',render:(v:string)=><Tag color="success">{v}</Tag>},{title:'开始时间',dataIndex:'time'},{title:'训练轮次',dataIndex:'rounds'},{title:'导出',render:()=><Space><Button size="small" icon={<FileImageOutlined />}>PNG</Button><Button size="small" icon={<FileExcelOutlined />}>CSV</Button><Button size="small" icon={<DownloadOutlined />}>JSON</Button></Space>}]} />
    </Panel>
  </div>;
}
