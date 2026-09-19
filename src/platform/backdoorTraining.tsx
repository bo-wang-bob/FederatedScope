import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Card, InputNumber, Modal, Select, Space } from 'antd';
import { FileTextOutlined, PlayCircleOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons';
import { api, statusText, type BackdoorTrainingJob, type BackdoorTrainingStatus } from './api';
import { DatasetUpload } from './datasetUpload';
import './backdoor.css';

const EXPORT_STAGE = '配置测试集';
const DEFAULT_STAGES = ['干净基线（无攻击）', 'SABRE 后门攻击', 'SABRE 攻击 + multi_metrics 防御'];

// 后门研究的训练面板: 选一个上传的数据集, 依次跑 干净基线 -> SABRE 攻击 -> SABRE+防御,
// 全部完成后自动导出测试集并登记结果组, 页面下方的挑图对比立即切换到新结果。
export function BackdoorTrainingPanel({ onCompleted }: { onCompleted?: () => void }) {
  const [status, setStatus] = useState<BackdoorTrainingStatus>();
  const [finished, setFinished] = useState<BackdoorTrainingJob>();
  const [datasetId, setDatasetId] = useState<string>();
  const [rounds, setRounds] = useState<number>();
  const [refresh, setRefresh] = useState(0);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');
  const [logs, setLogs] = useState('');
  const [logsOpen, setLogsOpen] = useState(false);
  const tracking = useRef<string | undefined>(undefined);
  const notified = useRef(false);
  // 接口异常或旧版本后端返回非对象时按“暂无数据”处理, 不让面板阻塞挑图流程。
  const safe = status && !Array.isArray(status) ? status : undefined;
  const datasets = Array.isArray(safe?.datasets) ? safe!.datasets : [];
  const templates = Array.isArray(safe?.templates) ? safe!.templates : [];
  const job = safe?.job || finished;
  const stages = [...(templates.length ? templates.map(t => t.label) : DEFAULT_STAGES), EXPORT_STAGE];
  const running = !!safe?.job;
  const selected = datasetId ?? safe?.group?.datasetId ?? datasets.at(-1)?.id;

  useEffect(() => {
    let active = true;
    void api<BackdoorTrainingStatus>('backdoor/training').then(data => {
      if (!active) return;
      setStatus(data);
      setError('');
      if (data && !Array.isArray(data) && data.job) { setFinished(undefined); notified.current = false; }
    }).catch(e => { if (active) setError(String((e as Error).message || e)); });
    return () => { active = false; };
  }, [refresh]);

  // 运行中每 2.5s 刷新; 任务从状态里消失时补拉一次终态, 完成则通知父级刷新测试集。
  useEffect(() => {
    if (safe?.job) {
      tracking.current = safe.job.id;
      const timer = setInterval(() => setRefresh(value => value + 1), 2500);
      return () => clearInterval(timer);
    }
    if (!tracking.current) return;
    const id = tracking.current;
    tracking.current = undefined;
    void api<BackdoorTrainingJob>('backdoor/training/' + id).then(final => {
      setFinished(final);
      if (final.status === 'completed' && !notified.current) {
        notified.current = true;
        onCompleted?.();
      }
    }).catch(() => { /* 终态暂时取不到就等下一次刷新 */ });
  }, [safe?.job?.id]);

  const start = async () => {
    if (running || starting || !datasets.length) return;
    setStarting(true); setError('');
    try {
      const created = await api<BackdoorTrainingJob>('backdoor/training', {
        ...(selected ? { datasetId: selected } : {}),
        ...(rounds ? { rounds } : {}),
      });
      tracking.current = created.id;
      notified.current = false;
      setFinished(undefined);
      setRefresh(value => value + 1);
    } catch (e) { setError(String((e as Error).message || e)); }
    finally { setStarting(false); }
  };

  const stop = async () => {
    if (!job?.id) return;
    try { await api<BackdoorTrainingJob>('backdoor/training/' + job.id + '/stop', {}); setRefresh(value => value + 1); }
    catch (e) { setError(String((e as Error).message || e)); }
  };

  const openLogs = async () => {
    if (!job?.id) return;
    try {
      const text = await api<string>('backdoor/training/' + job.id + '/logs');
      setLogs(typeof text === 'string' ? text : '');
    } catch (e) { setLogs(String((e as Error).message || e)); }
    setLogsOpen(true);
  };

  const stageIndex = job ? Math.min(job.stageIndex ?? 0, stages.length - 1) : 0;
  return <Card className="platform-panel backdoor-training" title={<span><PlayCircleOutlined /> 攻防实验</span>}
    extra={<Space>
      <DatasetUpload disabled={running || starting} onUploaded={() => setRefresh(value => value + 1)} />
      <Button icon={<ReloadOutlined />} disabled={running || starting} onClick={() => setRefresh(value => value + 1)}>刷新</Button>
    </Space>}>
    <div className="backdoor-training-form">
      <label><span>数据集</span>
        <Select aria-label="训练数据集" value={datasets.length ? selected : undefined}
          placeholder={datasets.length ? '选择数据集' : '暂无已上传的数据集'} disabled={running || starting}
          onChange={setDatasetId}
          options={datasets.map(d => ({ value: d.id, label: `${d.name}（${d.count} 张 · ${d.classes} 类）` }))} /></label>
      <label><span>训练轮数</span>
        <InputNumber aria-label="训练轮数" min={1} max={500} precision={0} value={rounds} disabled={running || starting}
          placeholder="默认" onChange={value => setRounds(typeof value === 'number' ? value : undefined)} /></label>
      <div className="backdoor-training-actions">
        {running ? <Button danger icon={<StopOutlined />} onClick={() => void stop()}>停止训练</Button> :
          <Button type="primary" icon={<PlayCircleOutlined />} loading={starting} disabled={!datasets.length}
            onClick={() => void start()}>启动训练</Button>}
        {job?.id && <Button icon={<FileTextOutlined />} onClick={() => void openLogs()}>查看日志</Button>}
      </div>
    </div>
    {job && <ol className="backdoor-training-stage" aria-label="训练进度">
      {stages.map((label, index) => {
        const state = job.status === 'completed' || index < stageIndex ? 'done'
          : index === stageIndex ? (job.status === 'failed' ? 'fail' : 'active') : '';
        return <li key={label} className={state}>{label}</li>;
      })}
    </ol>}
    {job && <p className="backdoor-training-meta">
      {running ? job.stage : `${statusText[job.status] || job.status} · ${job.stage}`} · {job.datasetName}
    </p>}
    {job?.status === 'failed' && job.error && <Alert className="backdoor-training-error" type="error" showIcon title={job.error}
      action={<Button size="small" onClick={() => void openLogs()}>日志</Button>} />}
    {!job && safe?.group && <Alert type="success" showIcon title={`当前结果组：${safe.group.datasetName}${safe.group.testsetName ? ` · 测试集：${safe.group.testsetName}` : ' · 未上传测试集'}`}
      description={safe.group.testsetName
        ? '三个实验已完成，下方挑图与对比即使用该结果与测试集；重新启动训练会生成新的结果组。'
        : '三个实验已完成；上传测试集后即可开始挑图对比，重新启动训练会生成新的结果组。'} />}
    {!running && !datasets.length && !error && <Alert type="info" showIcon title="还没有可用的数据集"
      description="点击右上角「添加数据集」上传按类别组织的图片文件夹，然后在这里启动三个攻防实验。" />}
    {!running && !!safe?.missing?.length && <Alert type="warning" showIcon title={'缺少实验配置模板：' + safe.missing.join('、')} />}
    {error && <Alert className="backdoor-training-error" type="error" showIcon title={error} />}
    <Modal open={logsOpen} title="训练日志（最近 256 KiB）" onCancel={() => setLogsOpen(false)} footer={null} width={920}>
      <pre className="backdoor-training-logs">{logs || '暂无日志'}</pre>
    </Modal>
  </Card>;
}
