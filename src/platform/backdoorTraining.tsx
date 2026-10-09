import { useEffect, useRef, useState } from 'react';
import { Alert, Button, InputNumber, Modal, Progress, Select, Space, Steps, Tag } from 'antd';
import { ArrowRightOutlined, FileTextOutlined, PlayCircleOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons';
import { api, statusText, terminal, type BackdoorTrainingAttack, type BackdoorTrainingJob, type BackdoorTrainingStatus } from './api';
import { Panel, State, Stat } from './ui';
import { DatasetUpload } from './datasetUpload';
import './backdoor.css';

const EXPORT_STAGE = '配置测试集';
const DEFAULT_STAGES = ['干净基线（无攻击）', 'SABRE 后门攻击', 'SABRE 攻击 + 防御'];

// 后门研究的训练面板: 选一个上传的数据集 + 一种后门攻击, 依次跑
// 干净基线 -> 攻击 -> 攻击+防御, 全部完成后自动导出测试集并登记结果组,
// 页面下方的挑图对比立即切换到新结果。
// 可选的攻击由后端的 attack catalog 提供（数据投毒/模型投毒/常见后门/新型后门）。
export function BackdoorTrainingPanel({ onCompleted, onOpenCompare }: {
  onCompleted?: () => void; onOpenCompare?: () => void;
}) {
  const [status, setStatus] = useState<BackdoorTrainingStatus>();
  const [finished, setFinished] = useState<BackdoorTrainingJob>();
  const [datasetId, setDatasetId] = useState<string>();
  const [rounds, setRounds] = useState<number>();
  const [attack, setAttack] = useState<string>();
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
  const attacks: BackdoorTrainingAttack[] = Array.isArray(safe?.attacks) && safe!.attacks.length
    ? safe!.attacks
    : [];
  // 用户本次的选择优先, 其次沿用上次任务/后端默认。
  // 旧后端不返回 attack catalog: 此时不渲染选择框, 也不提交 attack 字段。
  const selectedAttack = attacks.length
    ? (attack ?? safe?.attack ?? safe?.defaultAttack ?? attacks[0].key)
    : undefined;
  const attackName = attacks.find(item => item.key === selectedAttack)?.name;
  const job = safe?.job || finished;
  const fallbackStages = attackName
    ? ['干净基线（无攻击）', `${attackName} 后门攻击`, `${attackName} 攻击 + 防御`]
    : DEFAULT_STAGES;
  const stages = [...(templates.length ? templates.map(t => t.label) : fallbackStages), EXPORT_STAGE];
  // 终态任务不该再显示"运行中/停止训练"; 后端的 status() 只回运行中的任务,
  // 这里再挡一道, 让"查看终态任务"也能正确渲染成完成态。
  const running = !!safe?.job && !terminal(safe.job.status);
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
    if (safe?.job && !terminal(safe.job.status)) {
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
        ...(selectedAttack ? { attack: selectedAttack } : {}),
      });
      if (selectedAttack) setAttack(selectedAttack);
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
  const experimentCount = job?.total ?? 3;
  const done = job?.status === 'completed';
  const failed = job?.status === 'failed';
  const percent = done ? 100
    : Math.round(stageIndex / Math.max(1, stages.length) * 100);
  return <div className="backdoor-training">
    <div className="backdoor-training-workflow">
      <Panel title="实验配置" extra={<Space>
        <DatasetUpload disabled={running || starting} onUploaded={() => setRefresh(value => value + 1)} />
        <Button icon={<ReloadOutlined />} disabled={running || starting} onClick={() => setRefresh(value => value + 1)}>刷新</Button>
      </Space>}>
        <div className="backdoor-training-form">
          <label><span>数据集</span>
            <Select aria-label="训练数据集" value={datasets.length ? selected : undefined}
              placeholder={datasets.length ? '选择数据集' : '暂无已上传的数据集'} disabled={running || starting}
              onChange={setDatasetId}
              options={datasets.map(d => ({ value: d.id, label: `${d.name}（${d.count} 张 · ${d.classes} 类）` }))} /></label>
          {!!attacks.length && <label><span>攻击类型</span>
            <Select aria-label="后门攻击类型" value={selectedAttack} disabled={running || starting}
              onChange={setAttack}
              options={attacks.map(item => ({ value: item.key, label: item.name }))} /></label>}
          <label><span>训练轮数</span>
            <InputNumber aria-label="训练轮数" min={1} max={500} precision={0} value={rounds} disabled={running || starting}
              placeholder="默认" onChange={value => setRounds(typeof value === 'number' ? value : undefined)} /></label>
        </div>
        <div className="backdoor-training-actions">
          {running ? <Button danger icon={<StopOutlined />} onClick={() => void stop()}>停止训练</Button> :
            <Button type="primary" icon={<PlayCircleOutlined />} loading={starting} disabled={!datasets.length}
              onClick={() => void start()}>启动训练</Button>}
          {job?.id && <Button icon={<FileTextOutlined />} onClick={() => void openLogs()}>查看日志</Button>}
        </div>
        {!running && !datasets.length && !error && <Alert type="info" showIcon title="还没有可用的数据集"
          description="点击右上角「添加数据集」上传按类别组织的图片文件夹，然后在这里启动三个攻防实验。" />}
        {!running && !!safe?.missing?.length && <Alert type="warning" showIcon title={'缺少实验配置模板：' + safe.missing.join('、')} />}
        {error && <Alert className="backdoor-training-error" type="error" showIcon title={error} />}
      </Panel>
      <div className="backdoor-training-monitor">
        <Panel title="训练进度" extra={job ? <State value={job.status} /> : <Tag>等待启动</Tag>}>
          {job ? <>
            <div className="backdoor-training-heading">
              <strong>{job.datasetName}</strong>
              {job.attackName && <Tag color="blue">{job.attackName}</Tag>}
            </div>
            <Steps className="backdoor-training-steps" size="small" current={finished ? stages.length : stageIndex}
              status={failed ? 'error' : finished ? 'finish' : 'process'}
              items={stages.map(label => ({ title: label }))} />
            <Progress percent={percent} status={failed ? 'exception' : finished ? 'success' : 'active'} />
            <div className="backdoor-training-stats">
              <Stat label="实验进度" value={`${Math.min(stageIndex, experimentCount)} / ${experimentCount}`} sub="干净基线 → 攻击 → 防御" />
              <Stat label="当前实验" value={stages[stageIndex] ?? '—'} sub={done ? '全部完成' : '进行中'} />
            </div>
            <p className="backdoor-training-meta">
              {running ? job.stage : `${statusText[job.status] || job.status} · ${job.stage}`} · {job.datasetName}
              {job.attackName ? ` · ${job.attackName}` : ''}
            </p>
            {failed && job.error && <Alert className="backdoor-training-error" type="error" showIcon title={job.error}
              action={<Button size="small" onClick={() => void openLogs()}>日志</Button>} />}
            {done && onOpenCompare && <div className="backdoor-training-next">
              <Button type="primary" icon={<ArrowRightOutlined />} onClick={onOpenCompare}>前往对比测试</Button>
            </div>}
          </> : <div className="backdoor-training-await">
            <PlayCircleOutlined />
            <h3>准备开始训练</h3>
            <p>选择数据集与攻击类型后点击「启动训练」，三个攻防实验（干净基线 → 攻击 → 攻击 + 防御）的进度会显示在这里。</p>
          </div>}
        </Panel>
        {!job && safe?.group && <Alert type="success" showIcon title={`当前结果组：${safe.group.datasetName}${safe.group.attackName ? ` · ${safe.group.attackName}` : ''}${safe.group.testsetName ? ` · 测试集：${safe.group.testsetName}` : ' · 未上传测试集'}`}
          description={safe.group.testsetName
            ? '三个实验已完成，切到「对比测试」即可挑图并查看攻防对照；重新启动训练会生成新的结果组。'
            : '三个实验已完成；上传测试集后即可挑图对比，重新启动训练会生成新的结果组。'} />}
      </div>
    </div>
    <Modal open={logsOpen} title="训练日志（最近 256 KiB）" onCancel={() => setLogsOpen(false)} footer={null} width={920}>
      <pre className="backdoor-training-logs">{logs || '暂无日志'}</pre>
    </Modal>
  </div>;
}
