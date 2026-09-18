import { useRef, useState } from 'react';
import { Alert, Button, Input, Modal, Progress, Space, Tag } from 'antd';
import { FolderOpenOutlined, PlusOutlined } from '@ant-design/icons';
import { api } from './api';
import { inspectDatasetFolder, type FolderLayout } from './datasetLayout';
import './datasetUpload.css';

type Dataset = { id: string; name: string; kind: 'train' | 'test'; count: number; classes: string[]; group?: string };
const supported = /\.(jpe?g|png|webp|bmp)$/i;

export function DatasetUpload({ disabled = false, onUploaded }: { disabled?: boolean; onUploaded?: (group: string) => void }) {
  const [open, setOpen] = useState(false), [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState(''), [error, setError] = useState('');
  const [busy, setBusy] = useState(false), [done, setDone] = useState(0), [result, setResult] = useState<Dataset>();
  const [layout, setLayout] = useState<FolderLayout>();
  const input = useRef<HTMLInputElement>(null);
  const relative = (file: File) => file.webkitRelativePath.split('/').slice(1).join('/');
  const classes = layout?.classes || [];
  const select = (list: FileList | null) => {
    setError(''); setResult(undefined); setDone(0); setLayout(undefined);
    const next = Array.from(list || []).filter(file => supported.test(file.name));
    if (!next.length) { setFiles([]); setError('所选文件夹没有支持的图片'); return; }
    if (next.some(file => file.size > 25 * 1024 * 1024)) { setFiles([]); setError('单张图片不能超过 25 MiB'); return; }
    try { setLayout(inspectDatasetFolder(next.map(relative))); }
    catch (e) { setFiles([]); setError((e as Error).message); return; }
    if (next.length > 50000 || next.reduce((total, file) => total + file.size, 0) > 20 * 1024 ** 3) { setFiles([]); setError('每个数据集最多 50000 张图片、20 GiB'); return; }
    setFiles(next); setName(next[0].webkitRelativePath.split('/')[0]);
  };
  const upload = async () => {
    setBusy(true); setDone(0); setError('');
    try {
      const created = await api<Dataset>('datasets', { name: name.trim(), kind: 'train', layout: layout?.layout || 'classes' });
      // Sequential requests bound memory use and preserve clear per-file failures.
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const response = await fetch(`/api/platform/datasets/${created.id}/files?path=${encodeURIComponent(relative(file))}`,
          { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file });
        const value = await response.json();
        if (!response.ok) throw new Error(`${relative(file)}：${value.error?.message || '上传失败'}`);
        setDone(i + 1);
      }
      const saved = await api<Dataset>(`datasets/${created.id}/finish`, {});
      setResult(saved);
      window.dispatchEvent(new CustomEvent('datasets:changed', { detail: saved }));
      if (saved.group) onUploaded?.(saved.group);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  return <>
    <Button icon={<PlusOutlined />} disabled={disabled} onClick={() => setOpen(true)}>添加数据集</Button>
    <Modal title="添加数据集 · 自动识别目录" open={open} onCancel={() => !busy && setOpen(false)} maskClosable={!busy} closable={!busy}
      footer={result ? <Button type="primary" onClick={() => setOpen(false)}>完成</Button> : <Space><Button disabled={busy} onClick={() => setOpen(false)}>取消</Button><Button type="primary" loading={busy} disabled={!files.length || classes.length < 2 || !name.trim()} onClick={() => void upload()}>上传并添加</Button></Space>}>
      <div className="dataset-upload">
        <p>选择数据集文件夹，子文件夹名称为类别。</p>
        <div className="dataset-folder-example"><span>数据集 /</span><span>　├─ train / 类别名称 / 图片…</span><span>　└─ test 或 val / 类别名称 / 图片…</span></div>
        <input ref={node => { input.current = node; node?.setAttribute('webkitdirectory', ''); }} type="file" multiple hidden
          aria-label="选择训练数据集文件夹" onChange={event => { select(event.target.files); event.target.value = ''; }} />
        {!result && <><Button block size="large" icon={<FolderOpenOutlined />} disabled={busy} onClick={() => input.current?.click()}>选择整个数据集文件夹</Button>
          <label htmlFor="upload-dataset-name">数据集名称</label><Input id="upload-dataset-name" value={name} maxLength={120} disabled={busy} onChange={e => setName(e.target.value)} placeholder="例如：飞机分类数据集" /></>}
        {files.length > 0 && <div><strong>{files.length} 张图片 · {classes.length} 个类别</strong><div className="dataset-class-list">{classes.map(c => <Tag key={c}>{c}</Tag>)}</div></div>}
        {layout?.layout === 'split' && <Alert type="success" showIcon title={`已识别：训练 ${layout.trainCount} 张 · 测试 ${layout.testCount} 张${layout.validationCount ? ` · 验证 ${layout.validationCount} 张` : ''}`} description={`保留原始划分；${layout.testSource} 用作测试，不再随机拆分。`} />}
        <p>已有训练/测试划分保持不变；未划分时按 7:3 划分。</p>
        {busy && <Progress percent={Math.round(done / files.length * 100)} format={() => `${done}/${files.length}`} />}
        {error && <Alert type="error" showIcon title={error} />}
        {result && <Alert type="success" showIcon title="数据集已添加" />}
      </div>
    </Modal>
  </>;
}
