import { useId, useRef, useState } from 'react';
import { Alert, Button, Input, Modal, Progress, Space, Tag } from 'antd';
import { FolderOpenOutlined, PictureOutlined, UploadOutlined } from '@ant-design/icons';
import { api } from './api';
import { inspectDatasetFolder } from './datasetLayout';
import { imageAccept, uploadImages } from './imageFormats';
import './datasetUpload.css';

export type UploadedDataset = { id: string; name: string; kind: 'train' | 'test'; count: number; classes: string[]; group?: string };
const relative = (file: File) => file.webkitRelativePath ? file.webkitRelativePath.split('/').slice(1).join('/') : file.name;

export function DatasetUpload({ disabled = false, kind = 'train', single = false, onUploaded, onSaved }: {
  disabled?: boolean; kind?: 'train' | 'test'; single?: boolean;
  onUploaded?: (group: string) => void; onSaved?: (dataset: UploadedDataset) => void;
}) {
  const [open, setOpen] = useState(false), [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState(''), [error, setError] = useState('');
  const [busy, setBusy] = useState(false), [done, setDone] = useState(0), [result, setResult] = useState<UploadedDataset>();
  const [classes, setClasses] = useState<string[]>([]);
  const input = useRef<HTMLInputElement>(null), createdId = useRef<string | undefined>(undefined), uploadedCount = useRef(0);
  const nameId = useId();
  const isSingle = kind === 'test' && single;
  const title = kind === 'train' ? '上传训练集' : isSingle ? '上传单张图片' : '上传测试集';
  const reset = () => { setFiles([]); setName(''); setError(''); setResult(undefined); setDone(0); setClasses([]); createdId.current = undefined; uploadedCount.current = 0; };
  const select = (list: FileList | null) => {
    reset();
    let next: File[];
    try { next = uploadImages(Array.from(list || [])); }
    catch (e) { setError((e as Error).message); return; }
    if (!next.length) { setError('未选择支持的图片'); return; }
    if (isSingle && next.length !== 1) { setError('请选择一张图片'); return; }
    if (next.some(file => file.size > 25 * 1024 * 1024)) { setError('单张图片不能超过 25 MiB'); return; }
    if (next.length > 50000 || next.reduce((total, file) => total + file.size, 0) > 20 * 1024 ** 3) { setError('每个数据集最多 50000 张图片、20 GiB'); return; }
    const paths = next.map(relative);
    try {
      if (kind === 'train') {
        if (paths.some(path => path.split('/').length !== 2)) throw new Error('请选择训练集最外层文件夹，子文件夹名称作为类别');
        setClasses(inspectDatasetFolder(paths).classes);
      } else {
        const depths = new Set(paths.map(path => path.split('/').length));
        if (depths.size !== 1 || [...depths][0] > 2) throw new Error('请选择图片文件夹；带类别与无标签图片不能混合');
        setClasses(paths[0].includes('/') ? [...new Set(paths.map(path => path.split('/')[0]))].sort() : []);
      }
    } catch (e) { setError((e as Error).message); return; }
    setFiles(next); setName(isSingle ? next[0].name : next[0].webkitRelativePath.split('/')[0]);
  };
  const upload = async () => {
    setBusy(true); setDone(uploadedCount.current); setError('');
    try {
      // Reuse the upload on retry; per-file writes and finish are idempotent.
      if (!createdId.current) createdId.current = (await api<UploadedDataset>('datasets', { name: name.trim(), kind, layout: 'classes' })).id;
      for (let i = uploadedCount.current; i < files.length; i++) {
        const file = files[i];
        const response = await fetch(`/api/platform/datasets/${createdId.current}/files?path=${encodeURIComponent(relative(file))}`,
          { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file, signal: AbortSignal.timeout(90000) });
        const value = await response.json();
        if (!response.ok) throw new Error(`${relative(file)}：${value.error?.message || '上传失败'}`);
        uploadedCount.current = i + 1; setDone(i + 1);
      }
      const saved = await api<UploadedDataset>(`datasets/${createdId.current}/finish`, {});
      setResult(saved);
      window.dispatchEvent(new CustomEvent('datasets:changed', { detail: saved }));
      if (saved.group) onUploaded?.(saved.group);
      onSaved?.(saved);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  return <>
    <Button aria-label={title} icon={isSingle ? <PictureOutlined /> : <UploadOutlined />} disabled={disabled} onClick={() => { reset(); setOpen(true); }}>{title}</Button>
    <Modal title={title} open={open} onCancel={() => !busy && setOpen(false)} maskClosable={!busy} closable={!busy}
      footer={result ? <Button aria-label="完成" type="primary" onClick={() => setOpen(false)}>完成</Button> : <Space><Button disabled={busy} onClick={() => setOpen(false)}>取消</Button><Button type="primary" loading={busy} disabled={busy || !files.length || !name.trim()} onClick={() => void upload()}>上传并添加</Button></Space>}>
      <div className="dataset-upload">
        <input ref={node => { input.current = node; if (!isSingle) node?.setAttribute('webkitdirectory', ''); }} type="file" multiple={!isSingle} accept={imageAccept} hidden
          aria-label={isSingle ? '选择测试图片' : kind === 'train' ? '选择训练数据集文件夹' : '选择测试数据集文件夹'} onChange={event => { select(event.target.files); event.target.value = ''; }} />
        {!result && <><Button block size="large" icon={isSingle ? <PictureOutlined /> : <FolderOpenOutlined />} disabled={busy} onClick={() => input.current?.click()}>{isSingle ? '选择图片' : '选择文件夹'}</Button>
          <label htmlFor={nameId}>{isSingle ? '名称' : '数据集名称'}</label><Input id={nameId} value={name} maxLength={120} disabled={busy || !!createdId.current} onChange={e => setName(e.target.value)} /></>}
        {files.length > 0 && <div><strong>{files.length} 张图片{classes.length > 0 ? ` · ${classes.length} 个类别` : ''}</strong><div className="dataset-class-list">{classes.map(c => <Tag key={c}>{c}</Tag>)}</div></div>}
        {busy && <Progress percent={Math.round(done / files.length * 100)} format={() => `${done}/${files.length}`} />}
        {error && <Alert type="error" showIcon title={error} />}
        {result && <Alert type="success" showIcon title="上传完成" />}
      </div>
    </Modal>
  </>;
}
