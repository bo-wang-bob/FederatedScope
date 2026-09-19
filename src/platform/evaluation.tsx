import { useEffect, useState } from 'react';
import { Alert, Button, Card, Collapse, Empty, Form, Input, Select, Space, Table, Tag } from 'antd';
import { ArrowRightOutlined, CheckOutlined, DownloadOutlined, LineChartOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { api, methodLabel, percent, type EvaluationResult, type Job, type Library } from './api';
import { CompareCurves, Confusion, DomainBars } from './charts';

export function EvaluationPanel({library,initialModel,initialTestset,onSelectionChange,disabled,create,open}:{
  library:Library;initialModel?:string;initialTestset?:string;onSelectionChange?:(model:string,testset?:string)=>void;disabled:boolean;create:(action:string,payload:object)=>Promise<Job>;open:(id:string)=>void;
}) {
  const [modelId,setModelId]=useState(initialModel || ''),[testsetId,setTestsetId]=useState<string>();
  const [domains,setDomains]=useState<string[]>([]),[classes,setClasses]=useState<number[]>([]);
  const [name,setName]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState('');
  const model=library.models.find(m=>m.id===modelId),testsets=library.testsets.filter(t=>t.featureSpace===model?.featureSpace);
  const test=testsets.find(t=>t.id===testsetId);
  useEffect(()=>{if(!modelId&&library.models.length)setModelId((library.models.find(m=>m.kind==='final')||library.models[0]).id);},[modelId,library.models]);
  useEffect(()=>{
    if(!model)return;
    setTestsetId(library.testsets.find(t=>t.id===initialTestset&&t.featureSpace===model.featureSpace)?.id || library.testsets.find(t=>t.id===model.jobId&&t.featureSpace===model.featureSpace)?.id || library.testsets.find(t=>t.featureSpace===model.featureSpace)?.id);
    setDomains([]);setClasses([]);
  },[model?.id]);
  const submit=async()=>{
    if(busy||disabled||!model||!test)return;
    setBusy(true);setError('');
    try{const job=await create('evaluate',{modelId,testsetId,domains,classes,name:name.trim() || methodLabel(model.method)+' · 独立评测'});open(job.id);}
    catch(e){setError((e as Error).message);}finally{setBusy(false);}
  };
  if(!library.models.length) return <div className="studio-empty-state"><LineChartOutlined/><h2>暂无可评测模型</h2><p>训练完成后可选择已保存模型。</p><a className="studio-button primary" href="/?view=train">新建训练 <ArrowRightOutlined/></a></div>;
  return <div className="evaluation-layout"><section className="evaluation-form studio-surface"><div className="wizard-title"><h2>评测配置</h2></div>
    {error&&<Alert type="error" showIcon title={error}/>}
    {initialModel&&!model&&<Alert type="warning" title="指定模型不可用，请重新选择。"/>}
    <Form layout="vertical" disabled={busy||disabled}>
      <Form.Item label="模型" htmlFor="evaluation-model"><Select id="evaluation-model" aria-label="选择已有模型" showSearch optionFilterProp="label" value={model?modelId:undefined} onChange={id=>{setModelId(id);setTestsetId(undefined);setDomains([]);setClasses([]);onSelectionChange?.(id);}} options={library.models.map(m=>({value:m.id,label:m.name+' · '+methodLabel(m.method)+' / '+m.kind}))}/></Form.Item>
      <Form.Item label="测试集" htmlFor="evaluation-testset"><Select id="evaluation-testset" aria-label="选择兼容测试集" placeholder={model?'选择兼容测试集':'先选择模型'} value={testsetId} disabled={busy||disabled||!model} onChange={id=>{setTestsetId(id);setDomains([]);setClasses([]);onSelectionChange?.(modelId,id);}} options={testsets.map(t=>({value:t.id,label:t.name+' · '+t.samples?.toLocaleString()+' 个样本'}))}/></Form.Item>
      <div className="evaluation-scope"><h3>评测范围</h3><div className="wizard-fields"><Form.Item label="测试域"><Select aria-label="测试域" placeholder="全部域" mode="multiple" value={domains} disabled={busy||disabled||!test} onChange={setDomains} maxTagCount={2} options={test?.domains.map(d=>({value:d.name,label:d.name}))}/></Form.Item>
        <Form.Item label="测试类别"><Select aria-label="测试类别" placeholder="全部类别" mode="multiple" showSearch optionFilterProp="label" value={classes} disabled={busy||disabled||!test} onChange={setClasses} maxTagCount={2} options={test?.classes.map((c,i)=>({value:i,label:c.replaceAll('_',' ')}))}/></Form.Item></div></div>
      <Collapse ghost items={[{key:'name',label:'设置评测名称',children:<Input aria-label="评测名称" value={name} maxLength={120} placeholder="自动命名" onChange={event=>setName(event.target.value)}/>}]} />
      {model?.kind==='best'&&<Alert type="warning" title="best 按训练期测试集择优；正式对比建议使用 final。"/>}
      {model?.augmentationWarning&&<Alert type="warning" title={model.augmentationWarning}/>}
      <div className="evaluation-submit"><span>加载模型后独立计算指标</span><Button type="primary" size="large" icon={<PlayCircleOutlined/>} loading={busy} disabled={disabled||busy||!model||!test} onClick={()=>void submit()}>启动独立评测</Button></div>
    </Form>
  </section><aside className="evaluation-preview"><span className="studio-kicker">评测范围</span><h2>{test ? domains.length ? domains.join(' / ') : '全域评测' : '待选择测试集'}</h2>
    <dl>{[['模型',methodLabel(model?.method)],['检查点',model?.kind || '—'],['测试样本',test?.samples?.toLocaleString() || '—'],['类别',test ? classes.length || test.classes.length : '—']].map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
    {model&&<Button type="text" onClick={()=>open(model.jobId)}>训练记录 <ArrowRightOutlined/></Button>}
  </aside></div>;
}

export function EvaluationResults({ job, library }: { job: Job; library: Library }) {
  const [domain, setDomain] = useState('overall');
  const result = job.result as EvaluationResult | undefined;
  const model = library.models.find(m => m.id === job.request.modelId);
  if (!result) return <Empty description="等待独立评测结果" />;
  const selected = domain === 'overall' ? result : result.domains[domain] || result;
  const labels = model?.classes || result.confusionMatrix.map((_, i) => String(i));
  return <><div className="platform-stats">{[['总体准确率', percent(result.accuracy)], ['Macro-F1', percent(result.macroF1)], ['分域平均', percent(result.domainMean)], ['测试样本', result.samples]].map(([name, value]) => <div className="platform-stat" key={name}><span>{name}</span><strong>{value}</strong></div>)}</div>
    {model?.augmentationWarning && <Alert type="warning" title={model.augmentationWarning} />}
    <div className="platform-grid"><Card className="platform-panel" title="分域准确率"><DomainBars domains={result.domains} /></Card>
      <Card className="platform-panel" title="分域指标"><Table rowKey="domain" pagination={false} scroll={{ x: 420 }} size="small" dataSource={Object.entries(result.domains).map(([domain, m]) => ({ domain, ...m }))} columns={[
        { title: '域', dataIndex: 'domain' }, { title: '样本', dataIndex: 'samples' }, { title: '准确率', dataIndex: 'accuracy', render: percent }, { title: 'Macro-F1', dataIndex: 'macroF1', render: percent },
      ]} /><p className="platform-muted" style={{ marginTop: 15 }}>最差域 {percent(result.worstDomain)} · 域间差距 {percent(result.domainGap)}</p></Card></div>
    <Card className="platform-panel" title="分类指标与混淆矩阵" extra={<Space><Select value={domain} onChange={setDomain} style={{ minWidth: 150 }} options={[{ value: 'overall', label: '总体' }, ...Object.keys(result.domains).map(d => ({ value: d, label: d }))]} /><Button href={`/api/platform/jobs/${job.id}/csv`}>导出 CSV</Button></Space>}>
      <div className="platform-grid"><Confusion matrix={selected.confusionMatrix} classes={labels} /><Table rowKey="classIndex" dataSource={selected.perClass} pagination={{ pageSize: 8 }} size="small" scroll={{ x: 500 }} columns={[
        { title: '类别', dataIndex: 'classIndex', render: i => `${i}: ${labels[i]}` }, { title: '样本数', dataIndex: 'support' },
        { title: 'Precision', dataIndex: 'precision', render: percent }, { title: 'Recall', dataIndex: 'recall', render: percent }, { title: 'F1', dataIndex: 'f1', render: percent },
      ]} /></div></Card>
  </>;
}

export function ComparisonPanel({jobs,open}:{jobs:Job[];open:(id:string)=>void}) {
  const [ids,setIds]=useState<string[]>([]),[loaded,setLoaded]=useState<Job[]>([]),[error,setError]=useState('');
  const [kind,setKind]=useState('train'),[retry,setRetry]=useState(0);
  const selected=loaded.length===ids.length&&loaded.every(job=>ids.includes(job.id)&&job.action===kind)?loaded:[];
  useEffect(()=>{
    let alive=true;setError('');
    void Promise.all(ids.map(id=>api<Job>('jobs/'+id))).then(data=>{if(alive)setLoaded(data);}).catch(e=>{if(alive)setError(e.message);});
    return()=>{alive=false;};
  },[ids,retry]);
  const signature=(job:Job)=>job.action==='evaluate'?JSON.stringify([job.request.testsetId,[...(job.request.domains||[])].sort(),[...(job.request.classes||[])].sort()]):
    JSON.stringify([job.request.group,job.data?.testFingerprint,job.data?.partitionFingerprint,...['rounds','localEpochs','learningRate','batchSize','sampleClients','clientCount','seed','splitSeed','alpha','samplesPerClient'].map(k=>job.request[k as keyof Job['request']])]);
  const completeProtocol=selected.every(job=>job.action==='evaluate'?!!job.request.testsetId:!!job.data?.testFingerprint&&!!job.data?.partitionFingerprint);
  const comparable=selected.length>=2&&completeProtocol&&new Set(selected.map(signature)).size===1;
  const download=()=>{
    const blob=new Blob([JSON.stringify({comparable,definition:kind==='train'?'same test/partition fingerprints and training controls; augmentation changes compute':'same testset and selected subset; verify training controls separately',jobs:selected},null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob),anchor=document.createElement('a');anchor.href=url;anchor.download='federatedscope-comparison.json';anchor.click();URL.revokeObjectURL(url);
  };
  const candidates=jobs.filter(job=>job.action===kind&&job.status==='completed');
  return <div className="comparison-workspace">
    <section className="comparison-picker studio-surface"><div><label htmlFor="compare-kind">对比内容</label><Select id="compare-kind" aria-label="对比内容" value={kind} onChange={value=>{setKind(value);setIds([]);setLoaded([]);}} options={[{value:'train',label:'训练实验'},{value:'evaluate',label:'独立评测'}]}/></div><div><label htmlFor="compare-jobs">选择实验 <small>最多 4 项</small></label><Select id="compare-jobs" aria-label="选择对比实验" mode="multiple" maxCount={4} showSearch optionFilterProp="label" placeholder="搜索并添加已完成的实验" value={ids} onChange={setIds} options={candidates.map(job=>({value:job.id,label:(job.request.name || job.id.slice(0,8))+' · '+methodLabel(job.request.method)}))}/></div><Button icon={<DownloadOutlined/>} disabled={selected.length<2} onClick={download}>导出对比</Button></section>
    {error&&<Alert type="error" title={error} action={<Button onClick={()=>setRetry(x=>x+1)}>重试</Button>}/>}
    {!ids.length?<div className="studio-empty-state comparison-empty"><LineChartOutlined/><h2>请选择对比实验</h2><p>{candidates.length?'选择至少两项已完成实验。':'暂无已完成的'+(kind==='train'?'训练':'评测')+'实验。'}</p></div>:selected.length===0&&!error?<div className="comparison-loading">正在读取实验结果…</div>:selected.length>0&&<>
      <div className="comparison-status"><span>{selected.length} 项实验</span>{selected.length<2?<small>再添加一项即可对比</small>:comparable?<span className="protocol-ok"><CheckOutlined/> 对比条件一致</span>:<span className="protocol-warning">对比条件不一致</span>}</div>
      {selected.length>=2&&!comparable&&<Alert type="warning" showIcon title="测试集、划分或训练条件不一致，不能直接比较优劣。"/>}
      {selected.some(job=>job.data?.augmentation?.warning)&&<Alert type="warning" title="包含来源未完整核验的历史增强结果，仅展示观察值。"/>}
      {kind==='train'&&<Card className="platform-panel comparison-chart" title="准确率对比" extra={<span className="platform-muted">相同坐标 · 实际轮次</span>}><CompareCurves jobs={selected}/></Card>}
      <Card className="platform-panel" title={kind==='train'?'最终训练表现':'独立评测表现'}><Table<Job> rowKey="id" dataSource={selected} pagination={false} columns={[
        {title:'实验',render:(_,job)=><Button className="platform-job-link" type="link" onClick={()=>open(job.id)}>{job.request.name || job.id.slice(0,8)}<small>{methodLabel(job.request.method)}</small></Button>},
        ...(['accuracy',...(kind==='evaluate'?['macroF1']:[]),'domainMean','worstDomain'] as const).map(metric=>({title:({accuracy:'总体准确率',macroF1:'Macro-F1',domainMean:'分域平均',worstDomain:'最差域'} as Record<string,string>)[metric],render:(_:unknown,job:Job)=>percent((kind==='evaluate'?job.result as EvaluationResult:job.metrics.at(-1))?.[metric as 'accuracy' | 'domainMean' | 'worstDomain'])})),
        {title:kind==='train'?'实际轮次':'样本数',render:(_,job)=>kind==='train'?job.metrics.at(-1)?.round??'—':(job.result as EvaluationResult)?.samples??'—'},
      ]}/></Card>
    </>}
  </div>;
}
