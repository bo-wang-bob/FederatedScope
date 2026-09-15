import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Alert, Button, Collapse, Descriptions, Dropdown, Popconfirm, Progress, Space, Table, Tabs } from 'antd';
import { ArrowRightOutlined, DownloadOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons';
import { api, methodLabel, percent, terminal, type Client, type Job, type Library } from './api';
import { Curves, Distribution, LossChart, Topology } from './charts';
import { EvaluationResults } from './evaluation';
import { PredictionPanel } from './inference';
import { modelHref } from './navigation';
import { Panel, State, Stat } from './ui';

export function JobTable({jobs,open}:{jobs:Job[];open:(id:string)=>void}) {
  return <Table<Job> rowKey="id" dataSource={jobs} pagination={{pageSize:8,hideOnSinglePage:true}} locale={{emptyText:'没有匹配的实验'}} columns={[
    {title:'实验',key:'name',render:(_,job)=><Button className="platform-job-link" type="link" onClick={()=>open(job.id)}>{job.request.name || job.id.slice(0,8)}<small>{job.request.group} · {methodLabel(job.request.method)}</small></Button>},
    {title:'类型',dataIndex:'action',render:(action:string)=>({train:'训练',inspect:'配置检查',evaluate:'评测',predict:'预测'})[action]},
    {title:'状态',dataIndex:'status',render:value=><State value={value}/>},
    {title:'创建时间',dataIndex:'createdAt',render:value=><span className="table-date">{new Date(value).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'})}</span>},
    {title:'',key:'open',width:55,render:(_,job)=><Button type="text" aria-label={'打开实验 '+(job.request.name || job.id)} icon={<ArrowRightOutlined/>} onClick={()=>open(job.id)}/>},
  ]}/>;
}
export function JobDetail({job,library,stop,rerun}:{job:Job;library:Library;stop:(id:string)=>void;rerun:()=>void}) {
  const [tab,setTab]=useState('monitor'),[logs,setLogs]=useState('');
  useEffect(()=>{
    if(tab!=='detail') return;
    let alive=true,busy=false;
    const update=async()=>{
      if(busy)return;busy=true;
      try {const value=await api<string>('jobs/'+job.id+'/logs');if(alive)setLogs(value);}
      catch(e){if(alive)setLogs('日志读取失败：'+(e as Error).message);}
      finally{busy=false;}
    };
    void update();const timer=setInterval(update,3000);
    return()=>{alive=false;clearInterval(timer);};
  },[job.id,tab]);
  const clients=Object.values(job.clients || {}),points=job.metrics || [],final=points.at(-1);
  const finished=terminal(job.status), training=job.action==='train';
  const model=library.models.find(m=>m.jobId===job.id&&m.kind==='final');
  const imageModel=model && /^(officehome|digit3|domainnet)_/.test(model.group);
  const detail=<div className="job-details-grid"><Panel title="实际参数"><Descriptions column={2} items={Object.entries(job.request).map(([key,value])=>({key,label:key,children:JSON.stringify(value)}))}/><Collapse ghost items={[{key:'versions',label:'配置与数据版本',children:<pre className="platform-code">{JSON.stringify({config:job.config,provenance:job.provenance,data:job.data},null,2)}</pre>}]} /></Panel><Panel title="运行日志"><pre className="platform-log">{logs || '尚无日志'}</pre></Panel></div>;
  return <div className="job-detail"><div className="platform-job-heading"><div><div className="job-title-line"><State value={job.status}/><span>{new Date(job.createdAt).toLocaleString('zh-CN')}</span></div><h2>{job.request.name || job.id.slice(0,8)}</h2><p>{job.request.group} <i> / </i> {methodLabel(job.request.method)}</p></div><Space>
    {!finished&&<Popconfirm title="停止当前任务？" description="只回收此任务进程，保留配置、日志和已生成文件。" onConfirm={()=>stop(job.id)}><Button danger icon={<StopOutlined/>}>停止任务</Button></Popconfirm>}
    {finished&&['train','inspect'].includes(job.action)&&<Button onClick={rerun} icon={<ReloadOutlined/>}>复用配置</Button>}
    <Dropdown trigger={['click']} menu={{items:[
      {key:'json',label:<a href={'/api/platform/jobs/'+job.id+'/export'}>结果 JSON</a>},
      ...(finished?[{key:'bundle',label:<a href={'/api/platform/jobs/'+job.id+'/bundle'}>完整复现包</a>}]:[]),
      ...(model?[{key:'model',label:<a href={'/api/platform/jobs/'+job.id+'/model-final'}>下载模型 · final</a>}]:[]),
    ]}}><Button icon={<DownloadOutlined/>}>导出</Button></Dropdown>
  </Space></div>
    {job.error&&<Alert type="error" title={job.error} showIcon/>}
    {job.data?.augmentation?.warning&&<Alert type="warning" title={job.data.augmentation.warning}/>}
    {finished&&!job.cleanup.ok&&<Alert type="error" title={job.cleanup.message} action={<Button onClick={()=>stop(job.id)}>重试本任务清理</Button>}/>}
    {model&&job.status==='completed'&&<div className="job-next-action"><div><span className="live-dot"/><strong>模型已保存</strong></div><Link className="studio-button primary small" to={modelHref(model.id,!imageModel)}>验证模型 <ArrowRightOutlined/></Link></div>}
    {training&&<><div className="job-progress"><div><span>{job.stage}</span><strong>{final?.round ?? '—'} <small>/ {job.request.rounds} 轮已评测</small></strong></div><Progress percent={final ? Math.min(100,final.round/job.request.rounds*100):0} showInfo={false} strokeColor="#83bcc3" railColor="#263640" status={job.status==='failed'?'exception':job.status==='running'?'active':'normal'} size="small"/></div>
      <div className="platform-stats"><Stat label="总体准确率" value={percent(final?.accuracy)}/><Stat label="分域平均" value={percent(final?.domainMean)}/><Stat label="最差域准确率" value={percent(final?.worstDomain)}/><Stat label="测试样本" value={job.data?.testSamples?.toLocaleString() || '—'}/></div></>}
    {job.action==='predict'&&<div className="job-prediction-result"><PredictionPanel job={job}/></div>}
    {job.action==='evaluate'&&<EvaluationResults job={job} library={library}/>}
    {training||job.action==='inspect'?<Tabs activeKey={tab} onChange={setTab} items={[
      ...(training||job.action==='inspect'?[{key:'monitor',label:'训练结果',children:<><div className="job-chart-grid"><Panel title="准确率"><Curves points={points}/></Panel><Panel title="本地训练损失"><LossChart points={points}/></Panel></div>
        <Collapse className="job-client-disclosure" items={[{key:'clients',label:'客户端协作 · '+clients.length,children:<><Topology clients={clients}/><Table<Client> size="small" rowKey="id" dataSource={clients} pagination={{pageSize:10}} columns={[{title:'客户端',dataIndex:'id'},{title:'域',dataIndex:'domain'},{title:'样本',dataIndex:'samples'},{title:'状态',dataIndex:'stage'},{title:'轮次',dataIndex:'round',render:n=>n==null?'—':n+1},{title:'训练损失',dataIndex:'loss',render:n=>n?.toFixed(4)??'—'},{title:'训练准确率',dataIndex:'accuracy',render:percent}]}/></>}]} /></>},
      {key:'data',label:'数据分布',children:<Panel title="客户端与类别"><Distribution clients={clients} classes={job.data?.classes || []}/><p className="platform-muted">原始划分；配置抽样后的实际数量保存在实验日志。</p></Panel>}]:[]),
      {key:'detail',label:'参数与日志',children:detail},
    ]}/>:<Collapse className="job-client-disclosure" onChange={keys=>setTab(keys.length?'detail':'monitor')} items={[{key:'audit',label:'参数与日志',children:detail}]}/>}
  </div>;
}
