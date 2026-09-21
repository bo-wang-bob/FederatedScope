import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { TrainingForm } from '../platform/training';
import { LAUNCH_KEY, useTrainingLaunch } from '../platform/launch';
import { DRAFT_KEY, initialDraft, validateDraft } from '../platform/draft';
import type { Catalog, Job, RequestConfig } from '../platform/api';

vi.setConfig({testTimeout:30000});
beforeAll(() => {
  window.matchMedia=vi.fn().mockImplementation(query=>({matches:false,media:query,addListener:vi.fn(),removeListener:vi.fn(),addEventListener:vi.fn(),removeEventListener:vi.fn(),dispatchEvent:vi.fn()}));
  globalThis.ResizeObserver=class{observe(){}unobserve(){}disconnect(){}};
});
beforeEach(()=>{localStorage.clear();sessionStorage.clear();});
afterEach(()=>{cleanup();vi.unstubAllGlobals();vi.restoreAllMocks();});
const defaults:RequestConfig={group:'military_vit',method:'fedavg',name:'',rounds:3,clientCount:60,sampleClients:0,batchSize:32,localEpochs:1,learningRate:.0001,seed:42,splitSeed:42,alpha:.1,gpu:1,evaluationFrequency:1,samplesPerClient:0};
const catalog:Catalog={host:'test',address:'test',protocol:'test',evaluationPolicy:'test',groups:[{id:defaults.group,dataset:'MilitaryAircraft-3D',backbone:'vit',cacheFound:true,cacheFiles:8,cacheBytes:10,domains:4,partitionLocked:false,methods:[{id:'fedavg',label:'FedAvg',enabled:true,reason:null,defaults}]}]};
const preflight={id:'a'.repeat(32),action:'inspect',status:'completed',stage:'检查通过',clients:{},request:defaults,error:null} as Job;
const training={...preflight,id:'b'.repeat(32),action:'train',status:'running'} as Job;
const response=(data:unknown)=>({ok:true,json:async()=>({data})});
function Workflow({data=catalog,open=vi.fn()}:{data?:Catalog;open?:(id:string)=>void}){const launch=useTrainingLaunch(open);return <TrainingForm catalog={data} disconnected={false} launch={launch} open={open}/>;}
function next(){fireEvent.click(screen.getByRole('button',{name:/下一步/}));}

describe('three-step training workflow',()=>{
  it('uses the selected methods API defaults without retaining another methods local steps',()=>{
    const own={...defaults,method:'heterogeneous_solution',rounds:100,localEpochs:1,
      augmentationMode:'auto' as const,generatedPerSample:20,generatedPerPrototype:20,
      targetPerClass:40,covarianceScale:.01};
    const data={...catalog,groups:[{...catalog.groups[0],methods:[
      {...catalog.groups[0].methods[0],defaults:{...defaults,localEpochs:5,rounds:100}},
      {id:own.method,label:'本架构',enabled:true,reason:null,defaults:own}]}]};
    vi.stubGlobal('fetch',vi.fn());render(<Workflow data={data}/>);
    fireEvent.click(screen.getByRole('radio',{name:/本架构/}));next();
    expect(screen.getByRole('spinbutton',{name:'本地轮数'})).toHaveValue('1');
    expect(screen.getByRole('spinbutton',{name:'通信轮数'})).toHaveValue('100');
    fireEvent.click(screen.getByRole('button',{name:/上一步/}));
    fireEvent.click(screen.getByRole('radio',{name:/FedAvg/}));next();
    expect(screen.getByRole('spinbutton',{name:'本地轮数'})).toHaveValue('5');
  });
  it('starts once, automatically checks the exact visible configuration and binds the preflight',async()=>{
    const fetch=vi.fn().mockImplementation(async(path:string)=>response(path.endsWith('preflight')?preflight:training));
    vi.stubGlobal('fetch',fetch);const open=vi.fn();render(<Workflow open={open}/>);
    expect(screen.queryByText(/GPU|缓存/)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/实验名称/),{target:{value:'页面参数测试'}});next();
    fireEvent.change(screen.getByRole('spinbutton',{name:'通信轮数'}),{target:{value:'8'}});
    fireEvent.change(screen.getByRole('spinbutton',{name:'学习率'}),{target:{value:'0.002'}});
    fireEvent.change(screen.getByRole('spinbutton',{name:'批大小'}),{target:{value:'17'}});
    next();fireEvent.click(screen.getByRole('button',{name:/启动训练/}));
    await waitFor(()=>expect(open).toHaveBeenCalledWith(training.id));
    const posted=fetch.mock.calls.map(([url,init])=>({url,body:JSON.parse(init.body)}));
    expect(posted).toHaveLength(2);
    expect(posted[0]).toMatchObject({url:'/api/platform/preflight',body:{rounds:8,learningRate:.002,batchSize:17,gpu:1,name:'页面参数测试'}});
    expect(posted[1]).toMatchObject({url:'/api/platform/train',body:{rounds:8,learningRate:.002,batchSize:17,preflightId:preflight.id}});
    expect(posted[0].body.idempotencyKey).not.toBe(posted[1].body.idempotencyKey);
    expect(sessionStorage.getItem(LAUNCH_KEY)).toBeNull();
  });
  it('supports configured generation when generated data do not exist',async()=>{
    const own={...defaults,method:'heterogeneous_solution',augmentationMode:'auto' as const,generatedPerSample:20,generatedPerPrototype:20,targetPerClass:40,covarianceScale:.01,augmentationSourceId:'',allowLegacyAugmentation:false};
    const data={...catalog,groups:[{...catalog.groups[0],methods:[{id:own.method,label:'本架构',enabled:true,reason:null,augmentedCacheFound:false,defaults:own}]}]};
    const fetch=vi.fn().mockImplementation(async(path:string)=>response(path.endsWith('preflight')?preflight:training));vi.stubGlobal('fetch',fetch);
    render(<Workflow data={data}/>);next();
    expect(screen.getByRole('spinbutton',{name:'本地轮数'})).toHaveValue('1');
    expect(screen.getByText('全部')).toBeInTheDocument();
    expect(screen.queryByRole('spinbutton',{name:'每轮参与客户端'})).not.toBeInTheDocument();
    expect(screen.getByRole('spinbutton',{name:'每个样本生成数'})).toHaveValue('20');
    expect(screen.getByRole('spinbutton',{name:'每个原型生成数'})).toHaveValue('20');
    expect(screen.getByRole('spinbutton',{name:'每类目标样本数'})).toHaveValue('40');
    expect(screen.getByRole('spinbutton',{name:'协方差缩放'})).toHaveValue('0.01');
    expect(screen.queryByLabelText('生成方式')).not.toBeInTheDocument();
    expect(screen.queryByText('更多训练参数')).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole('spinbutton',{name:'每个样本生成数'}),{target:{value:'2'}});
    fireEvent.change(screen.getByRole('spinbutton',{name:'每类目标样本数'}),{target:{value:'20'}});
    next();fireEvent.click(screen.getByRole('button',{name:/启动训练/}));
    await waitFor(()=>expect(fetch).toHaveBeenCalledTimes(2));
    expect(JSON.parse(fetch.mock.calls[1][1].body)).toMatchObject({method:'heterogeneous_solution',augmentationMode:'auto',generatedPerSample:2,targetPerClass:20,allowLegacyAugmentation:false});
  });
  it('loads a completed augmentation run as a selectable generation preset',async()=>{
    const own={...defaults,method:'heterogeneous_solution',augmentationMode:'generate' as const,generatedPerSample:1,generatedPerPrototype:1,targetPerClass:1,covarianceScale:.01,augmentationSourceId:'',allowLegacyAugmentation:false};
    const preset={...own,name:'已跑配置',rounds:100,generatedPerSample:20,generatedPerPrototype:20,targetPerClass:10};
    const data={...catalog,groups:[{...catalog.groups[0],augmentationSources:[{id:'c'.repeat(32),name:'已跑配置',request:preset}],methods:[{id:own.method,label:'本架构',enabled:true,reason:null,augmentedCacheFound:true,defaults:own}]}]};
    vi.stubGlobal('fetch',vi.fn());render(<Workflow data={data}/>);next();
    fireEvent.mouseDown(screen.getByLabelText('已验证增强配置'));
    fireEvent.click(await screen.findByText('每类 10 · 样本 20 / 原型 20 · 100 轮'));
    expect(screen.getByRole('spinbutton',{name:'每个样本生成数'})).toHaveValue('20');
    expect(screen.getByRole('spinbutton',{name:'每个原型生成数'})).toHaveValue('20');
    expect(screen.getByRole('spinbutton',{name:'每类目标样本数'})).toHaveValue('10');
    expect(screen.getByRole('spinbutton',{name:'通信轮数'})).toHaveValue('100');
  });
  it('validates dependent parameters before confirmation and never submits invalid values',()=>{
    const fetch=vi.fn();vi.stubGlobal('fetch',fetch);render(<Workflow/>);next();
    fireEvent.change(screen.getByRole('spinbutton',{name:'客户端数量'}),{target:{value:'61'}});
    next();expect(screen.getAllByText('客户端数量须为 4 的倍数').length).toBeGreaterThan(0);
    expect(screen.queryByRole('button',{name:/启动训练/})).not.toBeInTheDocument();expect(fetch).not.toHaveBeenCalled();
  });
  it('persists edited drafts across unmount and restores the matching group defaults',()=>{
    const page=render(<Workflow/>);
    fireEvent.change(screen.getByLabelText(/实验名称/),{target:{value:'保留草稿'}});next();
    fireEvent.change(screen.getByRole('spinbutton',{name:'通信轮数'}),{target:{value:'15'}});page.unmount();
    render(<Workflow/>);expect(screen.getByLabelText(/实验名称/)).toHaveValue('保留草稿');next();
    expect(screen.getByRole('spinbutton',{name:'通信轮数'})).toHaveValue('15');
    localStorage.setItem(DRAFT_KEY,JSON.stringify({version:2,request:{group:'different',method:'fedavg',name:'other'}}));
    const other={...catalog.groups[0],id:'different',methods:[{...catalog.groups[0].methods[0],defaults:{...defaults,group:'different',gpu:0,clientCount:16}}]};
    expect(initialDraft({...catalog,groups:[...catalog.groups,other]})).toMatchObject({group:'different',gpu:0,clientCount:16});
  });
  it('preserves edited architecture drafts instead of replacing server defaults',()=>{
    const own={...defaults,method:'heterogeneous_solution',localEpochs:5,augmentationMode:'generate' as const,generatedPerSample:3,generatedPerPrototype:4,targetPerClass:60,covarianceScale:.5,augmentationSourceId:'',allowLegacyAugmentation:false};
    const data={...catalog,groups:[{...catalog.groups[0],methods:[{id:own.method,label:'本架构',enabled:true,reason:null,defaults:own}]}]};
    localStorage.setItem(DRAFT_KEY,JSON.stringify({version:2,request:own}));
    expect(initialDraft(data)).toMatchObject({method:'heterogeneous_solution',localEpochs:5,generatedPerSample:3,generatedPerPrototype:4,targetPerClass:60,covarianceScale:.5});
  });
});

describe('launch transaction safety',()=>{
  it('locks duplicate clicks and leaves the exact submitted request immutable',async()=>{
    let resolve!:(v:unknown)=>void;
    const fetch=vi.fn().mockImplementationOnce(()=>new Promise(done=>{resolve=done;})).mockResolvedValue(response(training));
    vi.stubGlobal('fetch',fetch);const open=vi.fn(),hook=renderHook(()=>useTrainingLaunch(open));
    const request={...defaults,rounds:4};
    act(()=>{hook.result.current.start(request);hook.result.current.start({...request,rounds:9});});
    request.rounds=99;
    expect(fetch).toHaveBeenCalledOnce();expect(hook.result.current.busy).toBe(true);
    await act(async()=>resolve(response(preflight)));
    await waitFor(()=>expect(open).toHaveBeenCalledWith(training.id));
    expect(JSON.parse(fetch.mock.calls[1][1].body).rounds).toBe(4);
  });
  it('reconciles an uncertain train POST after refresh with the original key, without repeating the preflight',async()=>{
    const fetch=vi.fn().mockResolvedValueOnce(response(preflight)).mockRejectedValueOnce(new Error('timeout')).mockResolvedValueOnce(response(training));
    vi.stubGlobal('fetch',fetch);const first=renderHook(()=>useTrainingLaunch(vi.fn()));
    act(()=>first.result.current.start(defaults));
    await waitFor(()=>expect(first.result.current.intent?.phase).toBe('recover'));
    const oldKey=JSON.parse(fetch.mock.calls[1][1].body).idempotencyKey;
    act(()=>first.result.current.edit());expect(first.result.current.intent).toBeDefined();
    first.unmount();const open=vi.fn(),second=renderHook(()=>useTrainingLaunch(open));
    expect(second.result.current.intent?.phase).toBe('recover');
    act(()=>second.result.current.cancel());expect(second.result.current.intent?.trainRequested).toBe(true);
    act(()=>second.result.current.resume());
    await waitFor(()=>expect(open).toHaveBeenCalledWith(training.id));
    expect(fetch).toHaveBeenCalledTimes(3);
    expect(fetch.mock.calls[2][0]).toBe('/api/platform/train');
    expect(JSON.parse(fetch.mock.calls[2][1].body).idempotencyKey).toBe(oldKey);
  });
  it('allows a known failed check to be edited and never starts training',async()=>{
    const fetch=vi.fn().mockResolvedValue(response({...preflight,status:'failed',error:'data missing'}));vi.stubGlobal('fetch',fetch);
    const hook=renderHook(()=>useTrainingLaunch(vi.fn()));act(()=>hook.result.current.start(defaults));
    await waitFor(()=>expect(hook.result.current.intent?.phase).toBe('blocked'));
    expect(hook.result.current.intent?.error).toBe('data missing');expect(fetch).toHaveBeenCalledOnce();
    act(()=>hook.result.current.edit());expect(hook.result.current.intent).toBeUndefined();
  });
  it('allows editing after an explicit server rejection, but not a timeout',async()=>{
    const fetch=vi.fn().mockResolvedValueOnce(response(preflight)).mockResolvedValueOnce({ok:false,status:409,json:async()=>({error:{code:'PLATFORM_ERROR',message:'配置已改变'}})});
    vi.stubGlobal('fetch',fetch);const hook=renderHook(()=>useTrainingLaunch(vi.fn()));
    act(()=>hook.result.current.start(defaults));await waitFor(()=>expect(hook.result.current.intent?.phase).toBe('blocked'));
    act(()=>hook.result.current.edit());expect(hook.result.current.intent).toBeUndefined();
    expect(fetch).toHaveBeenCalledTimes(2);
  });
  it('can cancel an uncertain preflight using its original key without starting train',async()=>{
    const fetch=vi.fn().mockRejectedValueOnce(new Error('timeout')).mockResolvedValueOnce(response({...preflight,status:'running'})).mockResolvedValueOnce(response({...preflight,status:'stopped'}));
    vi.stubGlobal('fetch',fetch);const hook=renderHook(()=>useTrainingLaunch(vi.fn()));
    act(()=>hook.result.current.start(defaults));await waitFor(()=>expect(hook.result.current.intent?.phase).toBe('recover'));
    act(()=>hook.result.current.cancel());await waitFor(()=>expect(hook.result.current.intent).toBeUndefined());
    expect(fetch.mock.calls.map(([path])=>path)).toEqual(['/api/platform/preflight','/api/platform/preflight','/api/platform/jobs/'+preflight.id+'/stop']);
    expect(JSON.parse(fetch.mock.calls[0][1].body).idempotencyKey).toBe(JSON.parse(fetch.mock.calls[1][1].body).idempotencyKey);
  });
  it('cancels only the owned preflight and never posts train',async()=>{
    let resolve!:(v:unknown)=>void;
    const fetch=vi.fn().mockImplementationOnce(()=>new Promise(done=>{resolve=done;})).mockResolvedValue(response({...preflight,status:'stopped'}));
    vi.stubGlobal('fetch',fetch);const hook=renderHook(()=>useTrainingLaunch(vi.fn()));
    act(()=>hook.result.current.start(defaults));act(()=>hook.result.current.cancel());
    await act(async()=>resolve(response({...preflight,status:'running'})));
    await waitFor(()=>expect(hook.result.current.intent).toBeUndefined());
    expect(fetch.mock.calls.map(([path])=>path)).toEqual(['/api/platform/preflight','/api/platform/jobs/'+preflight.id+'/stop']);
  });
  it('requires legacy consent and rejects incompatible client selection',()=>{
    const own={...defaults,method:'heterogeneous_solution',augmentationMode:'reuse' as const,generatedPerSample:1,generatedPerPrototype:1,targetPerClass:0,covarianceScale:1,allowLegacyAugmentation:false};
    expect(validateDraft({...own,sampleClients:61},catalog)).toMatchObject({sampleClients:'不能超过客户端总数',allowLegacyAugmentation:'历史生成结果需要确认来源限制'});
  });
});
