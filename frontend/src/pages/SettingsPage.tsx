import { ApiOutlined, BulbOutlined, DatabaseOutlined, EyeOutlined } from '@ant-design/icons';
import { Button, Input, InputNumber, Select, Switch, Tag } from 'antd';
import { Panel } from '../components/ChartPanel';
import { PageHeader } from '../components/PageHeader';

export function SettingsPage() {
  return <div className="page settings-page">
    <PageHeader eyebrow="SYSTEM SETTINGS" title="系统与演示设置" description="管理前端数据源、刷新频率和显示偏好；当前后端保持单机模拟边界。" />
    <div className="settings-grid">
      <Panel title="数据源" subtitle="演示模式无需启动 Python 服务"><div className="settings-section"><div className="setting-item"><i><DatabaseOutlined /></i><span><b>当前数据源</b><small>使用固定种子的完整前端模拟数据</small></span><Select value="演示模式" options={[{value:'演示模式'},{value:'单机训练联调'}]} /><Tag color="success">已连接</Tag></div><div className="setting-item"><i><ApiOutlined /></i><span><b>单机 API 地址</b><small>仅在联调模式下使用</small></span><Input value="http://127.0.0.1:8000/api" readOnly /><Button>测试连接</Button></div><div className="setting-item"><i>↻</i><span><b>状态刷新间隔</b><small>SSE 不可用时的轮询间隔</small></span><InputNumber value={2000} addonAfter="ms" /></div></div></Panel>
      <Panel title="展示偏好" subtitle="设置仅保存在当前浏览器"><div className="settings-section"><div className="setting-item"><i><EyeOutlined /></i><span><b>默认显示模拟真值</b><small>仅影响后门实验的复盘视图</small></span><Switch defaultChecked /></div><div className="setting-item"><i><BulbOutlined /></i><span><b>减少动态效果</b><small>停止拓扑粒子和高频状态动画</small></span><Switch /></div><div className="setting-item"><i>▦</i><span><b>拓扑默认密度</b><small>节点较多时自动折叠域内详情</small></span><Select value="自适应" options={[{value:'自适应'},{value:'完整'},{value:'域级汇总'}]} /></div></div></Panel>
      <Panel title="安全与边界" subtitle="平台演示约束"><div className="boundary-list"><div><b>单机模拟</b><span>不创建真实分布式客户端，不控制真实网络。</span></div><div><b>数据脱敏</b><span>只使用合成数据、公开测试数据和抽象逻辑编号。</span></div><div><b>分层投影</b><span>域子服务器及两级聚合均为前端展示模拟。</span></div><div><b>模式隔离</b><span>隐私攻击与后门攻击不能在同一实验中执行。</span></div></div></Panel>
    </div>
  </div>;
}
