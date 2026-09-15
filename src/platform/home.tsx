import { ArrowRightOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { PhotoHome } from '../design/Presentation';
import { terminal, type Job } from './api';
import { jobHref } from './navigation';
import { State } from './ui';

export function SystemHome({ jobs }: { jobs: Job[]; loading: boolean }) {
  const active = jobs.find(job => !terminal(job.status));
  return <div className="studio-home">
    <PhotoHome />
    {active && <Link className="home-active" to={jobHref(active.id)}><i className="live-dot" /><span>当前任务</span><strong>{active.request.name || active.id.slice(0, 8)}</strong><State value={active.status} /><ArrowRightOutlined /></Link>}
  </div>;
}
