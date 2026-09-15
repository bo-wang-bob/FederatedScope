import type { ReactNode } from 'react';
import { Card, Tag } from 'antd';
import { statusText } from './api';
export const State = ({ value }: { value: string }) => <Tag className="studio-state" color={value === 'completed' ? 'success' : value === 'failed' ? 'error' : value === 'running' ? 'processing' : 'default'}>{statusText[value] || value}</Tag>;
export const Panel = ({ title, extra, children }: { title: string; extra?: ReactNode; children: ReactNode }) => <Card title={title} extra={extra} className="platform-panel">{children}</Card>;
export const Stat = ({ label, value, sub }: { label: string; value: string | number; sub?: string }) => <div className="platform-stat"><span>{label}</span><strong>{value}</strong>{sub && <small>{sub}</small>}</div>;
