import { describe, expect, it } from 'vitest';
import { phases } from '../store/useAppStore';
import {
  canCentralAggregate,
  canDomainAggregate,
  isCentralLinkActive,
  isClientLinkActive,
  isUpperDirection,
} from '../utils/hierarchyProjection';

describe('三级态势投影', () => {
  it('区分下行与上行阶段', () => {
    expect(isUpperDirection('中央下发')).toBe(false);
    expect(isUpperDirection('域内广播')).toBe(false);
    expect(isUpperDirection('节点上传')).toBe(true);
    expect(isUpperDirection('全域聚合')).toBe(true);
  });

  it('只在对应阶段激活中央或域内链路', () => {
    expect(isCentralLinkActive('中央下发')).toBe(true);
    expect(isCentralLinkActive('节点处理')).toBe(false);
    expect(isClientLinkActive('域内广播')).toBe(true);
    expect(isClientLinkActive('全域聚合')).toBe(false);
    expect(phases).toHaveLength(7);
  });

  it('节点和域全部完成后才允许进入上一级聚合', () => {
    expect(canDomainAggregate(4, 5, 0)).toBe(false);
    expect(canDomainAggregate(4, 5, 1)).toBe(true);
    expect(canCentralAggregate(3, 4)).toBe(false);
    expect(canCentralAggregate(4, 4)).toBe(true);
  });
});
