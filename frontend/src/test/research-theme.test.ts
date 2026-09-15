import { describe, expect, it } from 'vitest';
import { theme } from 'antd';
import { researchTheme } from '../platform/theme';

function luminance(hex: string) {
  const channels = hex.slice(1).match(/.{2}/g)!.map(part => parseInt(part, 16) / 255)
    .map(value => value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4);
  return channels[0] * .2126 + channels[1] * .7152 + channels[2] * .0722;
}
const contrast = (a: string, b: string) => (Math.max(luminance(a), luminance(b)) + .05) / (Math.min(luminance(a), luminance(b)) + .05);

describe('research console design contract', () => {
  it('uses consistent desktop controls and dark portal surfaces', () => {
    const token = theme.getDesignToken(researchTheme);
    expect(token.fontSize).toBe(14);
    expect(token.fontSizeSM).toBe(12);
    expect(token.controlHeight).toBe(40);
    expect(token.colorBgElevated).toBe('#202c36');
    expect(token.motionDurationMid).toBe('0.16s');
  });
  it('keeps normal text and primary actions readable at 14px', () => {
    const token = researchTheme.token!;
    for (const background of [token.colorBgLayout!, token.colorBgContainer!, token.colorBgElevated!]) {
      expect(contrast(token.colorText!, background)).toBeGreaterThanOrEqual(4.5);
      expect(contrast(token.colorTextSecondary!, background)).toBeGreaterThanOrEqual(4.5);
    }
    expect(contrast('#0c1a20', token.colorPrimary!)).toBeGreaterThanOrEqual(4.5);
  });
});
