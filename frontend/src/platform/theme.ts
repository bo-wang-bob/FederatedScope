import { theme, type ThemeConfig } from 'antd';

// Includes portal content (Select, Tooltip, Popconfirm), not just the page shell.
export const researchTheme: ThemeConfig = {
  algorithm: theme.darkAlgorithm,
  token: {
    colorPrimary: '#6ba9b0', colorInfo: '#6ba9b0', colorSuccess: '#8da883',
    colorWarning: '#c3a572', colorError: '#d78585',
    colorBgBase: '#10161c', colorBgLayout: '#10161c', colorBgContainer: '#171f27',
    colorBgElevated: '#202c36', colorText: '#e0e7eb', colorTextSecondary: '#a2b0bb',
    colorTextTertiary: '#8d9da9', colorBorder: '#34424d', colorBorderSecondary: '#2a3742',
    borderRadius: 6, fontSize: 14, fontSizeSM: 12, controlHeight: 40,
    fontFamily: '"Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif',
    motionDurationMid: '0.16s', motionDurationSlow: '0.2s',
  },
  components: {
    Button: { primaryColor: '#0c1a20', primaryShadow: 'none', defaultShadow: 'none', controlHeightLG: 40, contentFontSizeLG: 14 },
    Input: { controlHeight: 40 }, InputNumber: { controlHeight: 40 }, Select: { controlHeight: 40 },
    Card: { headerFontSize: 16, headerHeight: 56 },
    Table: { headerBg: '#1c2731', headerColor: '#acbac4', rowHoverBg: '#202e38', cellPaddingBlock: 16 },
    Tabs: { titleFontSize: 14 },
    Tooltip: { colorBgSpotlight: '#2b3d49', colorTextLightSolid: '#edf3f6' },
  },
};
