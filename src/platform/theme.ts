import { theme, type ThemeConfig } from 'antd';

// Includes portal content (Select, Tooltip, Popconfirm), not just the page shell.
export const researchTheme: ThemeConfig = {
  algorithm: theme.darkAlgorithm,
  token: {
    colorPrimary: '#94cfe9', colorInfo: '#94cfe9', colorSuccess: '#8ebca7',
    colorWarning: '#c3a572', colorError: '#d78585',
    colorBgBase: '#091522', colorBgLayout: '#091522', colorBgContainer: '#101f30',
    colorBgElevated: '#15283c', colorText: '#e5eef7', colorTextSecondary: '#a6b8c9',
    colorTextTertiary: '#92a9bd', colorBorder: '#355169', colorBorderSecondary: '#294054',
    borderRadius: 6, fontSize: 14, fontSizeSM: 14, controlHeight: 40,
    fontFamily: '"Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif',
    motionDurationMid: '0.16s', motionDurationSlow: '0.2s',
  },
  components: {
    Button: { primaryColor: '#0c1a20', primaryShadow: 'none', defaultShadow: 'none', controlHeightLG: 40, contentFontSizeLG: 14 },
    Input: { controlHeight: 40 }, InputNumber: { controlHeight: 40 }, Select: { controlHeight: 40 },
    Card: { headerFontSize: 16, headerHeight: 56 },
    Table: { headerBg: '#1a3147', headerColor: '#b3c9db', rowHoverBg: '#203c52', cellPaddingBlock: 16 },
    Tabs: { titleFontSize: 14 },
    Tooltip: { colorBgSpotlight: '#29455e', colorTextLightSolid: '#edf3f6' },
  },
};
