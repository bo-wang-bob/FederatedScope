"""Public metric definitions for the standalone research console."""

from __future__ import annotations

from typing import Any, Dict


METRICS: Dict[str, Dict[str, Any]] = {
    'accuracy': {'label': '全局准确率', 'unit': 'ratio', 'modes': ['all']},
    'loss': {'label': '损失', 'unit': 'number', 'modes': ['all']},
    'worstDomain': {'label': '最弱域准确率', 'unit': 'ratio',
                    'modes': ['heterogeneity', 'privacy', 'backdoor']},
    'domainGap': {'label': '域间性能差距', 'unit': 'ratio',
                  'modes': ['heterogeneity', 'privacy', 'backdoor']},
    'privacyRisk': {'label': '隐私攻击指标', 'unit': 'ratio',
                    'modes': ['privacy']},
    'truePositiveRate': {'label': '真阳性率', 'unit': 'ratio',
                         'modes': ['backdoor', 'privacy']},
    'falsePositiveRate': {'label': '假阳性率', 'unit': 'ratio',
                          'modes': ['backdoor']},
    'attackSuccess': {'label': '攻击成功率', 'unit': 'ratio',
                      'modes': ['backdoor']},
    'noiseStd': {'label': '实际噪声标准差', 'unit': 'number',
                 'modes': ['privacy']},
    'clipBound': {'label': '实际裁剪阈值', 'unit': 'number',
                  'modes': ['privacy']},
    'clipFactor': {'label': '更新保留比例', 'unit': 'ratio',
                   'modes': ['privacy']},
    'reconstructionLoss': {'label': '重建优化损失', 'unit': 'number',
                           'modes': ['privacy']},
    'reconstructionPsnr': {'label': '重建质量', 'unit': 'dB',
                           'modes': ['privacy']},
}


def public_metric_registry() -> Dict[str, Dict[str, Any]]:
    return {key: dict(value) for key, value in METRICS.items()}
