import argparse
import csv
import glob
import json
import logging
import os
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# 以脚本方式运行时，让仓库根目录可被 import。
# 不能写死 '../'：本脚本可能位于 scripts/ 或 scripts/backdoor/，
# 差一级就会导致 import federatedscope 掉回 site-packages 的旧版本。
def _find_repo_root(start):
    cur = os.path.abspath(start)
    for _ in range(5):
        if os.path.isdir(os.path.join(cur, 'federatedscope')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(start, '..'))


_REPO_ROOT = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federatedscope.contrib.worker.ggeur_server import GGEURServer  # noqa: E402
from federatedscope.core.auxiliaries.utils import param2tensor  # noqa: E402

# 与 _load_a3fl_test_loaders 中一致的 CLIP 归一化参数
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

logger = logging.getLogger("eval_3x")


# --------------------------------------------------------------------------- #
# 轻量评估用 Server：跳过基类 Server.__init__（会构造 aggregator/trainer/
# monitor，需要真实 model 与完整 FL 数据，评估时无需也不具备），只初始化
# 评估方法所依赖的属性，从而复用 GGEURServer 上已测试好的评估逻辑。
# --------------------------------------------------------------------------- #
class GGEUREvalServer(GGEURServer):
    def __init__(self, cfg, device='cuda'):
        # 刻意不调用 super().__init__()：基类 Server 在
        # get_aggregator(model=model) 处会因 model=None 而失败，且会触发
        # 完整 FL 装配，评估场景不需要。
        self._cfg = cfg
        self._client_num = int(getattr(cfg.federate, 'client_num', 60))
        self.state = 0
        self.device = device

        # ggeur 配置与模式标志
        self.ggeur_cfg = cfg.ggeur
        self.head_only_mode = getattr(self.ggeur_cfg, 'head_only_mode', False)
        self.feature_extractor_type = getattr(self.ggeur_cfg,
                                              'feature_extractor', 'clip')
        self.use_separated_training = getattr(
            self.ggeur_cfg, 'use_separated_training', False)
        self.freeze_classifier = getattr(
            self.ggeur_cfg, 'freeze_classifier', True)
        self.training_phase = 'classifier'
        self.pretrained_classifier = None

        # 特征提取器（由 _load_feature_extractor 懒加载）
        self.global_mlp = None
        self.cnn_extractor = None
        self.timm_extractor = None
        self.clip_model = None
        self.clip_preprocess = None
        self.inferred_embedding_dim = None

        # FedOpt（评估时禁用；_build_global_mlp 会触碰这些属性）
        self.use_fedopt = False
        self.fedopt_optimizer = None
        self.fedopt_scheduler = None
        self.fedopt_annealing = False

        # 统计缓冲（仅当 inferred_embedding_dim 为 None 时才会被
        # _get_embedding_dim 读取；这里总会设置 inferred，初始化仅为保险）
        self.local_statistics_buffer = {}

        # a3fl / sabre 评估状态
        self.a3fl_enabled = str(
            getattr(cfg.attack, 'attack_method', '')).lower() == 'a3fl'
        self.latest_a3fl_meta = None
        self.a3fl_shared_trigger = None
        self.sabre_shared_trigger = None
        self.a3fl_test_loaders = {}
        self.a3fl_test_loaded = False
        self.a3fl_clean_feature_cache = {}

        # clean 特征缓存（部分路径会引用，这里一并初始化）
        self.test_features = {}
        self.test_labels = {}
        self.test_data_loaded = False

    # ----------------------------------------------------------------- #
    # 下面两个方法在完整研究仓库里由 GGEURServer 提供，但后端原型分支的
    # ggeur_server 是裁剪版（没有攻击/后门评估那部分），因此在这里补齐，
    # 使本脚本只依赖后端分支自身即可运行，不依赖外部研究仓库。
    # 实现与 GGEURServer 中的同名方法保持一致。
    # ----------------------------------------------------------------- #
    def _load_a3fl_test_loaders(self):
        """按 domain 构建原始测试集 DataLoader，供 ASR 评估与绘图取图。"""
        if self.a3fl_test_loaded:
            return

        logger.info("Server: Loading raw test loaders for A3FL evaluation...")
        data_type = str(self._cfg.data.type).lower()
        data_root = self._cfg.data.root
        splits = tuple(getattr(self._cfg.data, 'splits', (0.7, 0.0, 0.3)))
        train_ratio, val_ratio = splits[0], splits[1]
        seed = getattr(self._cfg, 'seed', 123)

        if 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
            dataset_kwargs = {}
        elif 'pacs' in data_type:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
            from federatedscope.cv.dataset.pacs import PACS
            dataset_class = PACS
            dataset_kwargs = {}
        else:
            self.a3fl_test_loaded = True
            return

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

        for domain in domains:
            dataset = dataset_class(root=data_root,
                                    domain=domain,
                                    split='test',
                                    transform=transform,
                                    train_ratio=train_ratio,
                                    val_ratio=val_ratio,
                                    seed=seed,
                                    **dataset_kwargs)
            if len(dataset) == 0:
                continue
            self.a3fl_test_loaders[domain] = DataLoader(dataset,
                                                        batch_size=32,
                                                        shuffle=False,
                                                        num_workers=0)
        self.a3fl_test_loaded = True

    @staticmethod
    def _restore_a3fl_tensor(value):
        """把序列化后的 trigger/mask 还原成 torch.Tensor。"""
        if isinstance(value, torch.Tensor):
            return value
        try:
            restored = param2tensor(value)
        except Exception:
            restored = value
        if isinstance(restored, torch.Tensor):
            return restored
        if isinstance(restored, np.ndarray):
            return torch.from_numpy(restored)
        if isinstance(restored, list):
            return torch.tensor(restored)
        return restored


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _dict_to_ns(d):
    """递归地把 dict 转成 SimpleNamespace，使属性访问可用（getattr 等）。"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(v) for v in d]
    return d


def load_config(config_yaml_path):
    """用 yaml.safe_load 构建轻量 cfg（绕开 yacs 注册/冻结约束）。
    保存的 config.yaml 是完整解析后的配置，所有键齐全。"""
    with open(config_yaml_path, 'r') as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


def find_artifacts(run_dir):
    """返回 (head_path, trigger_path)。优先 peak（峰值后门，最后一轮攻击
    的 MLP head），回退 final，再回退旧的无后缀 trigger。

    head 与 trigger 取自同一 tier：ASR 必须在峰值后门上测量——final 轮
    模型在攻击停止后的干净轮次（round 81-100）里会被持续洗掉 trigger->target
    映射，从而低估 ASR。"""
    # Tier 1: peak (last-attack-round snapshot)
    head = glob.glob(os.path.join(run_dir, '*_peak_mlp_head.pt'))
    trig = (glob.glob(os.path.join(run_dir, '*_a3fl_peak_trigger.pt')) +
            glob.glob(os.path.join(run_dir, '*_sabre_peak_trigger.pt')))
    if head:
        return head[0], (trig[0] if trig else None)
    # Tier 2: final (end-of-training)
    head = glob.glob(os.path.join(run_dir, '*_final_mlp_head.pt'))
    trig = (glob.glob(os.path.join(run_dir, '*_a3fl_final_trigger.pt')) +
            glob.glob(os.path.join(run_dir, '*_sabre_final_trigger.pt')))
    if head:
        return head[0], (trig[0] if trig else None)
    # Tier 3: legacy (older unsuffixed trigger saves)
    head = glob.glob(os.path.join(run_dir, '*_final_mlp_head.pt'))
    trig = (glob.glob(os.path.join(run_dir, '*_a3fl_trigger.pt')) +
            glob.glob(os.path.join(run_dir, '*_sabre_trigger.pt')))
    return (head[0] if head else None,
            (trig[0] if trig else None))


def _extract_clean_features(server):
    """构建 test loaders 并提取 clean 特征，写入 a3fl_clean_feature_cache。
    供 clean accuracy 计算与 ASR 评估（复用缓存、仅跑中毒前向）共用。"""
    server._load_a3fl_test_loaders()
    server._load_feature_extractor()
    server.global_mlp.eval()
    if server.feature_extractor_type == 'cnn' and server.cnn_extractor is not None:
        server.cnn_extractor.eval()
    elif server.feature_extractor_type == 'timm' and server.timm_extractor is not None:
        server.timm_extractor.eval()
    elif server.feature_extractor_type == 'clip' and server.clip_model is not None:
        server.clip_model.eval()

    with torch.no_grad():
        for domain, loader in server.a3fl_test_loaders.items():
            if server.a3fl_clean_feature_cache.get(domain) is not None:
                continue
            feats, labs = [], []
            for images, labels in loader:
                images = images.to(server.device)
                if server.feature_extractor_type == 'cnn':
                    cf = server.cnn_extractor(images)
                elif server.feature_extractor_type == 'timm':
                    cf = server.timm_extractor(images)
                else:
                    cf = server.clip_model.encode_image(images)
                feats.append(cf.cpu())
                labs.append(labels)
            server.a3fl_clean_feature_cache[domain] = (
                torch.cat(feats), torch.cat(labs))
            logger.info(
                f"  domain={domain}: cached {torch.cat(feats).shape[0]} "
                f"clean test features")


def evaluate_run(run_dir, device_str):
    head_path, trig_path = find_artifacts(run_dir)
    if head_path is None:
        raise FileNotFoundError(
            f"在 {run_dir} 未找到 *_final_mlp_head.pt")
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"在 {run_dir} 未找到 config.yaml")

    # artifact 元数据是权威来源
    head_data = torch.load(head_path, map_location='cpu', weights_only=True)
    seed = int(head_data.get('seed', 0))
    num_classes = int(head_data['num_classes'])
    embedding_dim = int(head_data.get('embedding_dim', 0)) or None
    mlp_hidden_dim = int(head_data.get('mlp_hidden_dim', 0))
    mlp_dropout = float(head_data.get('mlp_dropout', 0.0))
    feat_type = head_data.get('feature_extractor_type', 'cnn')

    # 从保存的 config.yaml 构建 cfg；强制 seed 用 artifact 的值，
    # 确保重建的 test set 与该模型训练时的数据划分一致
    cfg = load_config(cfg_path)
    cfg.seed = seed
    cfg.ggeur.feature_extractor = feat_type
    cfg.ggeur.mlp_hidden_dim = mlp_hidden_dim
    cfg.ggeur.mlp_dropout = mlp_dropout

    server = GGEUREvalServer(cfg, device=device_str)
    # embedding_dim 用 artifact 里的权威值（_get_embedding_dim 优先取它）
    server.inferred_embedding_dim = embedding_dim

    # 构建 + 加载训练好的 MLP head（只含训练部分，CLIP/CNN backbone 不存）
    server._build_global_mlp(num_classes)
    state_dict = {k: v.to(server.device)
                  for k, v in head_data['state_dict'].items()}
    server.global_mlp.load_state_dict(state_dict, strict=True)
    server.global_mlp.eval()

    results = {'seed': seed, 'run_dir': run_dir,
               'artifact_kind': head_data.get('artifact_kind', 'final'),
               'head_round': int(head_data.get('final_round', -1)),
               'clean_acc': {}, 'asr': None}

    # 1) 提取 clean 特征并缓存（clean acc 与 ASR 共用）
    _extract_clean_features(server)

    # 2) ASR（仅当保存了 trigger）：a3fl 用 replacement，sabre 用 additive
    if trig_path is not None:
        trig = torch.load(trig_path, map_location='cpu', weights_only=True)
        attack_name = trig.get('attack_name', '')
        additive = (attack_name == 'sabre')
        trigger_meta = {
            'trigger': trig['trigger'],
            'mask': trig['mask'],
            'target_label': int(trig['target_label']),
        }
        # 该方法内部会复用 a3fl_clean_feature_cache 的 clean 特征，
        # 仅对中毒图像做特征提取前向
        asr = server._evaluate_trigger_target_rate(
            trigger_meta, structured=True, additive=additive)
        results['asr'] = asr

    # 3) clean accuracy：用缓存的特征算预测 vs 真标签
    #    同时收集逐样本预测明细，写入 CSV（图片编号/真实标签/预测标签/预测成功）
    pred_rows = []
    with torch.no_grad():
        for domain, (cf, labels) in server.a3fl_clean_feature_cache.items():
            cf = cf.to(server.device).float()
            labels = labels.to(server.device).long()
            preds = torch.argmax(server.global_mlp(cf), dim=1)
            results['clean_acc'][domain] = (
                (preds == labels).float().mean().item())
            labels_list = labels.cpu().tolist()
            preds_list = preds.cpu().tolist()
            for i, (t, p) in enumerate(zip(labels_list, preds_list), start=1):
                # 图片编号带 domain 前缀，保证跨 domain 唯一且可溯源
                pred_rows.append((f"{domain}_{i:05d}", t, p,
                                  1 if t == p else 0))
    if results['clean_acc']:
        vals = list(results['clean_acc'].values())
        results['clean_acc']['average'] = sum(vals) / len(vals)

    # 3b) 逐样本预测明细 CSV：每张测试图片的预测是否正确
    if pred_rows:
        csv_path = os.path.join(run_dir, 'eval_predictions.csv')
        try:
            # utf-8-sig 带 BOM，保证 Excel 直接打开不乱码
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['图片编号', '真实标签', '预测标签', '预测成功'])
                w.writerows(pred_rows)
            n_correct = sum(r[3] for r in pred_rows)
            print(f"  -> 预测明细写入 {csv_path} "
                  f"({len(pred_rows)} 行, 正确 {n_correct}, "
                  f"错误 {len(pred_rows) - n_correct})")
        except Exception as e:
            print(f"  [WARN] 写入预测明细 CSV 失败: {e}")

    # 写入与 batch_parse_exp.py 兼容的 per-run eval_stats.json，供后续统计
    # 脚本合并 ASR/clean_acc（日志无测试环节；FPR/recall 等仍从日志解析）
    has_atk = _has_attack(cfg)
    is_def = _is_defense_run(cfg)
    # 三类分组：baseline（无攻击基准） / no_defense（有攻击无防御） / with_defense（有攻击有防御）
    if not has_atk:
        group = 'baseline'
    elif is_def:
        group = 'with_defense'
    else:
        group = 'no_defense'
    results['has_attack'] = has_atk
    results['is_defense'] = is_def
    results['group'] = group
    try:
        stats = _build_log_stats(run_dir, cfg, results)
        stats['group'] = group
        stats_path = os.path.join(run_dir, 'eval_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  -> stats 写入 {stats_path}  (group={group})")
    except Exception as e:
        import traceback
        print(f"  [WARN] 写入 stats 文件失败: {e}")
        traceback.print_exc()

    # 释放显存
    del server
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _mean_std(xs):
    if not xs:
        return None
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {'mean': m, 'std': s, 'n': len(xs),
            'values': [round(v, 6) for v in xs]}


def _is_defense_run(cfg):
    """根据 config.yaml 判断是否为有防御实验。
    defense_method 非空、或 multi_metrics_stats_enabled / defense 任一为 True。"""
    defense_method = str(getattr(cfg.ggeur, 'defense_method', '') or '')
    if defense_method:
        return True
    if bool(getattr(cfg.ggeur, 'multi_metrics_stats_enabled', False)):
        return True
    if bool(getattr(cfg.ggeur, 'multi_metrics_stats_defense', False)):
        return True
    return False


def _has_attack(cfg):
    """根据 config.yaml 判断是否启用了攻击方法。
    无攻击基准组（label_flip 的 baseline）attack_method 为空字符串。"""
    method = str(getattr(cfg.attack, 'attack_method', '') or '')
    return bool(method)


def _coerce_attacker_ids(raw):
    """把 config.yaml 里 attacker_id 的各种可能写法都归一化为 [int, ...]。

    支持的形式:
      - None / '' / False / 未设置 → []
      - [1, 2, 3] / (1, 2) / range(3) → 转 int 列表
      - 单个 int (1 / 12) → 包装为 [1] / [12]
      - 逗号分隔字符串 "1,2,3" → [1,2,3]
    label_flip 无攻击基准组里 attacker_id 可能不是列表，必须容错。
    """
    if raw is None:
        return []
    if isinstance(raw, bool):
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            return [int(x.strip()) for x in raw.split(',') if x.strip()]
        except ValueError:
            return []
    try:
        return sorted(int(x) for x in raw)
    except (TypeError, ValueError):
        return []


def _build_log_stats(run_dir, cfg, res):
    """构建与 parse_exp_log.py 输出格式兼容的 per-run stats dict，供
    batch_parse_exp.py 直接读取（日志已无测试环节，ASR/clean_acc 来自
    eval），攻击/防御配置来自 config.yaml。"""
    attack_method = str(getattr(cfg.attack, 'attack_method', '') or '')
    expname = str(getattr(cfg, 'expname', '') or os.path.basename(run_dir))
    defense_method = str(getattr(cfg.ggeur, 'defense_method', '') or '')
    stats_enabled = bool(getattr(cfg.ggeur, 'multi_metrics_stats_enabled', False))
    stats_defense = bool(getattr(cfg.ggeur, 'multi_metrics_stats_defense', False))

    # 攻击起始/结束轮
    start_round = -1
    poison_epochs = -1
    if attack_method:
        attack_block = getattr(cfg.attack, attack_method, None)
        if attack_block is not None:
            try:
                start_round = int(getattr(attack_block, 'start_round', -1) or -1)
            except (TypeError, ValueError):
                start_round = -1
            try:
                poison_epochs = int(getattr(attack_block, 'poison_epochs', -1) or -1)
            except (TypeError, ValueError):
                poison_epochs = -1

    attack_end_round = None
    if start_round > 0 and poison_epochs > 0:
        attack_end_round = start_round + poison_epochs - 1

    attacker_ids_raw = getattr(cfg.attack, 'attacker_id', []) or []
    attacker_ids = _coerce_attacker_ids(attacker_ids_raw)
    attacker_count = len(attacker_ids)
    client_num = int(getattr(cfg.federate, 'client_num', 60))
    benign_client_count = client_num - attacker_count

    total_rounds = int(getattr(cfg.federate, 'total_round_num', 0))

    final_clean_acc = res['clean_acc'].get('average')
    avg_asr = None
    if res.get('asr'):
        avg_asr = res['asr'].get('asr', {}).get('average')

    head_round = int(res.get('head_round', -1))

    stats = {
        'attack_method': attack_method,
        'expname': expname,
        'log_file': '',  # 无日志文件
        'defense_method': defense_method,
        'stats_defense_enabled': stats_enabled,
        'stats_defense_active': stats_defense,
        'total_rounds': total_rounds,
        'runtime_minutes': None,
        'runtime': None,
        'attack_start_round': start_round,
        'attack_end_round': attack_end_round,
        'attack_poison_epochs': poison_epochs if poison_epochs > 0 else None,
        'attacker_ids': attacker_ids,
        'attacker_count': attacker_count,
        'benign_client_count': benign_client_count,
        'avg_acc_overall': final_clean_acc,  # eval 只给最终轮值
        'final_clean_acc': final_clean_acc,
        'avg_asr': avg_asr,
        'asr_count': 1 if avg_asr is not None else 0,
        'asr_rounds': [{'round': head_round, 'asr': avg_asr}]
                       if avg_asr is not None else [],
        'acc_rounds_sample': [{'round': head_round, 'acc': final_clean_acc}]
                             if final_clean_acc is not None else [],
        'fpr': None,
        'tp_total': None,
        'fp_total': None,
        'fn_total': None,
        'recall': None,
        'precision': None,
        'defense_attack_rounds': 0,
        'defense_per_round': [],
        # eval 专属标记，供 batch_parse_exp.py 识别为 eval 产出并直接读取
        'stats_source': 'eval_3x',
        'eval_run_dir': run_dir,
        'eval_seed': res.get('seed'),
        'eval_artifact_kind': res.get('artifact_kind'),
        'eval_head_round': head_round,
        'eval_per_domain_clean_acc': {k: v for k, v in res['clean_acc'].items()
                                      if k != 'average'},
    }
    if res.get('asr'):
        stats['eval_per_domain_asr'] = {
            k: v for k, v in res['asr'].get('asr', {}).items() if k != 'average'}
    return stats


def summarize(runs_results):
    clean_avgs = [r['clean_acc']['average'] for r in runs_results
                  if r['clean_acc'].get('average') is not None]
    asr_avgs = []
    for r in runs_results:
        a = r.get('asr')
        if a and a.get('asr', {}).get('average') is not None:
            asr_avgs.append(a['asr']['average'])
    # 各 domain 的 clean acc 也汇总
    per_domain_clean = {}
    for r in runs_results:
        for dom, v in r['clean_acc'].items():
            if dom == 'average':
                continue
            per_domain_clean.setdefault(dom, []).append(v)
    per_domain_asr = {}
    for r in runs_results:
        a = r.get('asr')
        if not a:
            continue
        for dom, v in a.get('asr', {}).items():
            per_domain_asr.setdefault(dom, []).append(v)
    return {
        'clean_acc_avg': _mean_std(clean_avgs),
        'asr_avg': _mean_std(asr_avgs),
        'per_domain_clean_acc': {d: _mean_std(v)
                                 for d, v in per_domain_clean.items()},
        'per_domain_asr': {d: _mean_std(v)
                           for d, v in per_domain_asr.items() if d != 'average'},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True,
                    help='包含各 run 子目录的基目录')
    ap.add_argument('--runs', nargs='+', default=None,
                    help='run 子目录名（省略则自动发现）')
    ap.add_argument('--device', default='cuda', help='cuda 或 cpu')
    ap.add_argument('--out', default=None, help='输出 json 路径')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    # 自动发现 run 子目录
    if not args.runs:
        args.runs = []
        for name in sorted(os.listdir(args.base)):
            d = os.path.join(args.base, name)
            if os.path.isdir(d) and glob.glob(
                    os.path.join(d, '*_final_mlp_head.pt')):
                args.runs.append(name)
        if not args.runs:
            print(f"在 {args.base} 下未发现含 *_final_mlp_head.pt 的 run 目录")
            return

    runs_results = []
    for run in args.runs:
        run_dir = os.path.join(args.base, run)
        print(f"\n##### 评估 {run} #####")
        res = evaluate_run(run_dir, args.device)
        runs_results.append(res)
        print(f"  seed={res['seed']}  artifact={res.get('artifact_kind')}  "
              f"head_round={res.get('head_round')}")
        ca = res['clean_acc']
        print(f"  clean_acc: " + ", ".join(
            f"{k}={v:.4f}" for k, v in ca.items() if k != 'average')
              + f"  | avg={ca.get('average', float('nan')):.4f}")
        if res.get('asr'):
            a = res['asr'].get('asr', {})
            print(f"  ASR: " + ", ".join(
                f"{k}={v:.4f}" for k, v in a.items() if k != 'average')
                  + f"  | avg={a.get('average', float('nan')):.4f}")
        else:
            print("  ASR: 无 trigger，跳过")

    # 一组实验包含 3 组（label_flip）：baseline(无攻击基准) / no_defense(有攻击无防御)
    # / with_defense(有攻击有防御)；a3fl/sabre 只有 no_defense + with_defense。
    # 统一按 runs_results['group'] 分组
    baseline = [r for r in runs_results if r.get('group') == 'baseline']
    no_defense = [r for r in runs_results if r.get('group') == 'no_defense']
    with_defense = [r for r in runs_results if r.get('group') == 'with_defense']
    summary = {
        'all': summarize(runs_results),
        'baseline': summarize(baseline),
        'no_defense': summarize(no_defense),
        'with_defense': summarize(with_defense),
    }

    print("\n========== 汇总 (mean +/- std) ==========")
    for label, group, s in [
        ('全部', runs_results, summary['all']),
        ('无攻击基准', baseline, summary['baseline']),
        ('有攻击·无防御', no_defense, summary['no_defense']),
        ('有攻击·有防御', with_defense, summary['with_defense']),
    ]:
        print(f"\n--- {label} (n={len(group)}) ---")
        if not group:
            print("  (无实验)")
            continue
        if s['clean_acc_avg']:
            c = s['clean_acc_avg']
            print(f"  Clean accuracy: {c['mean']:.4f} +/- {c['std']:.4f} "
                  f"(n={c['n']})")
        if s['asr_avg']:
            a = s['asr_avg']
            print(f"  ASR: {a['mean']:.4f} +/- {a['std']:.4f} (n={a['n']})")
        if s['per_domain_clean_acc']:
            print("  Per-domain clean accuracy:")
            for d, st in s['per_domain_clean_acc'].items():
                print(f"    {d:12s}: {st['mean']:.4f} +/- {st['std']:.4f}")
        if s['per_domain_asr']:
            print("  Per-domain ASR:")
            for d, st in s['per_domain_asr'].items():
                print(f"    {d:12s}: {st['mean']:.4f} +/- {st['std']:.4f}")

    out = {'runs': runs_results, 'summary': summary}
    out_path = args.out or os.path.join(args.base, 'eval_3x_summary.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n汇总已保存到 {out_path}")


if __name__ == '__main__':
    main()
