import argparse
import logging
import math
import os
import random
import sys

# 以脚本方式运行时, scripts/ (本脚本) 与仓库根目录均可 import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


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


_REPO_ROOT = _find_repo_root(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use('Agg')  # 无显示器服务器上出图
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch

from eval import (  # 复用评估脚本的构件
    GGEUREvalServer, load_config, find_artifacts)

logger = logging.getLogger("plot_preds")

# 与 _load_a3fl_test_loaders 中一致的 CLIP 归一化参数 (反归一化用)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# --------------------------------------------------------------------------- #
# 绘图 (纯 numpy 输入, 便于独立测试)
# --------------------------------------------------------------------------- #
def setup_font(font_size):
    """优先 Times New Roman, 不可用时回退 serif 并提示。"""
    preferred = ['Times New Roman', 'Times New Roman', 'Nimbus Roman',
                 'Liberation Serif']
    avail = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in preferred if f in avail), 'serif')
    matplotlib.rcParams['font.family'] = chosen
    matplotlib.rcParams['font.size'] = font_size
    if chosen != 'Times New Roman':
        print(f"  [WARN] 系统无 Times New Roman 字体, 回退使用 {chosen}")


def _label_text(idx, class_names):
    """标签索引 → 展示文本; -1 表示未标注, 索引越界时回退为数字。"""
    if idx == -1:
        return '?'
    if 0 <= idx < len(class_names):
        return str(class_names[idx])
    return str(idx)


def plot_grid(images, labels, preds, class_names, out_path, title,
              font_size=28, highlight=None):
    """绘制预测展示网格。

    Args:
        images:  list/array of (H, W, 3) float in [0,1] (已反归一化)
        labels:  list[int] 真实标签
        preds:   list[int] 预测标签
        highlight: list[bool] 与 preds 等长, 指定哪些图的 Pred 行标红;
                   None 表示全部黑色 (clean 图预测错误不标红)
    每张图下方两行: True: xxx / Pred: xxx (仅 highlight 的图 Pred 标红)
    """
    n = len(images)
    cols = 5 if n > 5 else n
    rows = math.ceil(n / cols)
    # 每格 4.5in 宽; 高度留足两行 28pt 文字 (~0.5in/行)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4.5, rows * 5.8))
    # 展平 axes: plt.subplots(rows, cols) 在 rows>1 且 cols>1 时返回 2D
    # 数组 (如 4x5), rows==1 或 cols==1 时返回 1D, 1x1 时返回单个 Axes。
    # 必须统一展平成一维, 否则 20 张图 (4x5) 时 ax.axis('off') 会在
    # numpy 行数组上调用而崩溃, 导致 triggered 图画不出来。
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)
    for ax in axes:
        ax.axis('off')

    for i, (img, t, p) in enumerate(zip(images, labels, preds)):
        ax = axes[i]
        ax.imshow(img)
        t_text = _label_text(t, class_names)
        p_text = _label_text(p, class_names)
        # 两行文字位于图片正下方; 仅 highlight 的图 Pred 行标红
        ax.text(0.5, -0.08, f"True: {t_text}", transform=ax.transAxes,
                ha='center', va='top', fontsize=font_size, color='black')
        ax.text(0.5, -0.24, f"Pred: {p_text}", transform=ax.transAxes,
                ha='center', va='top', fontsize=font_size,
                color='red' if (highlight and highlight[i]) else 'black')

    fig.suptitle(title, fontsize=font_size)
    # 下方留白给文字 (hspace 按 axes 高度比例)
    fig.subplots_adjust(hspace=0.5, top=0.96, bottom=0.02)
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    n_correct = sum(1 for t, p in zip(labels, preds) if t == p)
    print(f"  -> 图已保存 {out_path}  ({n}/{n} 张, 预测正确 {n_correct})")
    return n_correct


# --------------------------------------------------------------------------- #
# 模型与数据 (复用 eval_3x_mlp_heads 的服务构件)
# --------------------------------------------------------------------------- #
def prepare_server(run_dir, device_str, data_root=None):
    """构建评估 server 并加载训练好的 MLP head。返回 (server, cfg, trig_path)。"""
    head_path, trig_path = find_artifacts(run_dir)
    if head_path is None:
        raise FileNotFoundError(f"在 {run_dir} 未找到 *_final_mlp_head.pt")
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"在 {run_dir} 未找到 config.yaml")

    head_data = torch.load(head_path, map_location='cpu', weights_only=True)
    seed = int(head_data.get('seed', 0))
    num_classes = int(head_data['num_classes'])
    embedding_dim = int(head_data.get('embedding_dim', 0)) or None
    cfg = load_config(cfg_path)
    # 换机器时数据集路径通常不同, 允许命令行覆盖 config.yaml 里的 data.root
    if data_root:
        cfg.data.root = data_root
    cfg.seed = seed  # 用 artifact 的 seed 重建与训练一致的测试集划分
    cfg.ggeur.feature_extractor = head_data.get(
        'feature_extractor_type', 'cnn')
    if cfg.ggeur.feature_extractor == 'clip':
        # 优先沿用本次训练实际用的权重文件, 避免"本机没有打包好的公共权重"
        # 就把本来能跑的评估挡在门外; 只有两者都不存在才报错。
        configured = str(getattr(cfg.ggeur, 'clip_model_path', '') or '')
        if configured and os.path.exists(configured):
            cfg.ggeur.clip_model_path = configured
        else:
            from federatedscope.standalone_api.paths import env_path
            resources = env_path('FS_PLATFORM_RESOURCES', 'resources', _REPO_ROOT)
            weights = env_path('FS_BACKDOOR_VIT_WEIGHTS',
                env_path('FS_PLATFORM_MILITARY_VIT_WEIGHTS', resources / 'models/ViT-B-16.pt', _REPO_ROOT), _REPO_ROOT)
            if not weights.is_file():
                raise FileNotFoundError(
                    f'缺少 CLIP 权重，不联网下载: 已尝试 {weights}'
                    + (f' 与配置里的 {configured}' if configured else ''))
            cfg.ggeur.clip_model_path = str(weights)
    cfg.ggeur.mlp_hidden_dim = int(head_data.get('mlp_hidden_dim', 0))
    cfg.ggeur.mlp_dropout = float(head_data.get('mlp_dropout', 0.0))

    server = GGEUREvalServer(cfg, device=device_str)
    server.inferred_embedding_dim = embedding_dim
    server._build_global_mlp(num_classes)
    state_dict = {k: v.to(server.device)
                  for k, v in head_data['state_dict'].items()}
    server.global_mlp.load_state_dict(state_dict, strict=True)
    server.global_mlp.eval()
    server._load_feature_extractor()
    # 特征提取器切 eval
    if server.feature_extractor_type == 'cnn' and server.cnn_extractor:
        server.cnn_extractor.eval()
    elif server.feature_extractor_type == 'timm' and server.timm_extractor:
        server.timm_extractor.eval()
    elif server.feature_extractor_type == 'clip' and server.clip_model:
        server.clip_model.eval()
    return server, cfg, trig_path


def get_class_names(cfg, num_classes):
    """按数据集类型取类别名 (Office-Home 65 类 / PACS 7 类), 失败回退数字。"""
    data_type = str(cfg.data.type).lower()
    names = None
    try:
        if 'office' in data_type and 'home' in data_type:
            from federatedscope.cv.dataset.office_home import OfficeHome
            names = list(OfficeHome.CLASSES)
        elif 'pacs' in data_type:
            from federatedscope.cv.dataset.pacs import PACS
            names = list(PACS.CLASSES)
        elif 'militaryaircraft' in data_type.replace(' ', ''):
            # 类别名必须与 loader 的 class_to_idx 同序, 因此复用同一套发现逻辑
            from federatedscope.cv.dataset.domainnet import (
                discover_domainnet_metadata)
            _, names = discover_domainnet_metadata(
                str(cfg.data.root), ['aerial', 'natural', 'recon'], False)
        elif 'domainnet' in data_type:
            # 上传数据集没有固定的类别表, 类别名随 manifest 走
            from eval import load_manifest
            bundle = load_manifest(getattr(cfg.ggeur, 'domainnet_manifest_path', None),
                                   str(cfg.data.root))
            names = bundle['classes']
    except Exception as e:
        print(f"  [WARN] 类别名加载失败 ({e}), 使用数字标签")
    if names and len(names) >= num_classes:
        return names[:num_classes]
    return [str(i) for i in range(num_classes)]


def sample_test_images(server, n, sample_seed):
    """从全部 domain 的测试集里随机抽 n 张 (全局均匀抽样, 大 domain 命中更多)。

    Returns:
        images: (n, 3, 224, 224) 归一化张量
        labels: list[int]
        ids:    list[str] 形如 "Art_00123" (可溯源)
    """
    pairs, domain_sizes = _enumerate_testset(server)
    rng = random.Random(sample_seed)
    picked = rng.sample(pairs, min(n, len(pairs)))
    images, labels, ids = _fetch_pairs(picked)
    print(f"  抽样 {len(picked)}/{len(pairs)} 张 "
          f"(各 domain 测试集大小: {domain_sizes}, sample_seed={sample_seed})")
    return images, labels, ids


def _load_test_override(server, manifest_path, data_root):
    """用用户上传的测试集替换 server 的测试划分。

    manifest 由后端 (_apply_testset) 生成: 每条 record 显式带
    split='test', DomainNet 会按 manifest 过滤而不再做比例划分,
    因此上传的测试集整体进入 loader, 与训练时的内部划分无关。
    loader 的 domain 名沿用 manifest 的 domains (如 'uploaded'),
    保证 _enumerate_testset 生成的编号与导出浏览图一致。
    """
    from torchvision import transforms
    from federatedscope.cv.dataset.domainnet import DomainNet
    from eval import load_manifest
    bundle = load_manifest(manifest_path, data_root)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    from torch.utils.data import DataLoader
    loaders = {}
    for domain, records in (bundle.get('records') or {}).items():
        if not records:
            continue
        dataset = DomainNet(root=data_root, domain=domain, split='test',
                            transform=transform, classes=bundle['classes'],
                            records=records)
        loaders[domain] = DataLoader(dataset, batch_size=32, shuffle=False,
                                     num_workers=0)
    if not loaders:
        raise RuntimeError(f"测试集 manifest 无可用记录: {manifest_path}")
    server.a3fl_test_loaders = loaders
    server.a3fl_test_loaded = True


def _enumerate_testset(server):
    """加载测试集, 返回 ([(domain, ds, idx), ...], {domain: size})。

    server 上挂有 a3fl_test_override (manifest 路径, 数据根) 时,
    用用户上传的测试集整体替换训练时的内部测试划分。
    """
    override = getattr(server, 'a3fl_test_override', None)
    if override:
        _load_test_override(server, override[0], override[1])
    server._load_a3fl_test_loaders()
    if not server.a3fl_test_loaders:
        raise RuntimeError("测试集加载失败 (数据类型不支持?)")
    all_pairs, domain_sizes = [], {}
    for domain, loader in server.a3fl_test_loaders.items():
        ds = loader.dataset
        domain_sizes[domain] = len(ds)
        all_pairs += [(domain, ds, i) for i in range(len(ds))]
    return all_pairs, domain_sizes


def _fetch_pairs(pairs):
    """[(domain, ds, idx), ...] → (images, labels, ids)。"""
    images, labels, ids = [], [], []
    for domain, ds, idx in pairs:
        img, label = ds[idx]
        images.append(img)
        labels.append(int(label))
        ids.append(f"{domain}_{idx + 1:05d}")
    return torch.stack(images), labels, ids


def read_ids_file(path):
    """读取图片编号文件: 每行一个编号 (如 Art_00001), 忽略空行与 # 注释。"""
    ids = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                ids.append(s)
    if not ids:
        raise SystemExit(f"[ERROR] ids 文件为空: {path}")
    return ids


def select_images_by_ids(server, want_ids):
    """按编号精确取测试集图片 (顺序与 want_ids 一致)。

    编号格式与保存的测试集图片文件名一致: {Domain}_{五位数序号}, 如
    Art_00001 (每个 domain 内从 1 递增)。
    """
    pairs, domain_sizes = _enumerate_testset(server)
    by_id = {f"{domain}_{idx + 1:05d}": (domain, ds, idx)
             for domain, ds, idx in pairs}
    missing = [i for i in want_ids if i not in by_id]
    if missing:
        raise SystemExit(
            f"[ERROR] 以下编号在测试集中不存在: {missing}\n"
            f"  (各 domain 测试集大小: {domain_sizes}; "
            f"编号格式如 Art_00001)")
    return _fetch_pairs([by_id[i] for i in want_ids])


def save_testset_images(server, out_dir, seed):
    """把整个测试集按编号保存为 jpg + index.csv (仅第一次, 已有标记则跳过)。

    用途: 供用户浏览图片后挑编号写入 ids 文件。标记文件记录 seed,
    换 seed 重复组后测试集划分不同, 会拒绝复用旧目录。
    """
    marker = os.path.join(out_dir, '.done')
    if os.path.exists(marker):
        with open(marker, 'r', encoding='utf-8') as f:
            old = f.read().strip()
        if old != f"seed={seed}":
            raise SystemExit(
                f"[ERROR] {out_dir} 已保存过其它 seed 的测试集 ({old}), "
                f"当前 seed={seed} 划分不同。请删除该目录后重试。")
        print(f"  测试集图片已保存过 ({old}), 跳过导出")
        return
    pairs, _ = _enumerate_testset(server)
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    n_saved = 0
    with open(os.path.join(out_dir, 'index.csv'), 'w',
              encoding='utf-8') as idx_f:
        idx_f.write("id,label\n")
        for domain, ds, idx in pairs:
            img, label = ds[idx]
            img_id = f"{domain}_{idx + 1:05d}"
            disp = denormalize(img.unsqueeze(0))[0]  # (H,W,3) [0,1]
            arr = (disp * 255.0).round().astype('uint8')
            Image.fromarray(arr).save(
                os.path.join(out_dir, f"{img_id}.jpg"), quality=95)
            idx_f.write(f"{img_id},{int(label)}\n")
            n_saved += 1
    with open(marker, 'w', encoding='utf-8') as f:
        f.write(f"seed={seed}")
    print(f"  -> 测试集导出完成: {n_saved} 张 -> {out_dir} "
          f"(含 index.csv 编号-标签对照表)")


def forward_preds(server, images):
    """(n,3,224,224) 归一化图片 → 特征提取 + MLP head → list[int] 预测。"""
    with torch.no_grad():
        x = images.to(server.device)
        if server.feature_extractor_type == 'cnn':
            feats = server.cnn_extractor(x)
        elif server.feature_extractor_type == 'timm':
            feats = server.timm_extractor(x)
        else:
            feats = server.clip_model.encode_image(x)
        logits = server.global_mlp(feats.float())
        return torch.argmax(logits, dim=1).cpu().tolist()


def apply_trigger(server, images, trig_path):
    """按攻击类型对图片施加 trigger。
    a3fl 替换式: trigger*mask + images*(1-mask)
    sabre 加性:  clamp(images + trigger*mask, -3, 3)
    Returns: (poisoned_images, target_label)"""
    trig = torch.load(trig_path, map_location='cpu', weights_only=True)
    trigger = server._restore_a3fl_tensor(trig['trigger']).to(
        server.device).float()
    mask = server._restore_a3fl_tensor(trig['mask']).to(
        server.device).float()
    target_label = int(trig['target_label'])
    attack_name = str(trig.get('attack_name', ''))
    x = images.to(server.device)
    if attack_name == 'sabre':  # additive
        poisoned = torch.clamp(x + trigger * mask, -3.0, 3.0)
    else:  # a3fl replacement
        poisoned = trigger * mask + x * (1.0 - mask)
    print(f"  施加 trigger: attack={attack_name}, "
          f"target_label={target_label}, "
          f"mask 覆盖像素比例={mask.mean().item():.3f}")
    return poisoned, target_label, attack_name


def denormalize(images):
    """归一化张量 (N,3,H,W) → (N,H,W,3) [0,1] numpy, 用于展示。"""
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    disp = (images.cpu().float() * std + mean).clamp(0, 1)
    return disp.permute(0, 2, 3, 1).numpy()


def _display_name(run_name):
    """去掉目录名末尾的 seed 后缀作为展示名 (文件命名与图片标题用)。
    如 a3fl_cnn_42 → a3fl_cnn; 无后缀原名返回。"""
    import re
    return re.sub(r"_\d+$", "", run_name or "")


def plot_run(run_dir, device_str, n, sample_seed, font_size,
             want_ids=None, save_testset_dir=None, out_suffix="",
             data_root=None):
    run_name = os.path.basename(os.path.normpath(run_dir))
    # 输出文件名与图标题不带 seed 后缀 (如 a3fl_cnn_42 → a3fl_cnn)
    disp_name = _display_name(run_name)
    print(f"\n##### 绘制 {run_name} #####")
    server, cfg, trig_path = prepare_server(run_dir, device_str, data_root)
    head_path, _ = find_artifacts(run_dir)
    head_data = torch.load(head_path, map_location='cpu', weights_only=True)
    num_classes = int(head_data['num_classes'])
    class_names = get_class_names(cfg, num_classes)

    # 0) 第一次运行: 导出整个测试集图片 (按编号), 供后续挑编号用
    if save_testset_dir:
        save_testset_images(server, save_testset_dir,
                            seed=int(head_data.get('seed', 0)))

    # 1) 取图 + clean 预测 (ids 文件精确指定 > 随机抽样)
    if want_ids is not None:
        images, labels, ids = select_images_by_ids(server, want_ids)
    else:
        images, labels, ids = sample_test_images(server, n, sample_seed)
    clean_preds = forward_preds(server, images)

    # 2) clean 图: 有 trigger 的 run 标注 before trigger, 无 trigger 不加提示
    clean_path = os.path.join(
        run_dir, f"{disp_name}_pred_clean{out_suffix}.png")
    if trig_path is not None:
        clean_title = f"{disp_name}: test set (before trigger)"
    else:
        clean_title = f"{disp_name}: test set"
    # clean 图预测错误不标红 (highlight=None 全黑)
    n_correct = plot_grid(denormalize(images), labels, clean_preds,
                          class_names, clean_path, clean_title, font_size)

    # 3) triggered 图: 添加后 (仅当存在 trigger artifact, 即 a3fl/sabre)
    if trig_path is not None:
        poisoned, target_label, attack_name = apply_trigger(
            server, images, trig_path)
        poison_preds = forward_preds(server, poisoned)
        triggered_path = os.path.join(
            run_dir, f"{disp_name}_pred_triggered{out_suffix}.png")
        target_name = _label_text(target_label, class_names)
        # 只把加入 trigger 后被预测成目标类别的标红;
        # 真实标签本身就是目标类别的图片, 预测成 target 属于正常正确预测, 不标红
        highlight = [p == target_label and t != target_label
                     for p, t in zip(poison_preds, labels)]
        n_correct_p = plot_grid(
            denormalize(poisoned), labels, poison_preds, class_names,
            triggered_path,
            f"{disp_name}: test set (after trigger, "
            f"target={target_name})",
            font_size, highlight=highlight)
        # 命中统计同样排除真实标签即 target 的图片 (与 ASR 口径一致)
        n_hit = sum(1 for hl in highlight if hl)
        print(f"  triggered: 20 张中 {n_hit} 张被预测为 target "
              f"({target_name}), clean 正确率 {n_correct}/{len(ids)}, "
              f"poisoned 正确率 {n_correct_p}/{len(ids)}")
    else:
        print("  无 trigger artifact (label_flip/baseline), 只绘制 clean 图")

    del server
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed_suffix(name):
    """目录名末尾的 seed 后缀, 如 a3fl_cnn_41 → '41'; 无后缀返回 None。"""
    import re
    m = re.search(r"_(\d+)$", name or "")
    return m.group(1) if m else None


def discover_runs(base):
    """自动发现含 MLP head artifact 的 run 子目录。"""
    import glob as _glob
    runs = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        if (_glob.glob(os.path.join(d, '*_final_mlp_head.pt'))
                or _glob.glob(os.path.join(d, '*_peak_mlp_head.pt'))):
            runs.append(name)
    return runs


def pick_seed_group(runs, seed=None):
    """同一组实验的不同重复用不同 seed 后缀区分 (如 _41/_42/_43)。

    按后缀分组后只取一组 (默认取 seed=42 组, 不存在则回退数值最小的一组),
    每组内包含该 seed 的所有变体 (如 a3fl_cnn_42 + a3fl_cnn_defense_42,
    或 label_flip 的 clean/train/train_defense 三种)。
    """
    groups = {}
    for name in runs:
        key = _seed_suffix(name) or ""
        groups.setdefault(key, []).append(name)
    if not groups:
        return []
    if seed is None:
        # 默认取 seed=42 组; 不存在则回退数值最小的一组
        if "42" in groups:
            key = "42"
        else:
            key = min((k for k in groups if k), key=int, default="")
    else:
        key = str(seed)
        if key not in groups:
            raise SystemExit(
                f"[ERROR] 未找到 seed 后缀为 _{key} 的实验目录, "
                f"可用后缀: {sorted(k for k in groups if k)}")
    return groups[key]


def run_already_plotted(run_dir, run_name, out_suffix=""):
    """该 run 的预期产出图都已存在则跳过 (避免重复绘图覆盖)。

    文件名与 plot_run 一致使用去 seed 后缀的展示名 (+ 可选 out_suffix)。
    有 trigger artifact 的 run 需 clean + triggered 两张都齐;
    无 trigger 的 run 只需 clean 一张。"""
    disp_name = _display_name(run_name)
    clean_png = os.path.join(
        run_dir, f"{disp_name}_pred_clean{out_suffix}.png")
    if not os.path.exists(clean_png):
        return False
    _, trig_path = find_artifacts(run_dir)
    if trig_path is None:
        return True
    return os.path.exists(
        os.path.join(run_dir, f"{disp_name}_pred_triggered{out_suffix}.png"))


def main():
    ap = argparse.ArgumentParser(
        description="绘制测试集逐样本预测展示图 (含 trigger 前后对比)")
    ap.add_argument('--base', required=True, help='包含各 run 子目录的基目录')
    ap.add_argument('--runs', nargs='+', default=None,
                    help='run 子目录名 (指定则精确绘制这些 run, 跳过 seed 分组; 已有结果仍会跳过)')
    ap.add_argument('--seed', default=None, type=int,
                    help='挑选哪个 seed 重复组 (默认取 42, 不存在则取最小)')
    ap.add_argument('--device', default='cuda', help='cuda 或 cpu')
    ap.add_argument('--data-root', default=None,
                    help='覆盖 config.yaml 里的 data.root (换机器时数据集路径不同)')
    ap.add_argument('--n', type=int, default=20, help='抽样图片数 (默认 20)')
    ap.add_argument('--sample-seed', type=int, default=1234,
                    help='抽样随机种子 (固定后各 run 抽同一批图)')
    ap.add_argument('--ids-file', default=None,
                    help='图片编号文件 (每行一个编号如 Art_00001, 指定后取代随机抽样)')
    ap.add_argument('--save-testset-dir', default=None,
                    help='测试集图片导出目录 (默认 {base}/testset_images, '
                         '仅第一次导出, 已导出则跳过)')
    ap.add_argument('--font-size', type=int, default=28,
                    help='标签文字字号 (要求 >=28)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    if args.font_size < 28:
        print(f"[WARN] 字号 {args.font_size} < 28, 已强制提升为 28")
        args.font_size = 28
    setup_font(args.font_size)

    if not args.runs:
        discovered = discover_runs(args.base)
        if not discovered:
            print(f"在 {args.base} 下未发现含 MLP head 的 run 目录")
            return
        args.runs = pick_seed_group(discovered, args.seed)
        print(f"发现 {len(discovered)} 个 run: {discovered}")
        print(f"按 seed 分组后选取: {args.runs}")

    # ids 文件: 指定图片编号取代随机抽样; 输出文件名附加 ids 文件名后缀
    want_ids, out_suffix = None, ""
    if args.ids_file:
        want_ids = read_ids_file(args.ids_file)
        out_suffix = "_" + os.path.splitext(
            os.path.basename(args.ids_file))[0]
        print(f"使用 ids 文件 {args.ids_file}: {len(want_ids)} 张指定图片 "
              f"{want_ids[:5]}{'...' if len(want_ids) > 5 else ''}")

    # 测试集图片导出目录 (默认放在 base 下, 与 run 同级)
    save_dir = args.save_testset_dir
    if save_dir is None:
        save_dir = os.path.join(args.base, 'testset_images')

    n_skipped = 0
    for run in args.runs:
        run_dir = os.path.join(args.base, run)
        if run_already_plotted(run_dir, run, out_suffix):
            print(f"\n##### 跳过 {run} (绘图结果已存在, 不覆盖) #####")
            n_skipped += 1
            continue
        plot_run(run_dir, args.device, args.n, args.sample_seed,
                 args.font_size, want_ids=want_ids,
                 save_testset_dir=save_dir, out_suffix=out_suffix,
                 data_root=args.data_root)
    if n_skipped:
        print(f"\n共跳过 {n_skipped} 个已绘图的 run。")
    print("\n完成。")


if __name__ == '__main__':
    main()
