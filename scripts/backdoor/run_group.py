"""后门研究三连对比: clean / triggered / defense。

由平台后端 (federatedscope/standalone_api/platform_backdoor.py) 调用, 也可命令行执行。

与 plot_predictions.py 的区别: 后者每个 run 都要重建一次 server, 重复加载
特征提取器与测试集。本脚本只加载一次测试集与特征提取器, 再依次把各 run 的
分类器 (MLP head) 与触发器套用到同一批图片上, 一次产出:

    clean.png         攻击模型 + 干净图      (后门注入前的正常表现)
    triggered.png     攻击模型 + 带触发器图   (后门生效, 命中目标类标红)
    defense.png       防御模型 + 带触发器图   (防御后后门失效)
    defense_clean.png 防御模型 + 干净图      (确认防御不损伤正常精度)
    result.json       逐样本真实标签/三次预测 + 汇总统计

Spec (JSON):
    base       包含各 run 子目录的基目录 (如 exp/sabre)
    runs       {"attack": "sabre_vit_42", "defense": "sabre_vit_defense_42"}
    ids        图片编号列表 (至多 20 个, 形如 Art_00001)
    output     输出目录
    device     cuda / cpu
    dataRoot   可选, 覆盖 config.yaml 里的 data.root
    testManifest 可选, 用户上传测试集的 manifest (全部 split='test'),
                存在时用它整体替换训练时的内部测试划分
    testRoot   可选, 测试集 manifest 对应的数据根目录
"""
import argparse
import json
import os
import sys

# --------------------------------------------------------------------------- #
# 路径: 与 eval.py / plot_predictions.py 相同, 不能写死 '../'
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


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

import torch  # noqa: E402

from eval import find_artifacts  # noqa: E402
from plot_predictions import (  # noqa: E402
    apply_trigger, denormalize, forward_preds, get_class_names,
    plot_grid, prepare_server, setup_font, _display_name, _enumerate_testset,
    _fetch_pairs, _label_text,
)

MAX_IDS = 20


def _load_head(run_dir):
    """读取 run 目录里的 MLP head 与触发器 artifact。"""
    head_path, trig_path = find_artifacts(run_dir)
    if head_path is None:
        raise FileNotFoundError(f"在 {run_dir} 未找到 *_final_mlp_head.pt")
    return torch.load(head_path, map_location='cpu', weights_only=True), trig_path


def _apply_head(server, cfg, head_data):
    """把某个 run 的分类器装到已就绪的 server 上 (特征提取器不重载)。"""
    num_classes = int(head_data['num_classes'])
    server.inferred_embedding_dim = int(head_data.get('embedding_dim', 0)) or None
    # 不同 run 的 MLP 结构可能不同 (hidden/dropout), 必须按 head 重建
    cfg.ggeur.mlp_hidden_dim = int(head_data.get('mlp_hidden_dim', 0))
    cfg.ggeur.mlp_dropout = float(head_data.get('mlp_dropout', 0.0))
    server._build_global_mlp(num_classes)
    state = {k: v.to(server.device) for k, v in head_data['state_dict'].items()}
    server.global_mlp.load_state_dict(state, strict=True)
    server.global_mlp.eval()
    return num_classes


def _summarize(labels, preds, target_label):
    """统计准确率与攻击成功率。

    ASR 只统计"真实标签不是目标类、却被预测成目标类"的样本, 与 eval.py 口径一致。
    """
    total = len(labels)
    correct = sum(1 for t, p in zip(labels, preds) if t == p)
    hits = [p == target_label and t != target_label for t, p in zip(labels, preds)]
    eligible = sum(t != target_label for t in labels)
    return dict(correct=correct, total=total, accuracy=correct / total if total else 0.0,
                asr=sum(hits), asrEligible=eligible, asrRate=(sum(hits) / eligible) if eligible else None)


def _plot(images, labels, preds, class_names, path, title, font_size, highlight=None):
    plot_grid(denormalize(images), labels, preds, class_names, path, title, font_size,
              highlight=highlight)


def run(spec):
    base = os.path.abspath(spec['base'])
    output = os.path.abspath(spec['output'])
    runs = spec['runs']
    ids = list(spec['ids'])
    device = spec.get('device', 'cuda')
    data_root = spec.get('dataRoot')
    font_size = int(spec.get('fontSize', 28))
    os.makedirs(output, exist_ok=True)

    # ids 文件落盘, 与手工维护的 ids_txt 等价, 便于复现
    with open(os.path.join(output, 'ids.txt'), 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(ids) + '\n')

    setup_font(font_size)
    attack_name_dir = runs['attack']
    attack_dir = os.path.join(base, attack_name_dir)

    # 1) 用攻击组配置建 server: 加载特征提取器 (CLIP/CNN) 一次
    server, cfg, _ = prepare_server(attack_dir, device, data_root)
    head0 = torch.load(find_artifacts(attack_dir)[0], map_location='cpu', weights_only=True)
    num_classes = int(head0['num_classes'])
    class_names = get_class_names(cfg, num_classes)

    # 1.5) 用户上传的测试集: 整体替换训练时的内部测试划分 (见 _apply_testset)
    if spec.get('testManifest'):
        server.a3fl_test_override = (spec['testManifest'],
                                      spec.get('testRoot') or cfg.data.root)

    # 2) 加载一次测试集, 按编号取图
    pairs, domain_sizes = _enumerate_testset(server)
    by_id = {f"{domain}_{idx + 1:05d}": (domain, ds, idx) for domain, ds, idx in pairs}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"[ERROR] 以下编号在测试集中不存在: {missing} "
                         f"(各 domain 测试集大小: {domain_sizes})")
    images, labels, resolved_ids = _fetch_pairs([by_id[i] for i in ids])
    print(f"  取图 {len(resolved_ids)} 张: {resolved_ids}")

    # 3) 依次套用各 run 的分类器与触发器
    per_run, target_label, attack, poisoned_images = {}, None, '', {}
    for key, run_name in runs.items():
        run_dir = os.path.join(base, run_name)
        head_data, trig_path = _load_head(run_dir)
        _apply_head(server, cfg, head_data)
        clean_preds = forward_preds(server, images)
        entry = dict(run=run_name, display=_display_name(run_name), clean=clean_preds)
        if trig_path:
            poisoned, run_target, run_attack = apply_trigger(server, images, trig_path)
            poisoned_images[key] = poisoned
            entry.update(poisoned=forward_preds(server, poisoned),
                         targetLabel=run_target, attackName=run_attack)
            if key == 'attack':
                target_label, attack = run_target, run_attack
        per_run[key] = entry
        print(f"  {key} ({run_name}): clean 正确 "
              f"{sum(1 for t, p in zip(labels, clean_preds) if t == p)}/{len(labels)}")

    if target_label is None:
        raise SystemExit("[ERROR] 攻击组未找到 trigger artifact, 无法生成后门对比")

    target_name = _label_text(target_label, class_names)
    attack_entry, defense_entry = per_run['attack'], per_run.get('defense')

    # 4) 三张主图 + 防御模型 clean 图
    attack_display = attack_entry['display']
    clean_path = os.path.join(output, 'clean.png')
    _plot(images, labels, attack_entry['clean'], class_names, clean_path,
          f"{attack_display}: clean test images", font_size)

    triggered_path = os.path.join(output, 'triggered.png')
    _plot(poisoned_images['attack'], labels, attack_entry['poisoned'], class_names, triggered_path,
          f"{attack_display}: after trigger (target={target_name})", font_size,
          highlight=[p == target_label and t != target_label
                     for p, t in zip(attack_entry['poisoned'], labels)])

    paths = dict(clean='clean.png', triggered='triggered.png')
    if defense_entry:
        defense_display = defense_entry['display']
        defense_path = os.path.join(output, 'defense.png')
        _plot(poisoned_images['defense'], labels, defense_entry['poisoned'], class_names, defense_path,
              f"{defense_display}: after trigger (target={target_name})", font_size,
              highlight=[p == target_label and t != target_label
                         for p, t in zip(defense_entry['poisoned'], labels)])
        paths['defense'] = 'defense.png'
        _plot(images, labels, defense_entry['clean'], class_names,
              os.path.join(output, 'defense_clean.png'),
              f"{defense_display}: clean test images", font_size)
        paths['defenseClean'] = 'defense_clean.png'

    # 5) 结构化结果
    stats = dict(clean=_summarize(labels, attack_entry['clean'], target_label),
                 triggered=_summarize(labels, attack_entry['poisoned'], target_label))
    if defense_entry:
        stats['defense'] = _summarize(labels, defense_entry['poisoned'], target_label)
        stats['defenseClean'] = _summarize(labels, defense_entry['clean'], target_label)

    per_image = [dict(id=image_id, label=label, labelName=_label_text(label, class_names),
                      clean=dict(label=attack_entry['clean'][i],
                                 name=_label_text(attack_entry['clean'][i], class_names)),
                      triggered=dict(label=attack_entry['poisoned'][i],
                                     name=_label_text(attack_entry['poisoned'][i], class_names),
                                     hit=attack_entry['poisoned'][i] == target_label and label != target_label),
                      **({'defense': dict(label=defense_entry['poisoned'][i],
                                          name=_label_text(defense_entry['poisoned'][i], class_names),
                                          hit=defense_entry['poisoned'][i] == target_label and label != target_label)}
                         if defense_entry else {}))
                 for i, (image_id, label) in enumerate(zip(resolved_ids, labels))]

    result = dict(ids=resolved_ids, classNames=class_names,
                  targetLabel=target_label, targetName=target_name, attackName=attack,
                  runs={key: dict(name=entry['run'], display=entry['display'])
                        for key, entry in per_run.items()},
                  images=per_image, stats=stats, paths=paths,
                  device=device, base=base)
    with open(os.path.join(output, 'result.json'), 'w', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"  ASR: triggered {stats['triggered']['asr']}/{len(labels)}"
          + (f" -> defense {stats['defense']['asr']}/{len(labels)}" if defense_entry else ""))
    return result


def main():
    parser = argparse.ArgumentParser(description='后门研究三连对比图生成')
    parser.add_argument('--spec', required=True, help='spec.json 路径')
    args = parser.parse_args()
    with open(args.spec, 'r', encoding='utf-8') as stream:
        spec = json.load(stream)
    ids = spec.get('ids') or []
    if not ids:
        raise SystemExit('[ERROR] ids 为空')
    if len(ids) > MAX_IDS:
        raise SystemExit(f'[ERROR] 最多选择 {MAX_IDS} 张图片, 当前 {len(ids)} 张')
    run(spec)
    print('完成。')


if __name__ == '__main__':
    main()
