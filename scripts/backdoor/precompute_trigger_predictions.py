"""预先跑一遍「带触发器」的攻击/防御预测, 供前端挑图时做偏好筛选。

后门研究页面一次最多展示 20 张图, 纯随机抽样时抽到的样本可能刚好是
"攻击没打中" 或 "防御也没拦住" 的, 展示效果不稳定。本脚本对**整个测试集**
(如 MilitaryAircraft-3D 的 225 张) 跑一次:

    attack  + trigger  -> 每张图的预测类别
    defense + trigger  -> 每张图的预测类别

结果写入 <base>/testset_images/trigger_predictions.json。平台后端读取该文件后,
挑图时会优先选 "攻击命中 ∧ 防御未命中" 的样本 (即攻击成功、防御拦住的那些),
使默认展示更能体现后门攻击有效 + 防御有效。

注意: 这**不是**修改模型或造假, 只是挑选展示样本; 缓存缺失时后端照旧随机抽样。

命令行:
    python precompute_trigger_predictions.py --base exp/sabre_newdataset \
        --runs attack=sabre_vit_newdataset,defense=sabre_vit_newdataset_defense \
        --device cuda --data-root C:/path/MilitaryAircraft3D
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
    apply_trigger, forward_preds, get_class_names, prepare_server,
    _enumerate_testset, _fetch_pairs, _label_text,
)
from run_group import _apply_head, _load_head  # noqa: E402

# 与 platform_backdoor.py 保持一致
CACHE_NAME = 'trigger_predictions.json'
DEFAULT_BATCH = 32


def _run_all(server, images, trig_path, batch):
    """分批施加 trigger 并前向, 避免一次性吃满显存。"""
    preds = []
    target_label, attack_name = None, ''
    for start in range(0, images.shape[0], batch):
        chunk = images[start:start + batch]
        poisoned, target_label, attack_name = apply_trigger(server, chunk, trig_path)
        preds.extend(forward_preds(server, poisoned))
    return preds, target_label, attack_name


def run(base, runs, device='cuda', data_root=None, batch=DEFAULT_BATCH,
        output=None, verbose=True):
    base = os.path.abspath(base)
    attack_name_dir = runs.get('attack')
    if not attack_name_dir:
        raise SystemExit('[ERROR] 缺少 attack run')
    attack_dir = os.path.join(base, attack_name_dir)

    # 1) 建一次 server: 特征提取器只加载一次
    server, cfg, _ = prepare_server(attack_dir, device, data_root)
    head0 = torch.load(find_artifacts(attack_dir)[0], map_location='cpu')
    num_classes = int(head0['num_classes'])
    class_names = get_class_names(cfg, num_classes)

    # 2) 整个测试集
    pairs, domain_sizes = _enumerate_testset(server)
    if verbose:
        print(f"  测试集 {len(pairs)} 张 (各 domain: {domain_sizes})")
    images, labels, ids = _fetch_pairs(pairs)

    # 3) 逐个 run 套 head + trigger
    per_run, target_label, attack = {}, None, ''
    for key, run_name in runs.items():
        if not run_name:
            continue
        run_dir = os.path.join(base, run_name)
        head_data, trig_path = _load_head(run_dir)
        _apply_head(server, cfg, head_data)
        # 触发器属于攻击方: 防御组没有自己的 trigger artifact 时复用攻击组的
        if not trig_path:
            trig_path = find_artifacts(attack_dir)[1]
        if not trig_path:
            print(f"  [WARN] {run_name} 无 trigger artifact, 跳过")
            continue
        preds, run_target, run_attack = _run_all(server, images, trig_path, batch)
        per_run[key] = dict(run=run_name, preds=preds)
        if target_label is None:
            target_label, attack = run_target, run_attack
        if verbose:
            hits = sum(1 for t, p in zip(labels, preds) if p == target_label and t != target_label)
            print(f"  {key} ({run_name}): 触发后命中目标类 {hits}/{len(labels)}")

    if target_label is None or 'attack' not in per_run:
        raise SystemExit('[ERROR] 攻击组未找到 trigger artifact, 无法预计算')

    items = {}
    attack_preds = per_run['attack']['preds']
    defense_preds = per_run.get('defense', {}).get('preds')
    for i, image_id in enumerate(ids):
        entry = dict(label=labels[i], attack=int(attack_preds[i]))
        if defense_preds is not None:
            entry['defense'] = int(defense_preds[i])
        items[image_id] = entry

    def _asr(name):
        preds = per_run[name]['preds']
        return sum(1 for t, p in zip(labels, preds) if p == target_label and t != target_label)

    summary = dict(attackHits=_asr('attack'),
                   **({'defenseHits': _asr('defense')} if defense_preds is not None else {}),
                   total=len(ids))

    payload = dict(base=base, targetLabel=int(target_label),
                   targetName=_label_text(int(target_label), class_names),
                   attackName=attack, classNames=class_names,
                   runs={key: value['run'] for key, value in per_run.items()},
                   device=device, batch=batch, **summary,
                   items=items)

    out_path = output or os.path.join(base, 'testset_images', CACHE_NAME)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)
    if verbose:
        print(f"  -> 写入 {out_path}: {summary}")
    return payload


def main():
    parser = argparse.ArgumentParser(description='预计算全测试集触发后预测')
    parser.add_argument('--base', required=True, help='实验基目录 (含各 run 子目录)')
    parser.add_argument('--runs', default='',
                        help='run 目录映射, 如 attack=xxx,defense=yyy')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-root', default=None, help='覆盖 config.yaml 的 data.root')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    runs = dict(item.split('=', 1) for item in args.runs.split(',') if '=' in item)
    run(args.base, runs, args.device, args.data_root, args.batch, args.output)
    print('完成。')


if __name__ == '__main__':
    main()
