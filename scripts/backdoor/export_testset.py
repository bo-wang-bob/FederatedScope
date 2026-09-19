"""导出某个 run 的测试集图片 + index.csv, 供平台"挑图"浏览。

训练跑到一半换过数据集时, 旧的 testset_images 属于上一个数据集; 由调用方
(platform_backdoor_training) 负责在换组时删掉目录, 这里只负责导出。
"""
import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from plot_predictions import prepare_server, save_testset_images  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='导出后门研究的测试集图片')
    parser.add_argument('--run', required=True, help='run 目录 (含 config.yaml 与 head artifact)')
    parser.add_argument('--out', required=True, help='输出目录 (<base>/testset_images)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-root', dest='data_root', default=None,
                        help='覆盖 config.yaml 里的 data.root')
    args = parser.parse_args()

    server, _cfg, _trig = prepare_server(args.run, args.device, args.data_root)
    head_path = None
    from eval import find_artifacts
    head_path, _ = find_artifacts(args.run)
    import torch
    head = torch.load(head_path, map_location='cpu', weights_only=True)
    save_testset_images(server, args.out, seed=int(head.get('seed', 0)))

    # 顺便记一份元信息, 便于前端展示
    meta = dict(run=os.path.basename(os.path.normpath(args.run)),
                seed=int(head.get('seed', 0)),
                numClasses=int(head.get('num_classes', 0)),
                dataRoot=str(_cfg.data.root))
    with open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8') as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)
    print('完成。')


if __name__ == '__main__':
    main()
