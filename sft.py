"""
sft.py — Teacher / Student 有监督微调（SFT）。

从 pretrain checkpoint 出发，在 CoT 数据集上精调，强化思维链生成能力。
训练逻辑完全复用 pretrain.py，SFT 的本质差异体现在配置文件中：

  data.cot_fraction   = 1.0   只用 CoT 样本（<think> TRACE </think> 格式）
  train.base_model_path       从 pretrain 权重热启动（而非随机初始化）
  train.lr            = 1e-4  比 pretrain (1e-3) 小 10×，保留已有知识
  model.dropout       = 0.0   精调阶段关闭 dropout

数据加载用 mode='sft'：prompt（[BOS] EXPR）的 label 置 -100，
只对 target（<think>...RESULT [EOS]）计算 cross-entropy loss，
即只对思维链轨迹和最终答案优化，不拟合随机输入表达式。

流程：
    1. python data.py config/teacher_sft.yaml      生成 CoT 训练集
    2. python sft.py [config/teacher_sft.yaml]     启动 SFT

输出：
    model/teacher_sft.pt
    log/teacher_sft/metrics.jsonl
"""
import os
import sys
from pathlib import Path

CONFIG_YAML = "config/teacher_sft.yaml"


def main(config_path: str):
    # 快速校验 base model 存在，避免跑完数据加载才报错
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = cfg.get('train', {}).get('base_model_path', '')
    if base and not Path(base).exists():
        print(f"✗ base_model_path 不存在: {base}")
        print("  请先运行 pretrain.py 生成 teacher pretrain checkpoint。")
        sys.exit(1)

    # 训练逻辑完全复用 pretrain.main()
    from pretrain import main as pretrain_main
    pretrain_main(config_path)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    sys.path.insert(0, str(Path(__file__).parent))
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_YAML
    main(config_path)
