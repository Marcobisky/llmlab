# LLMlab 全流程计划方案 (v7)

## 0. 目标

在一个完全可控的合成环境里走完 **预训练 → 后训练** 全流程，使笔记中每个概念都可观测、可度量：off-policy vs on-policy、exposure bias、teacher unreliability、hard label vs soft label、verifiable reward (RLVR)、REINFORCE、baseline、importance sampling + clip、forward/reverse KL、chain-of-thought。

核心结构：
- **解释器** = ground truth（真正的"上帝"），零成本生成无限数据，完美验证任何输出。
- **teacher 模型 $\Pi_T$** = 较大的 Transformer，走完整 pipeline（pretrain → SFT → GRPO/SDPO）训成强 teacher，为 KD/OPD 提供有熵的 soft 分布。**只在浅组合（depth ≤ 3）上训练，故在深组合上不完美。**
- **student 模型 $\pi_\theta$** = 较小的 Transformer，pretrain → SFT 作为公共热身，再分三条支路（KD / OPD / GRPO）对比。

---

## 1. 形式语言定义

### 1.1 设计原则

- 输出符号**全部来自输入**（置换 / 选取 / 复制），无"计算"出的新值（区别于加减乘除）。
- 确定性：给定表达式，正确答案唯一。
- 可组合：算子可链式嵌套，组合深度 = 推理步数。
- 可验证：一行解释器即可判定对错 → 天然 RLVR。

### 1.2 词表（V = 32，启用 CoT 时 V = 34）

| 范围 | token | 数量 |
|---|---|---|
| 数字 | `0 1 2 3 4 5 6 7 8 9` | 10 |
| 一元算子 | `i r s S d D h t e o L R p` | 13 |
| 二元算子 | `c z` | 2 |
| 特殊符号 | `= ?` | 2 |
| 判定 | `vc vw gw`（原子 token，仅出现在 `?` 之后） | 3 |
| 控制 | `[BOS] [EOS]` | 2 |
| （可选）CoT | `<think> </think>` | 2 |

### 1.3 运算目录

#### 置换类（输出是输入的重排，多重集不变）

| 算子 | 名称 | 语义 | 示例 |
|---|---|---|---|
| `i` | identity | 恒等 | `i1234=1234` |
| `r` | reverse | 反转 | `r1234=4321` |
| `s` | sort asc | 升序排列 | `s3142=1234` |
| `S` | sort desc | 降序排列 | `S3142=4321` |
| `L` | rotate left | 左旋一位 | `L1234=2341` |
| `R` | rotate right | 右旋一位 | `R1234=4123` |

#### 选择类（输出是输入的子序列）

| 算子 | 名称 | 语义 | 示例 |
|---|---|---|---|
| `d` | dedup | 折叠相邻重复 | `d11223=123` |
| `h` | head | 前 ⌊n/2⌋ 个 | `h12345=12` |
| `t` | tail | 后 ⌈n/2⌉ 个 | `t12345=345` |
| `e` | even | 偶数位 (0,2,4…) | `e12345=135` |
| `o` | odd | 奇数位 (1,3,5…) | `o12345=24` |

#### 扩张类（输出重复输入符号）

| 算子 | 名称 | 语义 | 示例 |
|---|---|---|---|
| `D` | double | 每符号翻倍 | `D123=112233` |
| `p` | palindrome | 接上自身反转 | `p123=123321` |

#### 二元类（两个操作数）

| 算子 | 名称 | 语义 | 示例 |
|---|---|---|---|
| `c` | concat | 拼接 | `12c34=1234` |
| `z` | zip | 交错（不等长则余部直接接上） | `12z34=1324`，`12z345=13245` |

### 1.4 形式语法（BNF）

```
STMT     := EXPR '=' RESULT [EOS]
           | EXPR '=' CANDIDATE '?' VERDICT [EOS]
           | EXPR '<think>' TRACE '</think>' '=' RESULT [EOS]

EXPR     := UOP EXPR
           | DIGITS BOP EXPR
           | DIGITS

TRACE    := EXPR ('=' EXPR)*

UOP      := 'i'|'r'|'s'|'S'|'d'|'D'|'h'|'t'|'e'|'o'|'L'|'R'|'p'
BOP      := 'c' | 'z'
DIGITS   := DIGIT+
DIGIT    := '0'..'9'
VERDICT  := 'vc' | 'vw' | 'gw'
```

### 1.5 CoT 归约轨迹

每步找最内层可归约的算子，应用、重写全式，直到只剩数字。

示例 `ir23c0`：`ir23c0 →(c) ir230 →(r) i032 →(i) 032`

---

## 2. 数据生成

### 2.1 数据集配置文件（`config/*_*.yaml`，纯数据生成参数）

数据生成配置与训练配置**分离存放**，命名风格体现数据集内容本身：

| 文件 | 生成目标 | 输出 |
|---|---|---|
| `config/expr_500k_depth5.yaml` | 500k expr/check/cot 混合，depth 0-5，teacher split | `data/teacher_pretrain.jsonl` |
| `config/cot_50k_depth5.yaml` | 50k 纯 CoT，depth 0-5，teacher split | `data/teacher_sft.jsonl` |
| `config/expr_500k_depth5_student.yaml` | 500k expr/check，无 CoT，depth 0-5，student split | `data/student_pretrain.jsonl` |
| `config/stmt_100k_depth5.yaml` | 100k 纯 stmt，depth 0-5，student split | `data/student_sft.jsonl` |
| `config/eval_10k_depth5.yaml` | 10k 混合，depth 0-5，eval split，与训练集去重 | `data/eval.jsonl` |

### 2.2 数据集字段规范

每条样本存为 `.jsonl`（每行一个 JSON）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `prompt` | str | 模型输入前缀（含 `[BOS]`，空格分隔 token） |
| `target` | str | 需要预测的 token 串（空格分隔） |
| `expr` | str | 原始表达式字符串 |
| `result` | str | 解释器计算的正确结果 |
| `depth` | int | 组合深度 |
| `type` | str | `'stmt'` \| `'check'` \| `'cot'` |
| `split` | str | `'teacher_train'` \| `'student_train'` \| `'eval'` |

---

## 3. 模型架构

| | teacher $\Pi_T$ | student $\pi_\theta$ |
|---|---|---|
| vocab V | 34 (32 + CoT) | 同 teacher |
| context C | 96 | 96 |
| d_model | 192 | 64 |
| n_layers | 6 | 3 |
| n_heads | 6 (d_head 32) | 4 (d_head 16) |
| d_ffn | 768 | 256 |
| **参数量** | **~2.67M** | **~0.15M** |

容量比 ~17.4×。Decoder-only，pre-LN，因果自注意力，GELU FFN，tie embedding。

---

## 4. 训练流程

### 4.1 完整工作流

```bash
# Install dependencies
python -m pip install -r requirements.txt
# 1. 生成数据集（必须先于训练）
python data.py --config config/expr_500k_depth5.yaml         # teacher pretrain data
python data.py --config config/cot_50k_depth5.yaml           # teacher SFT data
python data.py --config config/expr_500k_depth5_student.yaml # student pretrain data
python data.py --config config/stmt_100k_depth5.yaml         # student SFT data

python data.py --config config/eval_10k_depth5.yaml          # eval data

# 2. Teacher pipeline
python pretrain.py --config config/teacher_pretrain.yaml
python sft.py      --config config/teacher_sft.yaml
python grpo.py     --config config/teacher_grpo.yaml         # -> teacher_final

# 3. Student pipeline
python pretrain.py --config config/student_pretrain.yaml
python sft.py      --config config/student_sft.yaml          # shared starting point
python kd.py       --config config/student_kd.yaml           # off-policy soft label
python opd.py      --config config/student_opd.yaml          # on-policy soft label
python grpo.py     --config config/student_grpo.yaml         # verifiable reward

# 4. Visualization（--config 决定输出目录 log/<name>/fig/）
python visualize_loss.py   --config config/teacher_pretrain.yaml
python visualize_weight.py --config config/teacher_pretrain.yaml
python visualize_llm.py    --config config/teacher_pretrain.yaml rs1234

# 5. Inference REPL
python inference.py --config config/teacher_pretrain.yaml
python inference.py --config config/teacher_sft.yaml --model path/to/override.pt
```

### 4.2 配置文件规则

- 每个 Python 运行时**必须**通过 `--config` 显式指定配置文件，脚本内无任何硬编码默认路径。
- 配置文件名**全局唯一**；日志子目录按配置文件名建立（`log/teacher_pretrain/`）。
- **新建训练任务必须创建新配置文件**，禁止复用已有配置（否则日志会被覆盖）。
- 配置文件中必须包含**所有**可配置变量：数据路径、模型超参、训练超参、输出路径、日志路径、推理参数等。

---

## 5. 评估指标

| 指标 | 适用 | 说明 |
|---|---|---|
| CE / KL loss (train/val) | 全部 | 训练目标值 |
| **任务正确率 by depth** | 全部 | 评测集贪心解码 + 解释器判定，按 depth 0-5 拆开 |
| KL(ΠT ‖ πθ) on teacher prefix | KD/OPD | 干净路径上的学习程度 |
| KL(ΠT ‖ πθ) on student prefix | 全部 | 自身路径上的 KL；与上项之差 = **exposure bias** |
| mean reward | GRPO | verifier 平均 reward |
| KL(πθ ‖ π_ref) | GRPO | 偏离 reference 的程度 |

---

## 6. 日志规范

### 6.1 目录结构

```
log/<config_name>/           # e.g. log/teacher_pretrain/
├── metrics.jsonl            # 每 log_every 步一行标量指标（驱动所有曲线/柱状图）
├── landscape.npz            # 训练末一次性计算，驱动 loss landscape 图
├── pca_basis.npz            # PCA 主方向 d1, d2（仅 pretrain 阶段生成，下游阶段复用）
├── <config_name>.pt         # 该阶段最终模型权重（e.g. teacher_pretrain.pt）
├── checkpoint/              # 训练中保存的权重快照（float16，共 n_traj_ckpt 个，训练后保留）
│   ├── 000001.pt
│   ├── 001667.pt
│   └── ...
└── fig/                     # 可视化输出（每个 config 独立子目录）
    ├── A_loss_curves.png
    ├── B_reward_curves.png
    ├── C_acc_by_depth.png
    ├── D_exposure_bias.png
    ├── landscape_all.png    # 所有已训练阶段的 landscape 合图（单文件，无冗余）
    └── attn_L0_rs1234.png
```

不再生成 `config.json`（训练配置与日志目录一一对应，yaml 文件本身即为记录）。

### 6.2 metrics.jsonl 字段

```python
{
  "step": 1200,
  "train_loss": 0.83,
  "val_loss": 0.91,
  "task_acc": 0.74,
  "task_acc_by_depth": [0.99, 0.95, 0.88, 0.71, 0.42, 0.18],
  "kl_teacher_prefix": 0.12,
  "kl_student_prefix": 0.47,
  "mean_reward": 0.81,
  "kl_to_ref": 0.05,
  "grad_norm": 1.7,
  "param_step_norm": 0.014
}
```

### 6.3 Loss landscape 生成

训练中每隔固定步数将参数向量存入 `log/<stage>/checkpoint/`（float16，共 `n_traj_ckpt` 个，**训练结束后保留**，供后续分析）。训练结束后自动计算 landscape：

1. 加载 `log/<stage>/checkpoint/` 中全部 checkpoint。
2. **确定 PCA 方向**（跨阶段共享坐标系）：
   - **pretrain 阶段**（`pca_basis_path` 未配置）：对所有 checkpoint 相对 $\theta^*$ 的偏差矩阵做 SVD，取前 2 个右奇异向量 $d_1, d_2$，保存到 `log/<stage>/pca_basis.npz`。
   - **下游阶段**（`pca_basis_path` 已配置，如 `log/teacher_pretrain/pca_basis.npz`）：直接加载 pretrain 保存的 $d_1, d_2$，**不重新做 SVD**。所有阶段共用同一坐标系，landscape 可在同一张图中直接对比。
3. 将每个 checkpoint 投影到 $(d_1, d_2)$ 平面（相对 $\theta^*$）→ 训练轨迹。
4. 在 $\theta^* + \alpha d_1 + \beta d_2$ 网格（`grid_res × grid_res`，范围自动覆盖轨迹 + 15% 边距）上前向计算 loss → $Z$。
5. 写入 `log/<stage>/landscape.npz`。

`visualize_weight.py` 将所有已有 `landscape.npz` 的阶段绘入同一张 `landscape_all.png`，图标题注明 PCA 基准阶段。

---

## 7. 代码结构

```
llmlab/
├── lib/                      # 库模块（只被 import，不直接执行）
│   ├── lang.py               # 语言定义、解释器、表达式采样（唯一真相源）
│   ├── model.py              # Transformer 定义（teacher/student 共享）
│   └── metrics.py            # 共享：标量指标计算 + loss landscape
│
├── config/
│   # 数据生成配置（data.py 读取）
│   ├── expr_500k_depth5.yaml
│   ├── cot_50k_depth5.yaml
│   ├── expr_500k_depth5_student.yaml
│   ├── stmt_100k_depth5.yaml
│   └── eval_10k_depth5.yaml
│   # 训练配置（pretrain/sft/grpo 等读取）
│   ├── teacher_pretrain.yaml
│   ├── teacher_sft.yaml
│   ├── teacher_grpo.yaml
│   ├── teacher_sdpo.yaml
│   ├── student_pretrain.yaml
│   ├── student_sft.yaml
│   ├── student_kd.yaml
│   ├── student_opd.yaml
│   └── student_grpo.yaml
│
├── data/                     # 生成的数据集 *.jsonl
├── log/                      # 训练日志（每 config 一个子目录）
│   └── teacher_pretrain/
│       ├── metrics.jsonl
│       ├── landscape.npz
│       ├── pca_basis.npz     # PCA 主方向（pretrain 生成，下游阶段复用）
│       ├── teacher_pretrain.pt
│       ├── checkpoint/       # 训练中保存的权重快照（保留，供 landscape 和回溯分析）
│       └── fig/
│
├── data.py                   # 数据集生成
├── pretrain.py               # 预训练（teacher/student 共用）
├── sft.py                    # 有监督微调（teacher/student 共用）
├── grpo.py                   # GRPO 对齐（teacher/student 共用）
├── sdpo.py                   # SDPO 对齐（可选）
├── kd.py                     # 知识蒸馏（student 专用）
├── opd.py                    # On-policy 蒸馏（student 专用）
├── inference.py              # 交互式推理 REPL
├── visualize_loss.py         # 绘制 loss/accuracy 曲线（只读 metrics.jsonl）
├── visualize_weight.py       # 绘制 loss landscape（只读 landscape.npz）
└── visualize_llm.py          # 可视化 attention weights
```

### 7.1 职责划分

| 类型 | 文件 | 规则 |
|---|---|---|
| 库模块 | `lib/*.py` | 只被 import，不含 `if __name__ == '__main__'` |
| 可执行脚本 | 根目录 `*.py` | 必须通过 `--config` 指定配置，脚本内无硬编码路径 |
| 数据配置 | `config/*_*.yaml` | 文件名描述数据集内容；不含训练超参 |
| 训练配置 | `config/<stage>.yaml` | 文件名 = 日志目录名；不含数据生成参数 |

---

## 8. 超参数结构（训练配置 yaml 五个顶级块）

```yaml
data:
  path: data/teacher_pretrain.jsonl   # 已生成数据集的路径

model:                                # 模型架构超参
  vocab_size: 34
  context_len: 96
  d_model: 192
  ...

train:                                # 训练超参
  lr: 1.0e-3
  batch_size: 1024
  n_steps: 50000
  ...

output:                               # 输出路径（全部显式配置）
  model_path: log/teacher_pretrain/teacher_pretrain.pt
  log_dir: log/teacher_pretrain

logging:                              # 日志与 checkpoint 配置
  log_every: 1000
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  tmp_ckpt_dir: log/teacher_pretrain/checkpoint   # checkpoint 存入 log/ 并永久保留
  # pca_basis_path: log/teacher_pretrain/pca_basis.npz  # 下游阶段填此项以复用 PCA 坐标系
  ...

inference:                            # 推理参数（inference.py 和 visualize_llm.py 读取）
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 96
  show_cot: true
  show_raw: false
  device: cuda
```

---

## 9. 与笔记概念的对应

| 概念 | 体现 |
|---|---|
| SFT = hard label | student_sft（teacher 采样 token） |
| KD = soft label | student_kd（teacher 完整分布） |
| off / on-policy | KD off / OPD,GRPO on |
| exposure bias | KD vs OPD 正确率差 + 图 D（teacher vs student prefix KL） |
| teacher unreliability | OPD 被 teacher 封顶 vs GRPO 突破（depth 4–5） |
| verifiable reward (RLVR) | GRPO reward = 解释器，独立于 teacher |
| REINFORCE / baseline / IS+clip | GRPO 的 ∇log π·A / 组内归一化 / ρ+clip |
| forward / reverse KL | KD/OPD forward；可加 reverse KL ablation |
| chain-of-thought | CoT teacher pretrain vs 非 CoT；CoT 对深组合泛化 |
| SDPO | teacher_sdpo（feedback = prompt 内 ground-truth 答案） |
