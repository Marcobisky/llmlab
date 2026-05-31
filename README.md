# LLMlab 全流程计划方案 (v5)

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

> `h`+`t` 互补（拼起来 = 原串），`e`+`o` 同理。

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
STMT     := EXPR '=' RESULT [EOS]                                     # 陈述句
           | EXPR '=' CANDIDATE '?' VERDICT [EOS]                      # 检查句
           | EXPR '<think>' TRACE '</think>' '=' RESULT [EOS]          # CoT 句

EXPR     := UOP EXPR                      # 一元算子作用于右侧
           | DIGITS BOP EXPR               # 二元: 左数字串 + 算子 + 右表达式
           | DIGITS                        # 原子操作数

TRACE    := EXPR ('=' EXPR)*               # 逐步归约轨迹（含初始式与每步重写）

UOP      := 'i'|'r'|'s'|'S'|'d'|'D'|'h'|'t'|'e'|'o'|'L'|'R'|'p'
BOP      := 'c' | 'z'
DIGITS   := DIGIT+
DIGIT    := '0'..'9'
VERDICT  := 'vc' | 'vw' | 'gw'
```

文法是 LL(1) 无歧义。**执行方向**：最右算子（离操作数最近）最先执行。

**解析验证**：
- `rs18273399` = `r(s(18273399))` = `r(12337899)` = `99873321` ✓
- `ir23c0` = `i(r(23c0→230))` = `i(r(230))` = `i(032)` = `032` ✓
- `1cr343c98` = `c(1, r(c(343,98)→34398))` = `c(1, r(34398))` = `c(1, 89343)` = `189343` ✓

### 1.5 CoT 归约轨迹

每步找**最内层可归约的算子**（操作数已全是数字），应用、重写全式，直到只剩数字。

示例 `ir23c0`：`ir23c0 →(c) ir230 →(r) i032 →(i) 032`

对应 CoT 句：`ir23c0<think>ir23c0=ir230=i032=032</think>=032`

### 1.6 检查子语言

| 判定 | 条件 | 示例 |
|---|---|---|
| `vc` | EXPR 合语法 **且** CANDIDATE == interpret(EXPR) | `r138=831?vc` |
| `vw` | EXPR 合语法 **但** CANDIDATE ≠ interpret(EXPR) | `r138=813?vw` |
| `gw` | EXPR 不合语法 | `rs=9201?gw` |

`vw` 错误候选生成方式（可配 `vw_corruption`）：swap 两位 / 替换若干位 / 截断 / 打乱。

### 1.7 参考解释器

```python
UNARY = {
    'i': lambda x: x,
    'r': lambda x: x[::-1],
    's': lambda x: ''.join(sorted(x)),
    'S': lambda x: ''.join(sorted(x, reverse=True)),
    'd': lambda x: ''.join(c for i,c in enumerate(x) if i==0 or c!=x[i-1]),
    'D': lambda x: ''.join(c*2 for c in x),
    'h': lambda x: x[:len(x)//2],
    't': lambda x: x[len(x)//2:],
    'e': lambda x: x[0::2],
    'o': lambda x: x[1::2],
    'L': lambda x: x[1:]+x[:1] if x else '',
    'R': lambda x: x[-1:]+x[:-1] if x else '',
    'p': lambda x: x + x[::-1],
}
BINARY = {
    'c': lambda a, b: a + b,
    'z': lambda a, b: ''.join(sum(zip(a, b), ())) + a[len(b):] + b[len(a):],
}

def interpret(expr):                       # 返回 (result_str, is_valid)
    result, rest = _parse(expr)
    if result is None or rest != '': return None, False
    return result, True

def _parse(s):                             # 递归下降, 对应 §1.4 BNF
    if not s: return None, ''
    if s[0] in UNARY:
        inner, rest = _parse(s[1:])
        if inner is None: return None, ''
        return UNARY[s[0]](inner), rest
    if s[0].isdigit():
        j = 0
        while j < len(s) and s[j].isdigit(): j += 1
        left = s[:j]
        if j < len(s) and s[j] in BINARY:
            right, rest = _parse(s[j+1:])
            if right is None: return None, ''
            return BINARY[s[j]](left, right), rest
        return left, s[j:]
    return None, ''
```

---

## 2. 数据生成

### 2.1 表达式随机采样

```python
def sample_expr(depth, n_operands, operand_len_range, ops_enabled):
    # 1. 采样 n_operands 个随机数字串（长度从 operand_len_range 取）
    # 2. 用 n_operands-1 个随机二元算子（c/z）右结合串起来
    # 3. 在最外层前缀 depth 个随机一元算子
    # 返回 EXPR 字符串
```

### 2.2 Pretrain 语料（预训练）

| 成分 | 占比 | 格式 |
|---|---|---|
| 直接陈述句 | 1 − check_fraction − cot_fraction | `[BOS] EXPR = RESULT [EOS]` |
| 检查句 | check_fraction | `[BOS] EXPR = CAND ? VERDICT [EOS]` |
| CoT 句 | cot_fraction | `[BOS] EXPR <think> TRACE </think> = RESULT [EOS]` |

**不加无标签噪声**（noise_rate = 0）；错误只通过检查句的 `vw`/`gw` 引入，带标签。

### 2.3 Pair / CoT 语料（后训练）

| 任务 | prompt | target |
|---|---|---|
| 生成 | `[BOS] EXPR =` | `RESULT [EOS]` |
| 生成 (CoT) | `[BOS] EXPR` | `<think> TRACE </think> = RESULT [EOS]` |
| 验证 | `[BOS] EXPR = CAND ?` | `VERDICT [EOS]` |

### 2.4 训练 / 测试集严格分离（重要）

- 所有表达式生成后做**去重 + 哈希**，保证评测集与任何阶段的训练集**零重叠**。
- **深度分离**用于测泛化：teacher 只训 depth ≤ 3；评测集覆盖 depth 0–5。depth 4–5 是所有模型都没在 teacher 信号里见过的"域外"区，专门用来看 GRPO（verifier 在任意 depth 可靠）能否突破蒸馏类方法。
- 三套数据：`teacher_train`（depth ≤ 3）、`student_train`（depth ≤ 5）、`eval`（depth 0–5，与前两者无重叠）。

### 2.5 数据集字段规范

pretrain 和 post-train 共用**同一套字段**（存为 `.jsonl`，每行一个 JSON）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `prompt` | str | 模型输入前缀。pretrain: `"[BOS] EXPR"`；post-train 生成: `"[BOS] EXPR ="`；post-train 验证: `"[BOS] EXPR = CAND ?"` |
| `target` | str | 需要预测的 token 串。pretrain 陈述句: `"= RESULT [EOS]"`；pretrain CoT: `"<think> TRACE </think> = RESULT [EOS]"`；post-train 生成: `"RESULT [EOS]"`；post-train 验证: `"VERDICT [EOS]"` |
| `expr` | str | 原始表达式字符串，如 `"rs1234"` |
| `result` | str | 解释器计算的正确结果，如 `"4321"` |
| `depth` | int | 组合深度（一元算子嵌套层数 + 二元算子数，≥ 0） |
| `type` | str | 句式类型：`"stmt"`（陈述句）\| `"check"`（检查句）\| `"cot"`（CoT 句） |
| `split` | str | 所属划分：`"teacher_train"` \| `"student_train"` \| `"eval"` |

pretrain 中 `text = prompt + target`，训练对全序列做 next-token CE；post-train 只对 `target` 部分计 loss（`prompt` 部分 mask 掉）。

---

## 3. 模型架构

| | teacher $\Pi_T$ | student $\pi_\theta$ |
|---|---|---|
| vocab V | 32 (+2 CoT) | 同 teacher |
| context C | 48 | 48 |
| d_model | 192 | 64 |
| n_layers | 6 | 3 |
| n_heads | 6 (d_head 32) | 4 (d_head 16) |
| d_ffn | 768 | 256 |
| **参数量** | **~2.67M** | **~0.15M** |

容量比 ~17.4×。Decoder-only：token + 学习式 position embedding，pre-LN，因果自注意力，GELU FFN，输出层 tie embedding。（尺寸均为超参，后续可调。）

---

## 4. 训练流程

### 4.1 teacher pipeline（→ `teacher_final.pt`）

1. **pretrain**（`pretrain.py`）：在 teacher_train（depth ≤ 3）上 next-token CE，**启用 CoT（`cot_fraction` > 0）** → `teacher_pretrain.pt`
2. **SFT**（`sft.py`）：从 teacher_pretrain 出发，loss 只在 target 部分 → `teacher_sft.pt`
3. **GRPO**（`grpo.py`）：从 teacher_sft 出发，verifier reward，仍只用 depth ≤ 3 → `teacher_grpo.pt`
4. **SDPO**（`sdpo.py`，可选/进阶）：从 teacher_sft 出发 → `teacher_sdpo.pt`
   - SDPO 的 feedback 定义：把 ground-truth 答案放进 prompt 作为 feedback $f$，self-teacher $\pi(\cdot\mid \text{EXPR}, f, y_{<t})$（带答案，近乎完美）蒸馏回无 feedback 的 $\pi(\cdot\mid \text{EXPR}, y_{<t})$。advantage $=\log\frac{\pi(\hat y\mid x,f,y_{<t})}{\pi(\hat y\mid x,y_{<t})}$。
5. 在评测集上比较 grpo/sdpo，最强者存为 `teacher_final.pt`。

**为什么 depth ≤ 3 够用**：teacher 的作用是在其**训练域（depth 0–3）内**提供高质量 soft 分布——这是蒸馏信号的来源。depth 4–5 teacher 预计不完美，这正是"teacher unreliability"实验所需。在 depth 0–3 内，加了 CoT pretrain + GRPO 的 teacher 应能接近满分，soft 分布有足够的信息量。对 depth 4–5，teacher 犯错，GRPO（通过 verifier 绕开 teacher）才得以在该区间超越蒸馏方法——这是实验要验证的核心现象。

**teacher CoT pretrain 的作用**：pretrain 阶段启用 CoT（`cot_fraction` ≈ 0.2）让 teacher 在预训练期就学会逐步归约，显著提升深组合的泛化能力。与 SFT 才加入 CoT 相比，pretrain CoT 使模型在 depth 4–5 的正确率更高，蒸馏信号质量更好（即使 teacher 仍不完美），同时 CoT vs 非 CoT teacher 的对比本身也是一个实验维度。

**teacher 不完美的来源**：所有阶段都只用 depth ≤ 3 的数据 → teacher 在 depth 4–5 靠泛化，正确率自然下降。这是 teacher unreliability 的根。

### 4.2 student pipeline

1. **pretrain**（`pretrain.py`）：在 student_train（depth ≤ 5）上 next-token CE → `student_pretrain.pt`
2. **SFT 热身**（`sft.py`）：从 student_pretrain 出发，在 student_train 上 SFT → `student_sft.pt`
   - **这是后续所有支路的公共起点。** student_sft.pt 本身也作为 "SFT-only" 基线参与对比。
3. 从 `student_sft.pt` 出发，三条支路（都用 `teacher_final.pt` 作为 teacher）：

#### KD（`kd.py`）— off-policy, soft label
teacher prefix，匹配 teacher 完整分布（forward KL）：
$$\mathcal{L}_{\text{KD}}=\sum_t \text{KL}\big(\Pi_T(\cdot\mid s_t)\,\|\,\pi_\theta(\cdot\mid s_t)\big),\quad \hat y\sim\Pi_T(\cdot\mid x)$$

#### OPD（`opd.py`）— on-policy, soft label
student 自己 rollout 产生 prefix，teacher 在其上给 soft 分布：
$$\mathcal{L}_{\text{OPD}}=\sum_t \text{KL}\big(\Pi_T(\cdot\mid s_t)\,\|\,\pi_\theta(\cdot\mid s_t)\big),\quad y\sim\pi_\theta(\cdot\mid x)$$
与 KD 唯一区别：prefix 来源（teacher → student）→ 修正 exposure bias。

#### GRPO（`grpo.py`）— on-policy, verifiable reward
student rollout，reward 由**解释器**（`reward.py`）验证，与 teacher 无关：
$$r(y\mid x)=V(x,y)=\text{正确位置比例}$$
$$A_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)+\epsilon},\quad \mathcal{L}=-\sum_t\min\big(\rho_t A_i,\;\text{clip}(\rho_t,1\!-\!\epsilon,1\!+\!\epsilon)A_i\big)+\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$$
其中 $\rho_t=\pi_\theta(y_t\mid s_t)/\pi_{\theta_{\text{old}}}(y_t\mid s_t)$，reference = student_sft。

### 4.3 关键对照（注意：post-training 现在是串行 pipeline）

| 对照 | 共同起点 | 唯一变量 | 揭示 |
|---|---|---|---|
| student_sft vs student_kd | student_sft | 是否再做软标签蒸馏 | hard label 后软标签的增量 |
| **student_kd vs student_opd** | student_sft | off vs on-policy prefix | **exposure bias** |
| **student_opd vs student_grpo** | student_sft | teacher 分布 vs verifier | **teacher unreliability**（GRPO 突破 teacher） |

> KD/OPD/GRPO 都从 student_sft 出发，所以后两个对照变量隔离干净。SFT vs KD 因为 KD 接在 SFT 之后，不再是纯粹的 "hard vs soft" 2×2，但仍可解释为"软标签蒸馏带来的增量"。

---

## 5. 评估指标

| 指标 | 适用 | 说明 |
|---|---|---|
| CE / KL loss (train/val) | 全部 | 训练目标值 |
| **任务正确率 by depth** | 全部 | 评测集上贪心解码 + 解释器判定，按 depth 拆开。**核心跨范式指标** |
| $\text{KL}(\Pi_T\|\pi_\theta)$ on teacher prefix | KD/OPD | 干净路径上学得多像 |
| $\text{KL}(\Pi_T\|\pi_\theta)$ on student prefix | 全部 | 自己路径上学得多像 → 与上一项之差 = **exposure bias** |
| mean reward | GRPO（+teacher GRPO） | verifier 平均 reward |
| $\text{KL}(\pi_\theta\|\pi_{\text{ref}})$ | GRPO | 偏离 reference |
| teacher 正确率基准线 | 汇总图 | 水平线，看谁突破 teacher |

---

## 6. 日志规范：存什么、怎么算（核心）

**原则：区分瞬态与持久。** 完整权重 checkpoint 只在单次运行内瞬态保留，训练末算出紧凑数据后即删除；持久化的全是小文件。

### 6.1 每个 `log/<stage>/` 持久化的内容（都是小文件）

```
log/<stage>/
├── config.json         # 该次运行全部超参（从 config/*.yaml 拷贝 + git hash）
├── metrics.jsonl       # 每 N step 一行标量指标（见 6.2），驱动所有曲线/柱状图
└── landscape.npz       # 训练末一次性算出（见 6.3），驱动 loss landscape 图
```

模型权重单独存到 `model/<name>.pt`（每个 stage 一个最终权重，不是几十个）。

### 6.2 训练中每 N step 记录的标量指标（写入 metrics.jsonl）

每行一个 JSON，字段都是标量或小数组（每行几十字节，可高频记录）：

```python
{
  "step": 1200,
  "train_loss": 0.83,                  # 当前训练目标值（CE / KL / GRPO loss 等）
  "val_loss": 0.91,                    # 固定验证集上的目标值
  "task_acc": 0.74,                    # 评测集贪心解码总正确率
  "task_acc_by_depth": [0.99,0.95,0.88,0.71,0.42,0.18],  # 按 depth 0..5 拆
  "kl_teacher_prefix": 0.12,           # KD/OPD: teacher 路径上的 KL
  "kl_student_prefix": 0.47,           # 全部: student 路径上的 KL（与上项差 = exposure bias）
  "mean_reward": 0.81,                 # GRPO: verifier reward
  "kl_to_ref": 0.05,                   # GRPO: 偏离 reference
  "grad_norm": 1.7,                    # 梯度范数
  "param_step_norm": 0.014             # ||θ_t − θ_{t−1}||，看参数移动速度
}
```

- 这些指标在**固定的小评测 batch**（如 256 条，含各 depth）上计算，开销小、可每 N step 算一次。
- 图 A（loss 曲线）、图 B（reward 曲线）、图 C（正确率 by depth 柱状）、图 D（exposure bias）**全部只用 metrics.jsonl**，不需要任何权重。

### 6.3 loss landscape 数据的产生

**所有 stage 统一使用 PCA 方向 + 训练目标 loss**：

1. 训练中把完整参数向量存到**临时目录** `tmp/<stage>/`（`n_traj_ckpt` 个，float16，训练结束即删）。
2. 训练末：对这些临时 checkpoint 做 PCA 取前 2 个方向 $d_1,d_2$。
3. 在 $\theta^*+\alpha d_1+\beta d_2$ 网格（`grid_res`×`grid_res`，如 31×31）上，用固定 eval batch 前向计算**训练目标 loss** → $Z$：
   - pretrain / SFT：CE loss
   - KD / OPD：forward KL
   - GRPO：GRPO loss（见下）
4. 投影各临时 checkpoint 到 $(\alpha,\beta)$ → 轨迹。
5. 存 `landscape.npz = {alpha_grid, beta_grid, Z, traj_alpha, traj_beta}`（几十 KB），**删除临时 checkpoint**。

**GRPO landscape 的 loss 计算**：GRPO loss 是 $-\sum_t\min(\rho_t A_i,\,\text{clip}(\rho_t,\ldots)A_i)+\beta\,\text{KL}(\pi_\theta\|\pi_\text{ref})$，其中 $\rho_t=\pi_{\theta'}(y_t|s_t)/\pi_{\theta^*}(y_t|s_t)$。在网格点 $\theta'$ 处，只需用 $\pi_{\theta^*}$ 预先采好的固定 rollout batch（序列 $y$ 和优势 $A$ 均固定），就可以像 CE loss 一样做一次前向传播算出 GRPO loss，无需额外的 rollout 或 greedy decode。

临时 checkpoint 大小（float16）：
- student（0.15M params）：30 × 0.3MB ≈ 9MB，可忽略。
- teacher（2.67M params）：30 × 5.3MB ≈ 160MB，单次训练结束即清除，在硬盘上只短暂存在。

checkpoint 数量和存储格式均可配（`n_traj_ckpt`、`ckpt_dtype`）。

### 6.4 三层代码职责

- **train_*.py**（即 pretrain/sft/grpo/kd/opd/sdpo）：训练 → 写 metrics.jsonl → 训练末算 landscape.npz → 存最终 `model/*.pt` → 删临时 checkpoint。
- **visualize_loss.py**：只读 `log/*/metrics.jsonl`，画 loss/reward/accuracy/exposure-bias 曲线与柱状（图 A–D）。
- **visualize_weight.py**：只读 `log/*/landscape.npz`，画每个 stage 的 landscape 等高线 + 轨迹。
- 两个 visualize 脚本都**不碰模型与数据**，纯绘图。

---

## 7. 代码结构

```
llmlab/
├── data/               # 生成的数据集 *.jsonl (文件名体现该数据集的部分关键配置)
├── fig/                # 生成的图表 *.png (文件名体现该图表的部分关键配置)
├── config/             # 每个 stage 一个 yaml 超参文件
│   ├── teacher_pretrain.yaml    # teacher pretrain 所有超参 (所用数据集配置, model 超参, pretrain 配置, 输出 pt path 等)
│   ├── teacher_sft.yaml         # teacher sft 所有超参 (所用数据集配置, base model pt path, SFT 配置, 输出 pt path 等)
│   ├── teacher_grpo.yaml        # teacher grpo 所有超参 (所用数据集配置, base model pt path, GRPO 配置, 输出 pt path 等)
│   ├── teacher_sdpo.yaml        # teacher sdpo 所有超参 (所用数据集配置, base model pt path, SDPO 配置, 输出 pt path 等)
│   ├── student_pretrain.yaml    # student pretrain 所有超参 (所用数据集配置, model 超参, pretrain 配置, 输出 pt path 等)
│   ├── student_sft.yaml         # student sft 所有超参 (所用数据集配置, base model pt path, SFT 配置, 输出 pt path 等)
│   ├── student_kd.yaml          # student kd 所有超参 (所用数据集配置, base model pt path, KD 配置, 输出 pt path 等)
│   ├── student_opd.yaml         # student opd 所有超参 (所用数据集配置, base model pt path, OPD 配置, 输出 pt path 等)
│   ├── student_grpo.yaml        # student grpo 所有超参 (所用数据集配置, base model pt path, GRPO 配置, 输出 pt path 等)
│   ├── student_sdpo.yaml        # student sdpo 所有超参 (所用数据集配置, base model pt path, SDPO 配置, 输出 pt path 等)
│   └── inference.yaml           # 模型推理所有超参 (model pt path, temperature 等)
├── lib/                # 所有被 import、不能直接被执行的 python 文件
│   ├── lang.py         # 语言定义 + 解释器 + 表达式采样 + verifier（被多处 import，唯一真相源）
├── data.py             # import lang; 读入 config/ 中的某个指定 yaml (用户可在此文件中配置), 在 data/ 中生成满足配置的数据集
├── model.py            # Transformer 定义（student/teacher 共享, 具体超参数由 config/ 决定）
├── metrics.py          # 共享：训练中算标量指标 + 训练末算 landscape.npz
├── reward.py           # import lang；rollout 验证 + reward 计算
├── pretrain.py         # 读入 config/ 中的指定配置, 训练模型并按照配置输出到 model/ 和 log/
├── sft.py              # 同上，适用于 teacher 和 student 的 SFT 阶段 
├── grpo.py             # 同上，适用于 teacher 和 student 的 GRPO 阶段
├── sdpo.py             # 同上，适用于 teacher 和 student 的 SDPO 阶段（可选/进阶）
├── opd.py              # 同上，适用于 student 的 OPD 阶段
├── kd.py               # 同上，适用于 student 的 KD 阶段
├── inference.py        # 模型推理脚本，读入 config/inference.yaml 进行配置, 标准终端对话式接口
├── visualize_loss.py   # 只读 log/*/metrics.jsonl
├── visualize_weight.py # 只读 log/*/landscape.npz
├── model/              # 保存的权重 *.pt（每 stage 一个最终权重）
├── tmp/                # 训练时临时的权重 *.pt（每 stage 多个 checkpoint，训练结束即删）
└── log/                # 每 stage 一个子目录: config.json, metrics.jsonl, landscape.npz
```

---

## 8. 完整超参数汇总（config/*.yaml 结构）

每个 yaml 文件分为五个顶级 key：`data`、`model`、`train`、`output`、`logging`。相同模型架构的 yaml 之间 `model` 块完全一致；`train` 块只含该 stage 独有的超参。

### teacher_pretrain.yaml

```yaml
data:
  path: data/teacher_pretrain.jsonl   # 生成/读取路径
  split: teacher_train                # 写入每条数据的 split 字段
  n_samples: 50000
  depth_range: [0, 3]
  operand_len_range: [1, 6]
  n_operands_range: [1, 3]
  ops_enabled: all                    # 或列表如 [r, s, c, z, ...]
  check_fraction: 0.15
  verdict_ratio: [5, 3, 2]            # vc : vw : gw
  cot_fraction: 0.2                   # teacher pretrain 启用 CoT
  vw_corruption: swap                 # swap | replace | truncate | shuffle
  noise_rate: 0.0
  seed: 42

model:
  vocab_size: 34                      # 32 + <think> + </think>
  context_len: 48
  d_model: 192
  n_layers: 6
  n_heads: 6
  d_head: 32
  d_ffn: 768
  dropout: 0.0
  tie_embedding: true
  pos_embedding: learned              # learned | sinusoidal
  norm_type: pre_ln
  activation: gelu

train:
  lr: 3.0e-4
  lr_schedule: cosine                 # cosine | constant
  batch_size: 128
  n_steps: 3000
  warmup_steps: 200
  weight_decay: 0.1
  grad_clip: 1.0
  seed: 42
  device: cuda

output:
  model_path: model/teacher_pretrain.pt
  log_dir: log/teacher_pretrain/

logging:
  log_every: 50                       # 每 N step 写一次 metrics.jsonl
  eval_batch_size: 256
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  ckpt_dtype: float16
  tmp_ckpt_dir: tmp/teacher_pretrain/
  grid_res: 31                        # landscape 网格分辨率（grid_res × grid_res）
  landscape_alpha_range: [-1.0, 1.0]
  landscape_beta_range: [-1.0, 1.0]
```

### teacher_sft.yaml

```yaml
data:
  path: data/teacher_sft.jsonl
  split: teacher_train
  n_samples: 20000
  depth_range: [0, 3]
  operand_len_range: [1, 6]
  n_operands_range: [1, 3]
  ops_enabled: all
  check_fraction: 0.0                 # SFT 只用生成任务
  cot_fraction: 0.0                   # SFT target 是否含 CoT（可配）
  seed: 42

model:                                # 与 teacher_pretrain.yaml 完全一致
  vocab_size: 34
  context_len: 48
  d_model: 192
  n_layers: 6
  n_heads: 6
  d_head: 32
  d_ffn: 768
  dropout: 0.0
  tie_embedding: true
  pos_embedding: learned
  norm_type: pre_ln
  activation: gelu

train:
  base_model_path: model/teacher_pretrain.pt
  lr: 1.0e-4
  lr_schedule: cosine
  batch_size: 64
  n_steps: 1500
  warmup_steps: 100
  weight_decay: 0.1
  grad_clip: 1.0
  loss_on_prompt: false               # prompt 部分不计 loss
  seed: 42
  device: cuda

output:
  model_path: model/teacher_sft.pt
  log_dir: log/teacher_sft/

logging:
  log_every: 50
  eval_batch_size: 256
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  ckpt_dtype: float16
  tmp_ckpt_dir: tmp/teacher_sft/
  grid_res: 31
  landscape_alpha_range: [-1.0, 1.0]
  landscape_beta_range: [-1.0, 1.0]
```

### teacher_grpo.yaml

```yaml
data:
  path: data/teacher_sft.jsonl        # 只用 prompt 字段
  split: teacher_train
  depth_range: [0, 3]
  seed: 42

model:                                # 同上（teacher 尺寸）
  vocab_size: 34
  context_len: 48
  d_model: 192
  n_layers: 6
  n_heads: 6
  d_head: 32
  d_ffn: 768
  dropout: 0.0
  tie_embedding: true
  pos_embedding: learned
  norm_type: pre_ln
  activation: gelu

train:
  base_model_path: model/teacher_sft.pt
  lr: 5.0e-5
  lr_schedule: cosine
  batch_size: 32                      # prompts per step
  n_steps: 600
  warmup_steps: 50
  weight_decay: 0.0
  grad_clip: 1.0
  # GRPO 专有
  n_rollouts_per_prompt: 8            # G：每个 prompt 采样几条
  clip_eps: 0.2                       # importance sampling clip ε
  kl_coeff: 0.05                      # β：KL 惩罚系数
  reward_fn: interpreter              # interpreter | partial_match
  rollout_temperature: 1.0            # 采样温度
  loss_on_prompt: false
  seed: 42
  device: cuda

output:
  model_path: model/teacher_grpo.pt
  log_dir: log/teacher_grpo/

logging:
  log_every: 20
  eval_batch_size: 256
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  ckpt_dtype: float16
  tmp_ckpt_dir: tmp/teacher_grpo/
  grid_res: 31
  landscape_alpha_range: [-1.0, 1.0]
  landscape_beta_range: [-1.0, 1.0]
```

### teacher_sdpo.yaml

```yaml
data:
  path: data/teacher_sft.jsonl
  split: teacher_train
  depth_range: [0, 3]
  seed: 42

model:                                # 同上（teacher 尺寸）
  vocab_size: 34
  context_len: 48
  d_model: 192
  n_layers: 6
  n_heads: 6
  d_head: 32
  d_ffn: 768
  dropout: 0.0
  tie_embedding: true
  pos_embedding: learned
  norm_type: pre_ln
  activation: gelu

train:
  base_model_path: model/teacher_sft.pt
  lr: 5.0e-5
  lr_schedule: cosine
  batch_size: 32
  n_steps: 600
  warmup_steps: 50
  weight_decay: 0.0
  grad_clip: 1.0
  # SDPO 专有
  feedback_type: ground_truth         # 把 GT 结果拼入 prompt 作为 feedback
  sdpo_temperature: 1.0               # 采样温度
  kl_coeff: 0.05
  loss_on_prompt: false
  seed: 42
  device: cuda

output:
  model_path: model/teacher_sdpo.pt
  log_dir: log/teacher_sdpo/

logging:
  log_every: 20
  eval_batch_size: 256
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  ckpt_dtype: float16
  tmp_ckpt_dir: tmp/teacher_sdpo/
  grid_res: 31
  landscape_alpha_range: [-1.0, 1.0]
  landscape_beta_range: [-1.0, 1.0]
```

### student_pretrain.yaml

```yaml
data:
  path: data/student_pretrain.jsonl
  split: student_train
  n_samples: 50000
  depth_range: [0, 5]
  operand_len_range: [1, 6]
  n_operands_range: [1, 3]
  ops_enabled: all
  check_fraction: 0.15
  verdict_ratio: [5, 3, 2]
  cot_fraction: 0.0                   # student pretrain 默认不加 CoT
  vw_corruption: swap
  noise_rate: 0.0
  seed: 42

model:
  vocab_size: 34
  context_len: 48
  d_model: 64
  n_layers: 3
  n_heads: 4
  d_head: 16
  d_ffn: 256
  dropout: 0.0
  tie_embedding: true
  pos_embedding: learned
  norm_type: pre_ln
  activation: gelu

train:
  lr: 3.0e-4
  lr_schedule: cosine
  batch_size: 64
  n_steps: 3000
  warmup_steps: 200
  weight_decay: 0.1
  grad_clip: 1.0
  seed: 42
  device: cuda

output:
  model_path: model/student_pretrain.pt
  log_dir: log/student_pretrain/

logging:
  log_every: 50
  eval_batch_size: 256
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  ckpt_dtype: float16
  tmp_ckpt_dir: tmp/student_pretrain/
  grid_res: 31
  landscape_alpha_range: [-1.0, 1.0]
  landscape_beta_range: [-1.0, 1.0]
```

### student_sft.yaml

```yaml
data:
  path: data/student_sft.jsonl
  split: student_train
  n_samples: 20000
  depth_range: [0, 5]
  operand_len_range: [1, 6]
  n_operands_range: [1, 3]
  ops_enabled: all
  check_fraction: 0.0
  cot_fraction: 0.0
  seed: 42

model:                                # student 尺寸
  vocab_size: 34
  context_len: 48
  d_model: 64
  n_layers: 3
  n_heads: 4
  d_head: 16
  d_ffn: 256
  dropout: 0.0
  tie_embedding: true
  pos_embedding: learned
  norm_type: pre_ln
  activation: gelu

train:
  base_model_path: model/student_pretrain.pt
  lr: 1.0e-4
  lr_schedule: cosine
  batch_size: 64
  n_steps: 800
  warmup_steps: 80
  weight_decay: 0.1
  grad_clip: 1.0
  loss_on_prompt: false
  seed: 42
  device: cuda

output:
  model_path: model/student_sft.pt
  log_dir: log/student_sft/

logging:
  log_every: 50
  eval_batch_size: 256
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  ckpt_dtype: float16
  tmp_ckpt_dir: tmp/student_sft/
  grid_res: 31
  landscape_alpha_range: [-1.0, 1.0]
  landscape_beta_range: [-1.0, 1.0]
```

### student_kd.yaml

```yaml
data:
  path: data/student_sft.jsonl        # teacher 在这些 prompt 上 rollout
  split: student_train
  depth_range: [0, 5]
  seed: 42

model:                                # student 尺寸（同上）
  vocab_size: 34
  context_len: 48
  d_model: 64
  n_layers: 3
  n_heads: 4
  d_head: 16
  d_ffn: 256
  dropout: 0.0
  tie_embedding: true
  pos_embedding: learned
  norm_type: pre_ln
  activation: gelu

train:
  base_model_path: model/student_sft.pt
  teacher_model_path: model/teacher_final.pt
  lr: 1.0e-4
  lr_schedule: cosine
  batch_size: 64
  n_steps: 800
  warmup_steps: 80
  weight_decay: 0.0
  grad_clip: 1.0
  # KD 专有
  kl_direction: forward               # forward | reverse
  temperature: 1.0                    # softmax 温度（蒸馏时平滑 logits）
  alpha_ce: 0.0                       # CE loss 混合比（0 = 纯 KL）
  loss_on_prompt: false
  seed: 42
  device: cuda

output:
  model_path: model/student_kd.pt
  log_dir: log/student_kd/

logging:
  log_every: 50
  eval_batch_size: 256
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  ckpt_dtype: float16
  tmp_ckpt_dir: tmp/student_kd/
  grid_res: 31
  landscape_alpha_range: [-1.0, 1.0]
  landscape_beta_range: [-1.0, 1.0]
```

### student_opd.yaml

```yaml
data:
  path: data/student_sft.jsonl        # 只取 prompt，rollout 由 student 自己生成
  split: student_train
  depth_range: [0, 5]
  seed: 42

model:                                # student 尺寸（同上）
  vocab_size: 34
  context_len: 48
  d_model: 64
  n_layers: 3
  n_heads: 4
  d_head: 16
  d_ffn: 256
  dropout: 0.0
  tie_embedding: true
  pos_embedding: learned
  norm_type: pre_ln
  activation: gelu

train:
  base_model_path: model/student_sft.pt
  teacher_model_path: model/teacher_final.pt
  lr: 5.0e-5
  lr_schedule: cosine
  batch_size: 32
  n_steps: 600
  warmup_steps: 60
  weight_decay: 0.0
  grad_clip: 1.0
  # OPD 专有
  kl_direction: forward
  temperature: 1.0
  rollout_temperature: 1.0            # student rollout 时的采样温度
  loss_on_prompt: false
  seed: 42
  device: cuda

output:
  model_path: model/student_opd.pt
  log_dir: log/student_opd/

logging:
  log_every: 50
  eval_batch_size: 256
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  ckpt_dtype: float16
  tmp_ckpt_dir: tmp/student_opd/
  grid_res: 31
  landscape_alpha_range: [-1.0, 1.0]
  landscape_beta_range: [-1.0, 1.0]
```

### student_grpo.yaml

```yaml
data:
  path: data/student_sft.jsonl
  split: student_train
  depth_range: [0, 5]
  seed: 42

model:                                # student 尺寸（同上）
  vocab_size: 34
  context_len: 48
  d_model: 64
  n_layers: 3
  n_heads: 4
  d_head: 16
  d_ffn: 256
  dropout: 0.0
  tie_embedding: true
  pos_embedding: learned
  norm_type: pre_ln
  activation: gelu

train:
  base_model_path: model/student_sft.pt
  reference_model_path: model/student_sft.pt  # KL 惩罚的参考点
  lr: 5.0e-5
  lr_schedule: cosine
  batch_size: 32
  n_steps: 400
  warmup_steps: 40
  weight_decay: 0.0
  grad_clip: 1.0
  # GRPO 专有
  n_rollouts_per_prompt: 8
  clip_eps: 0.2
  kl_coeff: 0.05
  reward_fn: interpreter              # interpreter | partial_match
  rollout_temperature: 1.0
  loss_on_prompt: false
  seed: 42
  device: cuda

output:
  model_path: model/student_grpo.pt
  log_dir: log/student_grpo/

logging:
  log_every: 20
  eval_batch_size: 256
  eval_data_path: data/eval.jsonl
  n_traj_ckpt: 30
  ckpt_dtype: float16
  tmp_ckpt_dir: tmp/student_grpo/
  grid_res: 31
  landscape_alpha_range: [-1.0, 1.0]
  landscape_beta_range: [-1.0, 1.0]
```

### inference.yaml

```yaml
model_path: model/student_grpo.pt    # 要加载的权重
model:                               # 必须与训练时一致
  vocab_size: 34
  context_len: 48
  d_model: 64
  n_layers: 3
  n_heads: 4
  d_head: 16
  d_ffn: 256
  tie_embedding: true
  pos_embedding: learned
  norm_type: pre_ln
  activation: gelu
inference:
  temperature: 0.0                   # 0 = greedy，>0 = sampling
  top_p: 1.0
  max_new_tokens: 48
  show_cot: true                     # 是否打印 <think>...</think> 内容
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
| REINFORCE / baseline / IS+clip | GRPO 的 $\nabla\log\pi\cdot A$ / 组内归一化 / $\rho$+clip |
| forward / reverse KL | KD/OPD forward；可加 reverse KL ablation |
| chain-of-thought | CoT teacher pretrain vs 非 CoT；CoT 对深组合泛化 |
| SDPO | teacher_sdpo（feedback = prompt 内 ground-truth 答案） |

---

## 10. 已确认 / 待确认

- ✅ 全部 15 算子
- ✅ teacher 训 depth≤3、评 depth≤5，作为超参，训练/测试严格分离
- ✅ 架构尺寸作为超参
- ✅ teacher pretrain 启用 CoT（`cot_fraction` > 0）
- ✅ 数据集字段规范（§2.5）
- ✅ 所有 stage 的 landscape 统一使用训练目标 loss（GRPO 阶段用 GRPO loss，固定 rollout batch）
- ✅ 所有超参整理为 config/*.yaml 结构
- 下一步：从 `lang.py`（解释器 + 数据生成 + 单元测试验证解释器正确性）开始写代码。
