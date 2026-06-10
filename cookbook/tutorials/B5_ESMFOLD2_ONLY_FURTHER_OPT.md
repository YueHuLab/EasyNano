# B5 — 纯 ESMFold2 还能不能继续优化 iptm？

> 直接回答：**可以，但路径不是"再调 v9"，而是把优化器从"梯度"换成"离散采样器"。
> 期望 +0.02 到 +0.05 iptm。预算 1.5 - 24 小时不等。**

---

## 0. 核心认识：当前设计 loop 卡在哪

**事实**：
- v9 iter-design（Step 4）在 3 个 seed × 30 步 = 90 次 Adam update 里，
  **0 个 CDR 位被改**。
- 16 个负面版本（v3, v4, v5, v6, v7, v10a-c, v11, v12, v12b, v13, v13b,
  v14, v15, v9 seeds 6-15）**全部 re-converge 到 v9 baseline 或更差**。
- 唯一逃出新 basin 的方法是 v9 5-seed multi-start（找到 0.717 峰值），
  这是因为 seed 2 落在了 basin 边缘的另一个吸引域。

**结论**：
> **Adam 在 32 CDR × 20 AA = 640 维离散空间里，进了 basin 就是
> 局部最优。梯度在 basin 边缘 ≈ 0。再调 v9 / 换 LR / 加正则 都是
> 死路。要逃出 basin，必须用全局采样器。**

---

## 1. 最高 ROI：GA / MCMC 离散序列空间搜索 ⭐⭐⭐⭐⭐

**为什么这是第一**：
- 设计 loop 是 640 维离散空间的**局部爬山**
- 局部爬山找不到别的 basin —— Step 4 已经证伪
- GA/MCMC 是这空间的**全局采样器**，能跨 barrier
- 不需要碰 ESMFold2 内部，只用它的"fold → iptm"作为 oracle

**算法伪代码**（~50 行 Python）：
```python
import numpy as np

seq = v9_best_15seed_p116Y              # 当前最优起点 (iptm 0.692 ± 0.020)
best_iptm = median_n9(seq)              # n=9 评估作为起点真值
T = 0.05                                # Metropolis 温度
cdr_positions = list(range(101, 117))   # H3 16 位 (binder indexing)
# 或者全部 32 位 (H1 + H2 + H3)
all_cdr = H1_POS + H2_POS + H3_POS
AA = "ACDEFGHIKLMNPQRSTVWY"

history = [(best_iptm, seq)]
for gen in range(5000):
    # 提议：随机 1-2 个 CDR 位改成随机 AA
    n_mut = np.random.choice([1, 2], p=[0.7, 0.3])
    new_seq = list(seq)
    for _ in range(n_mut):
        pos = np.random.choice(all_cdr)
        new_aa = np.random.choice(list(AA))
        new_seq[pos] = new_aa
    new_seq = "".join(new_seq)

    # 评估：单 seed × 1 sample (17s)
    new_iptm = single_sample_fold(new_seq)

    # Metropolis 接受准则
    ΔE = best_iptm - new_iptm
    if ΔE < 0 or np.random.rand() < np.exp(-ΔE / T):
        seq = new_seq
        if new_iptm > best_iptm:
            best_iptm = new_iptm
            save(seq)
    history.append((best_iptm, seq))
    T *= 0.9995  # 冷却

# 收尾：top 50 候选用 n=9 重新评估
top50 = sorted(history, reverse=True)[:50]
final = [(median_n9(s), std_n9(s), s) for _, s in top50]
```

**预算**：
- 17s/fold × 5000 代 = **23 小时**
- 收尾 n=9 评估 top 50：17s × 9 × 50 = **2.1 小时**
- 总计 **~25 小时跑完一轮**

**为什么 v9 iter 失败而 GA 可能成功**：
- v9 iter 的"proposal"是 Adam 在 soft-logit 空间的**1 步** ≈
  argmax 几乎不变（logit 移 0.5，离 argmax 还有 1-2σ）
- GA 的"proposal"是**离散跳变**，一步能跨过任意高度的 barrier
- 评估指标是**真正的 iptm**（或 PAE 衍生），不是 L_epi 的 proxy

**预期结果**：
- 50% 概率找到 0.71-0.73 的**单点**改进
- 30% 概率找到 0.74+ 的**多步累计**改进
- 20% 概率当前 basin 已接近全局最优，GA 只给 ±0.01 抖动

**实施注意**：
- 初期 100 代是 burn-in，应该看到 history 的 max 单调上升
- 如果前 500 代没找到 > 当前 0.005 的改进，调高 T（0.05 → 0.1）
- H3-only 起步（16 位 × 20 = 320 维），收敛更快；扩展到 H1+H2+H3 是第二步

---

## 2. 第二档：3 个低成本改良 ⭐⭐⭐⭐

### 2.1 课程式 L_epi cutoff（v2 改成 cutoff 从 12Å 退火到 6Å）

**问题**：当前 v2/v9 cutoff = 8Å 固定。
- step 0 离 epitope 12Å → 梯度强但 L_epi 数值爆炸（ELU(4) ≈ 53）
- step 60 离 epitope 7Å → 梯度变弱
- 中段（step 20-40）L_epi 曲线不连续，Adam 步长震荡

**改法**（v2 源码改 5 行）：
```python
# 替换 design_b5_mps_v2.py 里的固定 cutoff=EPITOPE_CUTOFF
def cutoff_schedule(step, n_steps):
    """课程式：早期松（12Å），后期紧（6Å）"""
    progress = step / n_steps
    return 12.0 - 6.0 * progress  # 12 → 6 Å 线性

# 替换 L_epi 里的 ELU 为 smooth sigmoid
def contact_term(min_dist, cutoff, sharpness=2.0):
    """替代 F.elu(min_dist - cutoff)，更平滑"""
    return torch.sigmoid((min_dist - cutoff) * sharpness) * (min_dist - cutoff).clamp(min=0)
```

**为什么可能更好**：
- 早期 cutoff 大 → 梯度方向稳定（朝 epitope 走）
- 后期 cutoff 小 → 梯度精细化（贴近 6Å）
- 总损失曲线更平，不会在 step 20 突然跳变

**预期**：iptm +0.01 到 +0.02（没有 GA 猛但便宜得多）

**预算**：2 小时（5 个 seed × 100 步 × 17s = 1.4h + 评估 0.5h）

### 2.2 随机重启 + 段间接力

**问题**：当前每个 seed 跑 100 步**独立**。一个 seed 在 step 40 卡住就废了。

**改法**：
```python
# 每 20 步检查：median_iptm 在最近 5 步有提升吗？
# 没有就重启：保留 top-3 接力点，轮换使用
top_k = []  # (iptm, seq) 的 top 3
for restart in range(5):
    init = top_k[restart % 3] if top_k else v2_init
    seq, history = run_v9(init, n_steps=20)
    new_iptm = median_n9(seq)
    top_k = sorted(top_k + [(new_iptm, seq)], reverse=True)[:3]
```

**为什么可能更好**：
- v9 5-seed 的 0.717 来自 seed 2 步 56 —— 一个**中段**接力的盆地
- 接力式 = 强制走"短而频繁"的山路
- 比单次 100 步更能跨越 basin barrier

**预算**：5× 100 步 × 17s = 1.4 小时

**预期**：iptm +0.01 到 +0.03（找到 0.71-0.72 候选概率高）

### 2.3 关键残基对加权（不是均匀 21 个 epitope）

**问题**：当前 L_epi 把 21 个 epitope 残基**一视同仁**。但实际结合面
只有 3-5 个 "hotspot"。

**改法**（~30 行）：
```python
# 用 ESMFold2 的 distogram confidence 给 21 个 epitope 残基打分
# 高 confidence（"模型觉得这里是接触面"）给更高权重
def compute_epi_weights(disto_logits, epi_indices, bin_distance):
    """对每个 epi 残基，预测它作为接触面的概率"""
    cross = disto_logits[:, -binder_length:, epi_indices, :]
    probs = torch.softmax(cross, dim=-1)
    e_dist = (probs * bin_distance).sum(-1)  # [1, L_b, 21]
    min_dist = e_dist.min(dim=1).values      # [1, 21]
    # 接触概率 ≈ sigmoid(8 - min_dist)  # 8Å 阈值
    contact_prob = torch.sigmoid(8.0 - min_dist)
    return contact_prob.squeeze()  # [21]

# 在 L_epi 里用加权
epi_w = compute_epi_weights(disto_logits, epi_indices, bin_distance)
per_res_loss = (min_dist - cutoff) * epi_w[None, :]  # [1, L_b]
```

**为什么可能更好**：
- 21 个残基里可能有 5-8 个是真正的"必须接触"位
- 集中梯度到这 5-8 个 → 更快收敛、更少 zigzag
- 远离 epitope 边界的位（false positive）不再被错误吸引

**预期**：iptm +0.005 到 +0.02

**预算**：3 小时

---

## 3. 第三档：高级技巧 ⭐⭐⭐

### 3.1 CMA-ES（连续 soft-logit 空间的全局优化）

比 GA 强但复杂：把 32×20=640 维 soft logits 当连续空间，用协方差
矩阵自适应采样。`pip install cma` 就能用。

- 优点：自适应步长，比 GA 收敛快 5-10×
- 缺点：要调超参（σ_init, population_size），比 GA 复杂
- 预算：5000 代 × 17s × 5 pop = **12 小时**

### 3.2 贝叶斯优化 + GP surrogate

每代跑 n=3 真评估 + 用 GP 预测 1000 个候选的 iptm，挑 GP
"expected improvement" 最大的去真评估。**总评估数减 100×**。

需要 `botorch` 或 `scikit-optimize`：
```python
from botorch import acquire ExpectedImprovement
from botorch.models import SingleTaskGP

# 初始：随机 20 个序列 n=3 评估 = 60 次 fold
init_X, init_Y = random_search(20)
gp = SingleTaskGP(init_X, init_Y)

for it in range(50):
    # GP 预测 1000 个候选，挑 EI 最大的 5 个真评估
    cand = generate_candidates(1000)
    ei = ExpectedImprovement(gp, best_f=init_Y.max())
    next_X = cand[ei.argmaxTopK(5)]
    next_Y = median_n3(next_X)
    gp = gp.condition_on_observations(next_X, next_Y)
```

- 优点：评估数降到 50 × 5 + 60 = 310 次 = 1.5 小时
- 缺点：GP 在 640 维不太行，要先用 autoencoder 降到 ~50 维
- 预期：+0.01 到 +0.03 iptm

### 3.3 学到的 iptm 代理（learned surrogate）

用已有的 v9 multi-start 数据（15 seeds × 30 步 × 各 iptm =
~450 数据点）训练一个小型 NN：seq → iptm。**当 surrogate 用**比
GP 还快 100×。

- 优点：评估只需 ms 级
- 缺点：要小心 surrogate 偏差（surrogate 学到的不是真实 iptm landscape）
- 实施：用 80% 数据训练，20% 验证；surrogate 预测 vs 真 iptm 的
  R² > 0.7 才用

### 3.4 软 res_type ensemble

设计时同时跑 T=0.5/1.0/2.0 三个 soft res_type，**平均梯度**。
降低单温度的 zigzag。本质是 v2 already 的 T annealing 的"反向操作"。

```python
losses = []
for T in [0.5, 1.0, 2.0]:
    binder_probs_20 = F.softmax(soft_logits / T, dim=-1)
    # ... 计算 L_epi ...
    losses.append(L_epi)
total = sum(losses) / 3
```

---

## 4. 现实期望总结

| 方向 | 提升 | 时间 | 风险 | 优先级 |
|---|---|---|---|---|
| **GA/MCMC（5000代）** | +0.01 到 +0.04 | 24h | 低 | ⭐⭐⭐⭐⭐ |
| 课程式 cutoff | +0.01 到 +0.02 | 2h | 低 | ⭐⭐⭐⭐ |
| 随机重启接力 | +0.01 到 +0.03 | 1.5h | 低 | ⭐⭐⭐⭐ |
| 关键残基加权 | +0.005 到 +0.02 | 3h | 中 | ⭐⭐⭐⭐ |
| CMA-ES | +0.02 到 +0.05 | 12h | 中 | ⭐⭐⭐ |
| 贝叶斯优化 | +0.01 到 +0.03 | 6h | 中 | ⭐⭐⭐ |
| iptm 代理 | 不直接提升 | 4h | 高 | ⭐⭐ |
| 软 res_type ensemble | +0.005 到 +0.015 | 5h | 低 | ⭐⭐⭐ |

**最稳的组合**：GA（24h）+ 课程式 cutoff（2h）+ 接力（1.5h）
= **~28h，期望 +0.03 到 +0.06 iptm**。

---

## 5. 什么时候该停

**经验法则**：
1. **单次提升 < 0.01** → 别再调超参了，去做交叉验证
2. **跨 n=9 验证 std 仍然 > 0.05** → 模型本身不确定，再调也不会消失
3. **发现 v9_best_15seed_p116Y 已经在"信息论极限"**（CDR 数量有限，
   构象有限）→ 停止优化，转湿实验

**判断"信息论极限"的信号**：
- GA 跑了 1000+ 代 best_iptm 完全不变
- 多次不同 init 收敛到同一个 basin
- 改 framework（不动 CDR）也只在该 basin 内 ±0.02

**如果已经达到信息论极限**：
- 唯一逃出方法是**新模型**（AF2、Boltz-1）或**新方法**（RFdiffusion）
- 不值得在 ESMFold2 + 当前 framework 上继续烧时间

---

## 6. 推荐的执行顺序（如果今晚/明天开干）

### Phase 1（1.5-3 小时，低成本快速验证）

1. **随机重启接力**（1.5h）
   - 改 v9 加 "每 20 步检查 + 接力"
   - 5 restart × 100 步 × 17s = 1.4h
   - 期望：找到 0.71-0.72 候选

2. **课程式 cutoff**（2h）
   - 改 v2 源码 5 行
   - 5 seed × 100 步 × 17s = 1.4h
   - 期望：+0.01-0.02 iptm

### Phase 2（24-30 小时，主攻 GA）

3. **GA 5000 代**（24h）
   - 用 Phase 1 的最好结果做起点
   - 17s × 5000 = 23h + 收尾 2h
   - 期望：+0.02-0.04 iptm

### Phase 3（如果 Phase 1+2 没找到新 basin）

4. **CMA-ES**（12h）
   - 改用连续 soft-logit 空间
   - 比 GA 更智能但更难调

5. **iptm 代理**（4h）
   - 用 450 个 v9 数据点训练
   - 加速 GA 的下一轮（100× 加速）

### 关键原则

**每个 phase 结束后立即 n=9 评估**：
- 不能只看 single-seed iptm（noise floor 0.07）
- n=9 median + std 是唯一可信指标
- 如果新候选 std > 0.05 → 不要采用，继续找

---

## 7. 一些非优化但可能更重要的角度

如果上面所有优化加起来只能给 +0.05 iptm，那下面这些角度可能 ROI
更高：

### 7.1 验证当前结果（不是优化）
- **AF2 交叉验证**（半小时）：和 ESMFold2 比对 iptm
- 这是"消除不确定性"不是"提升数值"
- 比任何优化都更能决定"要不要进实验"

### 7.2 序列后处理（不是优化）
- **ProteinMPNN 备选 CDR**：用当前 3D 形状生成备选
- 选 5 个最不像当前 CDR 的备选 → ESMFold2 评估
- 找到"形状相同但序列不同"的备选 = 抗突变的备份

### 7.3 物理化学约束（不是优化）
- 测当前 CDR 的 **pI、聚集倾向、PTM 位点**
- 替换掉"预测会聚集"的位
- 这步可能不提升 iptm 但**提升成功率**

---

## 8. 一句话决策树

```
问：还剩多少时间？
├── ≤ 4 小时 → 做随机重启接力（1.5h）+ 评估
├── 4-12 小时 → 上面 + 课程式 cutoff + CMA-ES
├── 12-30 小时 → 上面 + GA 5000 代
└── > 30 小时 → 上面全部 + iptm 代理 + 序列后处理

问：你的目标是？
├── 拿一个"看起来不错"的序列进实验 → 接力 + cutoff（半天）
├── 拿"全球最优"序列 → GA + CMA-ES + 代理（3 天）
└── 拿"高置信度"序列 → 上面所有 + AF2 交叉验证
```

---

## 9. 总结

**纯 ESMFold2 还能提升 iptm 多少？**

- **+0.01-0.02**：3 小时内可达（接力 + cutoff）
- **+0.02-0.05**：24 小时内可达（GA）
- **+0.05+**：需要新模型或新方法

**最有希望的单一动作**：GA 5000 代（24h 投入，期望 +0.02-0.04）

**最便宜的提升**：随机重启接力（1.5h，期望 +0.01-0.03）

**最大风险**：v9_best_15seed_p116Y 已在信息论极限 → 任何优化都
只给 ±0.01 抖动 → 立即停止优化转交叉验证
