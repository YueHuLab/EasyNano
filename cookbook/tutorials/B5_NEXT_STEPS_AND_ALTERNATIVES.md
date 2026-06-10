# B5 binder design — 现状评估 + 替代方案

> 今晚 4 次深度分析后（设计流程总结、v2 详解、版本效果排名、PAE vs
> ipSAE 区分）我们到达的位置。直接回答两个问题：
> **(1) 目前方案能用吗？** (2) **还有更好的思路吗？**

---

## 0. 一句话先回答

**能用，但是有重大不确定性**。当前最好设计
（`v9_best_15seed_p116Y`）在一个单一模型（Full ESMFold2）下 iptm
0.692 ± 0.020，**没有交叉验证**、**没有实验验证**、**没有亲和力
数据**。对实验验证来说这是合理的起点；对生产级抗体来说还不够。
**最值得做的一步是交叉验证**（AF2 / Boltz-1），如果两个独立模型
都给出高 iptm，可以比较放心地进实验。

---

## 1. 当前方案到底给我们什么

### 1.1 数值
- **CDR**: `GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIY` (32 个 CDR 位)
- **Framework 微调**: 1 个位 (CDR H3-16: R→Y；索引错位 bug 后正名)
- **Robust iptm** (n=9, Full ESMFold2): median 0.692, std 0.020,
  range [0.651, 0.716]
- **pTM**: 0.83-0.85
- **ipSAE_p10(CDR, epi)**: 6.3-7.5 Å

### 1.2 这意味着什么
- **模型对预测有信心**：iptm 0.69 = 模型说"界面应该长这样"
- **预测的几何合理**：ipSAE_p10 = 6-7 Å = 几对 CDR-epitope 接触在 6-7 Å
- **无明显双峰**：std 0.020 = 所有 9 个样本都在 [0.65, 0.72]，单 basin
- **CDR→epi 距离 ~10 Å** = CDR 整体在 epitope 附近

### 1.3 这**不**意味着什么
- **不**意味着它在体外或体内真的结合
- **不**意味着亲和力是 nM、μM 还是 mM
- **不**意味着别的模型 (AF2, Boltz-1) 也同意
- **不**意味着它在不同表位上做得好
- **不**意味着它稳定、可溶、无聚集

ESMFold2 的 iptm 0.69 可能对应一个**真实**的、纳摩尔级 binder，也可能对应一个**模型自信的、错误的**预测。**没有任何 in silico 指标能代替湿实验**。

---

## 2. 当前方案的 4 个主要不足

### 2.1 缺乏交叉验证
- 整个 7 天项目只用了一个模型 (ESMFold2)
- Boltz-1 在离线环境装失败 (CCD download timeout)
- AF2 / AF2-complex 没试过
- **风险**：如果 ESMFold2 的 iptm 头有偏差，0.69 可能是"模型对自己自信"的伪信号
- **修复成本**：低 (1-2 天)

### 2.2 缺乏解离常数/亲和力数据
- iptm 是"界面预测置信度"，不是 ΔG
- 一个 iptm 0.7 的设计可能亲和力是 nM、μM、还是 mM
- 真实 iptm-affinity 相关性在不同 benchmark 上波动很大
- **风险**：高 (这是 in silico 设计的基本限制)
- **修复成本**：必须做 SPR / BLI / ELISA

### 2.3 局限于一个 framework + 一个 H3 长度
- 我们用了 framework III (VH 单一家族) + H3 长度 16
- 实际抗体 H3 长度 8-25，framework 有几十种
- **风险**：中等 (可能错过了更好的设计空间)
- **修复成本**：高 (要重跑整个 pipeline × N 个 framework × M 个 H3 长度)

### 2.4 没有开发性 (developability) 评估
- 当前 CDR 富含 Y、W (4 个 Y, 1 个 W)
- 可能聚集、不稳定、有免疫原性
- 没做：Tm、聚集倾向、PTM 风险位点
- **风险**：中等
- **修复成本**：中 (需要序列分析工具 + 实验)

---

## 3. 替代设计思路（6 个没试过的方向）

按 **"可能提升 + 实现难度"** 综合排序：

### 3.1 AF2 / Boltz-1 交叉验证 ⭐⭐⭐⭐⭐ （最值得做）

**做什么**：用 AlphaFold2-complex (或 Boltz-1，等离线解决) 折叠当前最优 CDR，比对 iptm 和 ipSAE。

**为什么值得**：
- **零设计新东西**，只验证现有的
- 30 分钟 - 2 小时就能做完
- 如果 ESMFold2 和 AF2 **一致**说"高 iptm"，可信度大幅提升
- 如果**不一致**，就要非常小心 — 可能是 ESMFold2 的偏差

**预期结果**：
- 一致高 iptm (0.6+ in both) → **直接进实验**
- 一个高一个低 → **风险高**，可能需要重新设计
- 都低 → 我们的设计是 ESMFold2 偏差，**需要换方法**

**实现**：
- AF2: 装 `colabfold` 或本地 AF2 weights，跑 binder+target complex
- Boltz-1: 等离线解决 CCD 后用
- 不需要重设计，只需要再 fold 一次

### 3.2 遗传算法 / MCMC 在当前设计基础上搜索 ⭐⭐⭐⭐

**做什么**：把当前 `v9_best_15seed_p116Y` 当起点，用 GA 在 CDR 32 位 × 20 AA = 640 维离散空间搜索：
- 每代：随机突变 1-2 个 CDR 位
- 评估：Full ESMFold2 × 1 sample (17s/次)
- 选择：Metropolis 准则 (接受更好 iptm，或以概率 exp(-ΔE/T) 接受更差)
- 跑 1000-5000 代 ≈ 5-25 小时

**为什么值得**：
- v9 iter-design (Step 4) 是 no-op 因为 v9 优化的是 structure_prior + epitope loss，不是 iptm
- GA/MCMC 直接优化 iptm，跳出了 v9 损失函数的限制
- 17s/次评估下，5000 代 = 24 小时，可行

**预期结果**：
- 可能找到 iptm 0.75+ 的单点改进
- 可能发现当前 basin 内有 0.7-0.8 的"小高点"
- 不会找到全新 basin (搜索半径有限)

**实现**：
- 简单的 `for mutation in proposed: fold + accept/reject`
- 关键：种子策略、突变率、Metropolis T 调度
- 比 RFdiffusion / Chroma 简单

### 3.3 H3 长度扫描 ⭐⭐⭐⭐

**做什么**：把 H3 长度从 16 改成 8、10、12、14、18、20、22，对每个长度跑 v9 multi-start (5 seeds)

**为什么值得**：
- H3 是抗体最可变的 loop，长度直接决定 paratope 形状
- 我们只用了 16 (framework III 默认)，但实际抗体 H3 长度 8-25
- 不同的 H3 长度可能打开全新的 basin
- 实施成本：~7 个长度 × 5 seeds × 30 步 ≈ 35 runs ≈ 4 小时

**预期结果**：
- H3=12-14 可能给出更紧密的 epitope 接触 (H3 短一点反而好)
- H3=18-20 可能接触 epitope 更广的面积
- 最佳长度 + 最佳 seed 可能达到 iptm 0.75+

**实现**：
- 改 `binder_template` (要重做 abnumber CDR 识别)
- 改 `v2_init` 序列 (用对应长度的 H3 序列，比如 J-segment 拼接)

### 3.4 ProteinMPNN / ESM-IF 反向折叠 ⭐⭐⭐

**做什么**：把当前的 binder 3D 坐标（已折叠的）当固定 backbone，用 ProteinMPNN 设计新序列。

**为什么值得**：
- ProteinMPNN 是 orthogonal 方法 (它只看 3D 结构，不看序列相似性)
- 可以验证"当前 CDR 对这个 3D 形状是否最优"
- 可以生成"在 3D 形状相似但序列更自然"的备选 CDR
- 一次跑几分钟，**极快**

**预期结果**：
- 备选 CDR 序列（可能 70% 序列相似，30% 替换）
- 这些序列有相似的预测 3D 形状
- 折叠回 iptm 后，可信的备选序列集
- 某些备选可能比当前序列 iptm 更高

**实现**：
- `pip install proteinmpnn`
- 输入 binder 的 CA 坐标 → 输出 50-100 个备选序列
- 用 Full ESMFold2 评估每个备选
- 选 top 5 进实验

### 3.5 RFdiffusion / Chroma 生成新骨架 ⭐⭐⭐

**做什么**：用 RFdiffusion 在 B5 表位周围**生成全新的 binder 骨架**（不限于 Ig fold），然后用 ProteinMPNN 设计序列。

**为什么值得**：
- 完全不同的设计空间 (非 Ig fold binder，比如 DARPins、FN3、de novo mini-protein)
- 可能给出比 Ig fold 更高的亲和力
- 已有 benchmark 表明 RFdiffusion 在 de novo binder 上可达 sub-nM

**预期结果**：
- 完全不同拓扑的 binder (没有 framework，只有 de novo 结构)
- iptm 应该 0.5-0.8 都有
- 真实亲和力可能比 Ig fold 更好

**实现**：
- 装 `rfdiffusion` (要 ~10 GB 模型权重)
- 需要 GPU (RFdiffusion 在 MPS 上跑得慢，建议 GPU 服务器)
- 5-10 个 generation run × 100 个 backbone = 500-1000 个候选
- 然后 ProteinMPNN 序列设计 + ESMFold2 折叠验证

**风险**：
- 完全 de novo 的 binder 在表达、纯化、稳定性上比 Ig fold 难
- 不一定能产生可溶、稳定的蛋白
- 实验成本高

### 3.6 不同 framework 家族扫描 ⭐⭐

**做什么**：换不同的 VH/VL framework (IGHV3-23, IGHV5-51, IGKV1-39, etc.)，每个 framework 跑 5-seed v9 multi-start。

**为什么值得**：
- 不同 framework 给出不同的全局 pose
- 不同 framework 的"自然"接触化学不同
- 可能有 framework 给出比 III 更高的 iptm

**风险**：
- 实施成本高 (重做 abnumber + 重跑 pipeline)
- iptm 提升可能不大 (因为 B5 的 epitope 几何是固定的)
- 收益不一定

### 3.7 其他值得考虑的小事

- **Boltz-2**: 比 Boltz-1 更新，2024 年发布。可能有更好的 iptm calibration
- **AlphaFold3 (AF3)**: 2024 年发布，专门为复合物设计。如果能拿到权重，值得用
- **MSA-based scoring**: 用类似抗体的 MSA 看当前 CDR 是否"自然"
- **结合 pLDDT 过滤**: 当前 binder 的 CDR pLDDT 应该 > 80，否则模型自己都不信
- **Tm 预测**: 用 ProteinMPNN-Tm 或 Rosetta 预测 Tm，过滤掉不稳定的

---

## 4. 不同的需求对应不同的"够用"标准

| 用途 | 现状够不够 | 还需要什么 |
|---|---|---|
| 论文 proof-of-concept | ✓ 够 | 写好方法学描述即可 |
| 实验验证起点 | ⚠️ 勉强够 | AF2 交叉验证（半小时工作） |
| 学术发表 | ⚠️ 需要补一些 | 交叉验证 + 1-2 个备选 + 湿实验亲和力 |
| 工业级 lead | ✗ 远远不够 | 多 framework 扫描 + 亲和力成熟 + developability + 免疫原性 + 表达 + 稳定性 |
| 临床候选 | ✗ 还差 100 步 | 完全重新走 lead optimization 流程 |

---

## 5. 我的建议：3 步走

### 第一步（今晚/明天）：交叉验证
- 用 AF2-complex 折叠当前 CDR + 4-5 个 v9 multi-start 的备选
- 比对 ESMFold2 vs AF2 的 iptm
- **如果两者一致高** → 直接进实验
- **如果不一致** → 风险高，需要重新评估

### 第二步（接下来一周）：用 GA 在当前 basin 内搜索
- 实现 3.2 (GA/MCMC)
- 跑 5000 代，看能不能找到 iptm 0.75+ 的单点改进
- 同时用 ProteinMPNN 备选 CDR (3.4)
- 选 top 5 序列进实验

### 第三步（如果时间允许）：换方法对比
- 装 RFdiffusion，生成 de novo binder backbone
- 和当前 ESMFold2 设计对比
- 哪个 iptm 高、表达稳定、用哪个

### 第四步（如果有钱/时间）：湿实验
- 选 top 3 序列，做基因合成
- 表达为 VHH-Fc 或 scFv
- 用 BLI 或 SPR 测 B5 结合
- 测 Tm、聚集、稳定性
- 这是任何 in silico 设计的最终判决

---

## 6. 关键的认识

**任何 in silico 抗体设计都需要湿实验闭环**：
- iptm 0.69 是个**信号**，不是**结论**
- 同样的 iptm 在不同模型上、不同靶点上、不同 framework 上的可靠性差别巨大
- ESMFold2 在 Ig fold binder 上的 benchmark 还不完整
- 唯一可靠的判断是：**模型预测 + 实验验证 + 迭代优化**

**当前方案作为"第一个可测试的序列"是合格的**。但**不要把它当成"答案"**。它是一个需要被验证、被精炼、被实验反驳或确认的假设。

**如果让我赌一把**：
- 我赌 ESMFold2 iptm 0.69 至少有 60% 概率对应一个**真实结合**的抗体 (K_D 1 μM 或更好)
- 但 40% 概率它可能亲和力很弱 (mM 级) 或不结合
- 这个不确定性只能用实验消除
- **AF2 交叉验证**是消除不确定性最便宜的方法（半小时工作）

**如果让我推荐路线**：
1. 今晚就做 AF2 交叉验证
2. 如果一致高 iptm，明天就合成基因做实验
3. 同时跑 GA 搜索看有没有 iptm 0.75+ 的备选
4. 实验结果出来后，要么确认这个方案，要么重新设计
