# 学习型潜空间世界模型（阶段 1）

用规则引擎生成轨迹，训练一个小型潜空间世界模型，导出为浏览器可直接加载的权重
`prototype/learned_weights.js`。游戏里用顶栏的"引擎"选择框或 `?engine=learned` 切换。

## 复现

```bash
pip install torch numpy                         # CPU 版即可
node model/gen_data.js /tmp/uc-data 3000 60 1   # 训练集：3000 条 × 60 年
node model/gen_data.js /tmp/uc-val 400 60 2     # 验证集
python model/train.py /tmp/uc-data /tmp/uc-val prototype/learned_weights.js   # 4 核 CPU 约 4 分钟
node model/evaluate.js                          # 与规则引擎对比
```

## 模型

| 部分 | 结构 | 作用 |
|---|---|---|
| 编码器 E | 6 → 64 → 64 → 8 | 可观测 logit → 潜状态 z |
| 解码器 G | 8 → 64 → 64 → 6 | z → 可观测 logit |
| 转移集成 T₁..T₃ | (8+6) → 64 → 64 → 8（残差） | z' = z + f(z, a)，a 为当年征兆强度 |

- 损失：重建 + 12 步开环多步预测（每个集成成员各自展开）+ 潜空间一致性。
- 噪声：从一步残差估计每维潜噪声标准差，游戏中注入潜空间，然后投影回编码器流形 `z ← E(G(z))`。
- 分歧：三个转移模型预测的标准差（以噪声标准差归一化），在同样的开环展开下取第 95 百分位作为 1。
  游戏中分歧超过 1 时额外推高梦蚀——"模型没把握"直接成为世界在"做梦"的原因。

## 游戏如何使用它

`prototype/sim.js` 中所有机制都通过两个接口作用于潜状态，因此两种引擎共用同一套玩法：

- `omenDir(world, key)`：征兆的潜空间方向。学习引擎中为 `mean_k [T_k(z, a) − T_k(z, 0)]`，随状态变化。
- `pushObs(world, dl)`：在可观测 logit 空间里推动世界（恐慌、居民行动、锚定、世界愿望的拉力）。
  学习引擎中用解码器雅可比的最小范数伪逆映射回潜空间。

训练数据不含梦蚀、噩梦、居民对冲和世界愿望——这些都由游戏层在潜空间之外施加。
