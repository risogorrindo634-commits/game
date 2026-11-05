"""训练潜流的潜空间世界模型（阶段 1）。

结构（全部是小 MLP，tanh 激活，方便在浏览器里用纯 JS 推理）：
  encoder    E: 可观测 logit (6) -> 潜状态 z (8)
  decoder    G: z (8) -> 可观测 logit (6)
  transition T_k: [z, a] (8+6) -> z + Δz，k = 1..K 的集成（ensemble）
其中 a 是当年施加的征兆强度向量（反向为负，合成为两项各 0.6）。

损失：
  重建      |G(E(l_t)) - l_t|²
  多步预测  从 E(l_0) 出发用 T 展开 H 步，|G(z_t) - l_t|²   （每个集成成员各自展开）
  潜一致性  |T(E(l_t), a_t) - sg(E(l_{t+1}))|²

训练后从残差估计潜空间噪声标准差（每维），游戏里把噪声注入潜空间，并在每步之后把潜状态
投影回编码器流形（z ← E(G(z))）；集成成员之间的分歧作为"离开训练分布"的度量，驱动游戏中的梦蚀，
其尺度在同样的开环展开下校准（第 95 百分位记为 1）。

用法：python model/train.py <data_dir> <val_dir> <out_js> [--steps 6000]
依赖：torch、numpy
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument('data'); p.add_argument('val'); p.add_argument('out')
p.add_argument('--steps', type=int, default=4000)
p.add_argument('--latent', type=int, default=8)
p.add_argument('--hidden', type=int, default=64)
p.add_argument('--ensemble', type=int, default=3)
p.add_argument('--horizon', type=int, default=12)
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.set_num_threads(os.cpu_count() or 1)


def load(d):
    meta = json.load(open(os.path.join(d, 'meta.json')))
    N, T, D = meta['N'], meta['T'], meta['D']
    L = np.fromfile(os.path.join(d, 'logits.f32'), dtype='<f4').reshape(N, T + 1, D)
    A = np.fromfile(os.path.join(d, 'actions.f32'), dtype='<f4').reshape(N, T, D)
    return torch.tensor(L), torch.tensor(A), meta


L, A, meta = load(args.data)
Lv, Av, _ = load(args.val)
D, Z, Hd, K, H = meta['D'], args.latent, args.hidden, args.ensemble, args.horizon


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, Hd), nn.Tanh(), nn.Linear(Hd, Hd), nn.Tanh(), nn.Linear(Hd, o))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = mlp(D, Z), mlp(Z, D)
        self.trans = nn.ModuleList([mlp(Z + D, Z) for _ in range(K)])

    def step(self, k, z, a):
        return z + self.trans[k](torch.cat([z, a], -1))


m = Model()
opt = torch.optim.Adam(m.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
N, T = L.shape[0], L.shape[1] - 1


def batch(Ls, As, B):
    n = torch.randint(0, Ls.shape[0], (B,))
    t0 = torch.randint(0, Ls.shape[1] - 1 - H + 1, (B,))
    idx = t0[:, None] + torch.arange(H + 1)[None]
    l = Ls[n[:, None], idx]                     # [B, H+1, D]
    a = As[n[:, None], idx[:, :H]]              # [B, H, D]
    return l, a


def losses(l, a):
    z_all = m.enc(l)                            # [B, H+1, Z]
    rec = ((m.dec(z_all) - l) ** 2).mean()
    pred, cons = 0.0, 0.0
    for k in range(K):
        z = z_all[:, 0]
        for t in range(H):
            zn = m.step(k, z, a[:, t])
            pred = pred + ((m.dec(zn) - l[:, t + 1]) ** 2).mean() / H
            z = zn
        z1 = m.step(k, z_all[:, :-1].reshape(-1, Z), a.reshape(-1, D))
        cons = cons + ((z1 - z_all[:, 1:].reshape(-1, Z).detach()) ** 2).mean()
    return rec, pred / K, cons / K


t_start = time.time()
for it in range(1, args.steps + 1):
    l, a = batch(L, A, 256)
    rec, pred, cons = losses(l, a)
    loss = rec + pred + 0.5 * cons
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); sched.step()
    if it % 500 == 0 or it == 1:
        with torch.no_grad():
            lv, av = batch(Lv, Av, 1024)
            vr, vp, vc = losses(lv, av)
        print(f'step {it:5d}  train rec {rec.item():.4f} pred {pred.item():.4f} cons {cons.item():.4f}'
              f' | val rec {vr.item():.4f} pred {vp.item():.4f}  ({time.time() - t_start:.0f}s)', flush=True)

# ---- 估计潜空间噪声与集成分歧的尺度 ----
with torch.no_grad():
    l = L[:, :-1].reshape(-1, D); ln = L[:, 1:].reshape(-1, D); a = A.reshape(-1, D)
    z, zn = m.enc(l), m.enc(ln)
    preds = torch.stack([m.step(k, z, a) for k in range(K)])         # [K, M, Z]
    resid = zn - preds.mean(0)
    noise_std = resid.std(0)                                          # [Z]
    # 分歧的尺度要在"游戏里的用法"下校准：开环展开（均值 + 潜噪声 + 投影回编码器流形 z ← E(G(z))），
    # 而不是只看一步预测——展开时潜状态会漂离编码器产生过的区域。
    lv0, av0 = Lv[:, 0], Av
    z = m.enc(lv0); dis_list = []
    for t in range(av0.shape[1]):
        pk = torch.stack([m.step(k, z, av0[:, t]) for k in range(K)])
        dis_list.append((pk.std(0) / noise_std).mean(-1))
        z = pk.mean(0) + noise_std * torch.randn_like(z)
        z = m.enc(m.dec(z))
    dis_scale = float(torch.quantile(torch.cat(dis_list), 0.95))
    # 一步预测误差（可观测 logit 空间，用集成均值）
    one_step = float(((m.dec(preds.mean(0)) - ln) ** 2).mean())
    # 不施加噪声时数据本身的一步方差（作为下限参考）
    print(f'one-step logit MSE {one_step:.4f}; latent noise std {noise_std.numpy().round(3)}; '
          f'disagreement p95 {dis_scale:.3f}')


def export_linear(layer):
    return {'W': [[round(float(x), 5) for x in row] for row in layer.weight.detach().numpy()],
            'b': [round(float(x), 5) for x in layer.bias.detach().numpy()]}


def export_mlp(seq):
    return [export_linear(seq[i]) for i in (0, 2, 4)]


out = {
    'version': 1, 'obs': meta['obs'], 'omens': meta['omens'], 'latent': Z,
    'enc': export_mlp(m.enc), 'dec': export_mlp(m.dec), 'trans': [export_mlp(t) for t in m.trans],
    'noiseStd': [round(float(x), 5) for x in noise_std], 'disagreeScale': round(dis_scale, 5),
    'trainInfo': {'steps': args.steps, 'trajectories': int(N), 'years': int(T), 'oneStepLogitMSE': round(one_step, 5)},
}
with open(args.out, 'w') as f:
    f.write('// 由 model/train.py 生成的潜空间世界模型权重（勿手改）。\n')
    f.write('(function (root) { root.UNDERCURRENT_LEARNED = ')
    json.dump(out, f, separators=(',', ':'))
    f.write('; if (typeof module !== "undefined") module.exports = root.UNDERCURRENT_LEARNED; })'
            '(typeof window !== "undefined" ? window : globalThis);\n')
print('wrote', args.out, f'{os.path.getsize(args.out) / 1024:.0f} KB')
