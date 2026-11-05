// 用规则引擎（prototype/sim.js）生成训练轨迹，供 train.py 学习潜空间世界模型。
// 输出（二进制 float32，little-endian）：
//   <out>/logits.f32   [N, T+1, 6]  每年的可观测量 logit（裁剪到 ±6）
//   <out>/actions.f32  [N, T, 6]    当年年初施加的征兆强度（反向为负，合成为两项各 0.6）
//   <out>/meta.json
// 为了得到"干净"的动力学：不使用梦蚀、噩梦、居民对冲、世界愿望（这些由游戏层在潜空间外施加）。
// 用法：node model/gen_data.js <out_dir> [N=3000] [T=60] [seed=1]
const fs = require('fs');
const path = require('path');
const Sim = require('../prototype/sim.js');

const [outDir, N = 3000, T = 60, seed0 = 1] = [process.argv[2], ...process.argv.slice(3).map(Number)];
if (!outDir) { console.error('usage: node model/gen_data.js <out_dir> [N] [T] [seed]'); process.exit(1); }
fs.mkdirSync(outDir, { recursive: true });

const ids = Sim.OMENS.map((o) => o.id);
const D = Sim.OBS.length;
const rng = Sim.mulberry32(seed0 * 7919);
const NO_WISH = { pull: {}, cond: () => false };
const clipLogit = (p) => Math.max(-6, Math.min(6, Math.log(p / (1 - p))));
const logits = (w) => { const o = Sim.decode(w); return Sim.OBS.map((b) => clipLogit(o[b.key])); };

function randomAction() {
  const a = new Array(D).fill(0);
  if (rng() > 0.12) return a;
  const s = 0.4 + 1.2 * rng(), r = rng();
  const i = Math.floor(rng() * D);
  if (r < 0.6) a[i] = s;
  else if (r < 0.8) a[i] = -s;
  else { let j = Math.floor(rng() * D); if (j === i) j = (j + 1) % D; a[i] = 0.6 * s; a[j] = 0.6 * s; }
  return a;
}
function applyAction(w, a) {
  for (let k = 0; k < D; k++) if (a[k]) {
    const v = w.omenVecs[ids[k]];
    for (let i = 0; i < v.length; i++) w.z[i] += a[k] * v[i];
  }
}

const L = new Float32Array(N * (T + 1) * D), A = new Float32Array(N * T * D);
for (let n = 0; n < N; n++) {
  const w = Sim.createWorld(1000 + n + seed0 * 100000);
  w.wish = NO_WISH;
  const stepRng = Sim.mulberry32(n * 31 + seed0);
  const burn = Math.floor(rng() * 200); // 随机预热，覆盖更多状态
  for (let t = 0; t < burn; t++) { applyAction(w, randomAction()); Sim.step(w, stepRng); }
  for (let t = 0; t <= T; t++) {
    L.set(logits(w), (n * (T + 1) + t) * D);
    if (t === T) break;
    const a = randomAction();
    A.set(a, (n * T + t) * D);
    applyAction(w, a); Sim.step(w, stepRng);
  }
}
fs.writeFileSync(path.join(outDir, 'logits.f32'), Buffer.from(L.buffer));
fs.writeFileSync(path.join(outDir, 'actions.f32'), Buffer.from(A.buffer));
fs.writeFileSync(path.join(outDir, 'meta.json'), JSON.stringify({ N, T, D, omens: ids, obs: Sim.OBS.map((b) => b.key) }));
console.log(`wrote ${N} trajectories x ${T} years to ${outDir}`);
