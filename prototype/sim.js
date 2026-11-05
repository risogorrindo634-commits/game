// 潜流 · Undercurrent —— 原型模拟核心
// 用一个手写的小型"潜空间动力系统"代替真正学习得到的世界模型：
//   状态 z ∈ R^6 存在于潜空间；玩家看不到 z，只能看到解码器 decode(z) 的六个可观测量。
//   动力学在潜空间中推进，噪声也注入潜空间（因此噪声在可观测量上是纠缠的）。
//   玩家的干预 = 沿某个"征兆向量"推动 z。征兆的语义不是被告知的，而是被观察出来的。
(function (root) {
  'use strict';

  const OBS = [
    { key: 'rain',  name: '雨水', color: '#5aa9e6' },
    { key: 'soil',  name: '沃土', color: '#7cb518' },
    { key: 'pop',   name: '人口', color: '#f4d35e' },
    { key: 'faith', name: '信仰', color: '#c77dff' },
    { key: 'strife',name: '纷争', color: '#ef476f' },
    { key: 'craft', name: '技艺', color: '#adb5bd' },
  ];
  const D = OBS.length;

  // ---------- 小工具 ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const logit = (p) => Math.log(p / (1 - p));
  const matVec = (M, v) => M.map((row) => row.reduce((s, m, j) => s + m * v[j], 0));
  function invert(M) {
    const n = M.length;
    const A = M.map((row, i) => [...row, ...row.map((_, j) => (i === j ? 1 : 0))]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      [A[c], A[p]] = [A[p], A[c]];
      const d = A[c][c];
      for (let j = 0; j < 2 * n; j++) A[c][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = A[r][c];
        for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j];
      }
    }
    return A.map((row) => row.slice(n));
  }

  // ---------- 征兆（潜空间中的方向） ----------
  // effect 是该方向在"可观测 logit 空间"里的真实效果——这正是纠缠所在：
  // 没有任何一个征兆只影响一个量。玩家一开始只知道名字和一句含糊的谶语。
  const OMENS = [
    { id: 'rain',   name: '甘霖',   hint: '天空在哭泣，或在祝福。',
      effect: { rain: 1.6, soil: 0.2, faith: 0.35, strife: -0.25 } },
    { id: 'comet',  name: '彗星',   hint: '一道光划破夜空，人们抬头。',
      effect: { faith: 1.4, strife: 0.6, craft: -0.3 } },
    { id: 'dream',  name: '共梦',   hint: '某一夜，所有人做了同一个梦。',
      effect: { craft: 0.9, faith: 0.5, strife: -0.7, pop: -0.15 } },
    { id: 'plague', name: '灰雾',   hint: '雾从河谷升起，久久不散。',
      effect: { pop: -1.3, soil: 0.5, faith: 0.8, strife: -0.4 } },
    { id: 'ore',    name: '地鸣',   hint: '大地深处传来金属的回响。',
      effect: { craft: 1.5, strife: 0.7, rain: -0.4, soil: -0.3 } },
    { id: 'hush',   name: '静默',   hint: '风停了。鸟也停了。',
      effect: { strife: -1.2, faith: -0.6, pop: 0.3, craft: -0.4 } },
  ];

  function createWorld(seed) {
    const rng = mulberry32(seed);
    // 解码器：接近单位阵但被随机旋转/混合 —— 潜空间坐标本身没有可读语义
    const Wd = [];
    for (let i = 0; i < D; i++) {
      Wd.push([]);
      for (let j = 0; j < D; j++) Wd[i].push((i === j ? 1 : 0) + 0.35 * gauss(rng));
    }
    const Winv = invert(Wd);
    const omenVecs = {};
    for (const o of OMENS) {
      const e = OBS.map((ob) => o.effect[ob.key] || 0);
      omenVecs[o.id] = matVec(Winv, e); // 潜空间中的方向
    }
    const start = [0.55, 0.55, 0.3, 0.35, 0.2, 0.15];
    const z = matVec(Winv, start.map(logit));
    // 地形：静态的"可视化解码"种子
    const W = 48, H = 28;
    const height = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const nx = x / W, ny = y / H;
      let h = 0;
      for (let k = 1; k <= 4; k++) {
        const a = rng() * 6.28, b = rng() * 6.28;
        h += Math.sin(nx * k * 5 + a) * Math.cos(ny * k * 4 + b) / k;
      }
      const dx = nx - 0.5, dy = ny - 0.5;
      height.push(h * 0.5 + 0.6 - 1.6 * (dx * dx + dy * dy) + 0.15 * (rng() - 0.5));
    }
    return {
      seed, rngState: seed ^ 0x9e3779b9, Wd, Winv, omenVecs, z,
      year: 0, map: { W, H, height }, history: [], alive: true,
    };
  }

  function decode(world, z = world.z) {
    const l = matVec(world.Wd, z);
    const o = {};
    OBS.forEach((ob, i) => (o[ob.key] = sigmoid(l[i])));
    return o;
  }

  // 世界的"物理"：在可观测量上写因果规则，再拉回潜空间推进。
  // 真实游戏里这一层由学习到的潜空间转移模型 z_{t+1} = f(z_t, a_t) + ε 取代。
  function drift(o) {
    const { rain: r, soil: f, pop: p, faith: h, strife: s, craft: c } = o;
    return {
      rain:   0.5 * (0.55 - r) - 0.9 * c * c * p,
      soil:   1.0 * (r - 0.4) - 0.85 * p * (1 - 0.7 * c) + 0.5 * (0.55 - f),
      pop:    1.6 * (f + 0.35 * c - p - 0.1) - 1.4 * s * s,
      faith:  0.9 * s + 0.6 * Math.max(0, 0.4 - r) - 0.7 * c + 0.4 * (0.35 - h),
      strife: 1.4 * (p - f - 0.2 * c) + 0.9 * h * c + 0.5 * (0.2 - s) - 0.5 * (1 - p) * s,
      craft:  0.9 * p * (1 - s) - 0.35 + 0.25 * (1 - h) - 0.4 * (c - 0.6) * (c > 0.6),
    };
  }

  function step(world, rng, noise = 1) {
    const o = decode(world);
    const g = drift(o);
    const dt = 0.22;
    const gl = OBS.map((ob) => g[ob.key] * dt);
    const dz = matVec(world.Winv, gl);
    for (let i = 0; i < D; i++) world.z[i] += dz[i] + noise * 0.09 * gauss(rng);
    world.year += 1;
  }

  function intervene(world, omenId, strength) {
    const v = world.omenVecs[omenId];
    for (let i = 0; i < D; i++) world.z[i] += strength * v[i];
  }

  // 繁荣度：既要人口，也要和平，也不能把一切押在单一维度上
  function flourishing(o) {
    const vals = OBS.map((ob) => o[ob.key]);
    const mean = vals.reduce((a, b) => a + b, 0) / D;
    const spread = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / D);
    return Math.max(0, o.pop * (1 - o.strife) * (0.6 + 0.4 * o.soil) * (1 - 0.6 * spread));
  }

  // 预言：从当前（或假设的）状态出发采样 N 条未来，返回分位数扇形
  function prophecy(world, { years = 25, samples = 32, omen = null, strength = 1, seed = 1 } = {}) {
    const rng = mulberry32(seed);
    const runs = [];
    for (let k = 0; k < samples; k++) {
      const w = { ...world, z: world.z.slice() };
      if (omen) intervene(w, omen, strength);
      const traj = [decode(w)];
      for (let t = 0; t < years; t++) { step(w, rng); traj.push(decode(w)); }
      runs.push(traj);
    }
    return runs;
  }

  root.Sim = { OBS, OMENS, createWorld, decode, step, intervene, flourishing, prophecy, mulberry32, gauss };
  if (typeof module !== 'undefined') module.exports = root.Sim;
})(typeof window !== 'undefined' ? window : globalThis);
