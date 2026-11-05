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

  // ---------- 征兆合成 ----------
  // 两个已观测过的征兆可以合成一次干预：各以 0.6 倍同时推动。
  // 居民从未见过这种组合（预期 = 两者预期之积），所以很难被对冲，但容易引发恐慌；
  // 而"不自然的组合"让世界更快地离开流形（梦蚀 ×2）。
  const COMPOUND_SCALE = 0.6, COMPOUND_EROSION = 2;
  const COMPOUND_NAMES = {
    'comet+rain': '星雨', 'dream+rain': '雨梦', 'plague+rain': '瘴雨', 'ore+rain': '锈雨', 'hush+rain': '静雨',
    'comet+dream': '星谕', 'comet+plague': '赤雾', 'comet+ore': '陨铁', 'comet+hush': '无声之星',
    'dream+plague': '迷梦', 'dream+ore': '机巧之梦', 'dream+hush': '长眠',
    'ore+plague': '毒烟', 'hush+plague': '死寂', 'hush+ore': '沉睡的熔炉',
  };
  const compoundKey = (a, b) => [a, b].sort().join('+');

  // ---------- 反向征兆 ----------
  // 观测过 REVERSE_UNLOCK 次的征兆可以被"逆转"（−v）。键名形如 "~rain"。
  // 逆转从未在世界里自然出现过，所以居民毫无预期，而且更快地离开流形（梦蚀 ×1.3）。
  const REVERSE_UNLOCK = 2, REVERSE_EROSION = 1.3;
  const REVERSE_NAMES = { rain: '旱风', comet: '星隐', dream: '无梦之夜', plague: '春潮', ore: '地息', hush: '喧嚣' };
  const isReverse = (key) => key[0] === '~';
  const omenName = (key) => isReverse(key) ? REVERSE_NAMES[key.slice(1)]
    : COMPOUND_NAMES[key] || (OMENS.find((o) => o.id === key) || {}).name || key;

  // ---------- 锚定 ----------
  // 把一个可观测量钉在当前值上 ANCHOR_YEARS 年：每年把 z 沿该维度拉回。
  // 代价：一个纪元的全部注意力，以及持续的梦蚀（世界被迫违背自己的动力学）。
  // 梦蚀中解码器漂移，而锚定用的是原始逆映射——锚会"打滑"，并把扰动泄漏到其他维度。
  const ANCHOR_YEARS = 25, ANCHOR_EROSION = 0.009;

  // ---------- 世界愿望 ----------
  // 每个世界有一个隐藏的长期倾向：它会轻微地把世界拉向某个状态（pull，logit 空间/年），
  // 编年史会以越来越明确的方式暗示它。在愿望状态中连续保持 WISH_YEARS 年即视为实现。
  const WISH_YEARS = 30, WISH_PULL = 0.2;
  const WISHES = [
    { id: 'sea', name: '海', reveal: '这片土地想要海。',
      cond: (o) => o.rain > 0.75 && o.pop > 0.2, pull: { rain: 0.6 },
      hints: ['渔夫说，夜里能听见远方的潮声。', '孩子们在干涸的河床上画波浪。', '所有的井水都带着一丝咸味。人们梦见无边的水。'] },
    { id: 'fire', name: '火', reveal: '这片土地想要火与锻造。',
      cond: (o) => o.craft > 0.75 && o.strife < 0.35 && o.soil > 0.25, pull: { craft: 0.5 },
      hints: ['矿脉在地表露出，像是有意被人发现。', '雷击总是落在铁匠铺附近。', '石头里传来锤击声。大地在等待熔炉。'] },
    { id: 'god', name: '神', reveal: '这片土地想要被崇拜。',
      cond: (o) => o.faith > 0.72 && o.strife < 0.35 && o.pop > 0.25, pull: { faith: 0.5 },
      hints: ['山顶的石头被风雕成了人形。', '新生儿的第一个词总是同一个音节。', '大地本身在祈祷。它想听到回应。'] },
    { id: 'throng', name: '繁衍', reveal: '这片土地想要人群。',
      cond: (o) => o.pop > 0.68 && o.soil > 0.3, pull: { pop: 0.45, soil: 0.2 },
      hints: ['野花开在每一条小路旁边。', '荒地上自己长出了可以吃的根茎。', '土地向人张开怀抱。它想被填满。'] },
    { id: 'wild', name: '荒野', reveal: '这片土地想回到荒野。',
      cond: (o) => o.pop < 0.35 && o.soil > 0.5, pull: { soil: 0.8, pop: -0.4 },
      hints: ['田地的边缘，森林每年都向前推进一步。', '狼群在城墙外徘徊，却从不进攻。', '藤蔓爬满了空房子。大地想要安静地生长。'] },
    { id: 'peace', name: '寂静', reveal: '这片土地想要和平与沉默。',
      cond: (o) => o.strife < 0.06 && o.faith < 0.2 && o.pop > 0.45, pull: { strife: -0.5, faith: -0.3 },
      hints: ['武器在仓库里莫名生锈。', '争吵的人会突然忘记自己要说什么。', '风在战场上停住了。大地厌倦了喧哗。'] },
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
    const panicVec = matVec(Winv, OBS.map((ob) => ({ strife: 1, faith: 0.6, pop: -0.2 })[ob.key] || 0));
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
    // 梦蚀用的扭曲矩阵：两组固定随机矩阵，随年份缓慢旋转
    const warpA = [], warpB = [];
    for (let a = 0; a < D; a++) {
      warpA.push([]); warpB.push([]);
      for (let b = 0; b < D; b++) { warpA[a].push(gauss(rng)); warpB[a].push(gauss(rng)); }
    }
    const wish = WISHES[Math.floor(rng() * WISHES.length)];
    return {
      seed, Wd, Winv, panicVec, warpA, warpB, omenVecs, z,
      wish, wishYears: 0, wishDone: false, wishJustDone: false, inWish: false,
      year: 0, map: { W, H, height },
      erosion: 0,                 // 梦蚀：离开"训练流形"的程度 ∈ [0,1]
        expect: Object.fromEntries(OMENS.map((o) => [o.id, 0])), // 居民对每个征兆的预期 ∈ [0,1]
      eraUses: Object.fromEntries(OMENS.map((o) => [o.id, 0])), // 本纪元使用次数
      totalUses: Object.fromEntries(OMENS.map((o) => [o.id, 0])),
    };
  }

  function cloneWorld(w) {
    return { ...w, z: w.z.slice(), anchor: w.anchor && { ...w.anchor }, expect: { ...w.expect }, eraUses: { ...w.eraUses }, totalUses: { ...w.totalUses } };
  }

  // ---------- 梦蚀 ----------
  // 世界模型在分布外会"做梦"：解码器发生漂移（指标失去原有含义），
  // 而动力学仍按原来的逆映射推进——于是规则本身开始失灵。
  function warpedDecoder(world) {
    const e = world.erosion;
    if (e < 1e-3) return world.Wd;
    const t = world.year / 30, ca = Math.cos(t) * 0.55 * e, sb = Math.sin(t) * 0.55 * e;
    return world.Wd.map((row, a) => row.map((v, b) => v + ca * world.warpA[a][b] + sb * world.warpB[a][b]));
  }

  function decode(world, z = world.z) {
    const l = matVec(warpedDecoder(world), z);
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

  const EROSION_DECAY = 0.985; // 每年；半衰期约 46 年 ≈ 2 个纪元

  function step(world, rng, noise = 1) {
    const o = decode(world);
    const g = drift(o);
    const dt = 0.22;
    for (const k in world.wish.pull) g[k] += WISH_PULL * world.wish.pull[k] * (world.wishDone ? 0.3 : 1);
    const gl = OBS.map((ob) => g[ob.key] * dt);
    const dz = matVec(world.Winv, gl);
    const sigma = noise * 0.09 * (1 + 2.5 * world.erosion); // 梦蚀让未来更不确定
    for (let i = 0; i < D; i++) world.z[i] += dz[i] + sigma * gauss(rng);
    world.nightmare = null;
    // 噩梦：梦蚀超过 0.35 后，世界会随机地"梦到"某个征兆（方向随机），并伴随恐慌
    const pN = Math.max(0, world.erosion - 0.35) * 0.8;
    if (rng() < pN) {
      const om = OMENS[Math.floor(rng() * OMENS.length)];
      const sign = rng() < 0.5 ? -1 : 1, k = 1.1 * world.erosion;
      const v = world.omenVecs[om.id];
      for (let i = 0; i < D; i++) world.z[i] += sign * k * v[i] + 0.35 * k * world.panicVec[i];
      world.nightmare = { id: om.id, sign };
    }
    if (world.anchor && world.anchor.years > 0) {
      const k = OBS.findIndex((ob) => ob.key === world.anchor.key);
      const l = matVec(warpedDecoder(world), world.z)[k];
      const dl = logit(world.anchor.value) - l;
      for (let i = 0; i < D; i++) world.z[i] += world.Winv[i][k] * dl;
      world.erosion = Math.min(1, world.erosion + ANCHOR_EROSION);
      if (--world.anchor.years <= 0) world.anchor = null;
    }
    world.erosion *= EROSION_DECAY;
    world.year += 1;
    world.wishJustDone = false;
    world.inWish = world.wish.cond(decode(world));
    world.wishYears = world.inWish ? world.wishYears + 1 : 0; // 连续年数
    world.wishBest = Math.max(world.wishBest || 0, world.wishYears);
    if (world.inWish && !world.wishDone && world.wishYears >= WISH_YEARS) {
      world.wishDone = world.wishJustDone = true;
      world.erosion = 0; // 愿望实现：世界回到清醒，此后愿望的拉力减弱
    }
  }

  // ---------- 居民对你的预测 ----------
  // 学者的预测能力取决于技艺（文字、记录、观测），信仰过高时会被教义取代。
  function scholarAbility(o) {
    return Math.min(1, Math.max(0, 0.15 + 0.85 * o.craft - 0.35 * o.faith));
  }

  // 干预：返回本次干预的详细结果，供 UI 叙述
  //  - 梦蚀放大：已经离开流形的世界更"软"，同样的推动效果更大，但也更不可预测
  //  - 对冲：居民预期到的征兆，效果被削弱（他们早已做好准备）
  //  - 震惊：完全出乎预料的征兆会引发恐慌（纷争、信仰上升）
  function omenVec(world, key) {
    if (isReverse(key)) return world.omenVecs[key.slice(1)].map((x) => -x);
    if (!key.includes('+')) return world.omenVecs[key];
    const [a, b] = key.split('+');
    return world.omenVecs[a].map((x, i) => COMPOUND_SCALE * (x + world.omenVecs[b][i]));
  }

  // second 不为空时为合成征兆；omenId 以 "~" 开头时为反向征兆（不可合成）。
  function intervene(world, omenId, strength, second = null) {
    const o = decode(world);
    const ability = scholarAbility(o);
    const amp = 1 + 0.8 * world.erosion;
    const compound = !!second && second !== omenId && !isReverse(omenId) && !isReverse(second);
    const key = compound ? compoundKey(omenId, second) : omenId;
    // 合成征兆的预期：要么居民已经学会了这个组合本身，要么只能从两个成分去猜
    const E = compound ? Math.max(world.expect[omenId] * world.expect[second], world.expect[key] || 0) : world.expect[key] || 0;
    const hedge = Math.min(0.85, E * (0.35 + 0.65 * ability));
    const eff = strength * amp * (1 - hedge);
    const v = omenVec(world, key);
    for (let i = 0; i < D; i++) world.z[i] += eff * v[i];
    const surprise = 1 - E;
    const shock = 0.45 * surprise * surprise * strength * (0.3 + o.faith);
    if (shock > 0.01) for (let i = 0; i < D; i++) world.z[i] += shock * world.panicVec[i];
    const dE = 0.05 * strength * strength * amp * (compound ? COMPOUND_EROSION : isReverse(key) ? REVERSE_EROSION : 1);
    world.erosion = Math.min(1, world.erosion + dE);
    world.eraUses[key] = (world.eraUses[key] || 0) + 1;
    if (compound) { world.eraUses[omenId] += 0.5; world.eraUses[second] += 0.5; }
    world.totalUses[key] = (world.totalUses[key] || 0) + 1;
    return { key, compound, reverse: isReverse(key), amp, hedge, eff, shock, erosionGain: dE, ability };
  }

  // ---------- 居民的集体行动（来自 LLM 居民层） ----------
  // LLM 只能选择"哪个指标、往哪个方向"，幅度由这里固定并限幅——叙事可以影响世界，但不能接管世界。
  const RESIDENT_NUDGE = 0.25, RESIDENT_MAX_NUDGES = 2;
  function residentNudge(world, nudges) {
    const applied = [];
    for (const n of (nudges || []).slice(0, RESIDENT_MAX_NUDGES)) {
      const k = OBS.findIndex((ob) => ob.key === n.indicator);
      if (k < 0 || (n.direction !== 'up' && n.direction !== 'down')) continue;
      const d = n.direction === 'up' ? RESIDENT_NUDGE : -RESIDENT_NUDGE;
      for (let i = 0; i < D; i++) world.z[i] += world.Winv[i][k] * d;
      applied.push({ indicator: n.indicator, direction: n.direction });
    }
    return applied;
  }

  function anchor(world, key) {
    world.anchor = { key, value: Math.min(0.97, Math.max(0.03, decode(world)[key])), years: ANCHOR_YEARS };
    return world.anchor;
  }

  // 纪元结束：学者根据你过去的行为更新预期，并为"预期中的征兆"提前做准备。
  // 准备是一种反向推动——即使你这次没有降下它，准备也已经发生（可能白费）。
  function anticipate(world) {
    const o = decode(world);
    const ability = scholarAbility(o);
    const preps = [];
    // 基础征兆与用过的合成征兆一视同仁：用得越规律，越会被学会
    const keys = new Set([...OMENS.map((om) => om.id), ...Object.keys(world.eraUses), ...Object.keys(world.expect)]);
    for (const key of keys) {
      const used = Math.min(1, (world.eraUses[key] || 0) / 2);
      const cult = (world.totalUses[key] || 0) >= 5 ? 0.15 : 0; // 崇拜它的教派会额外努力预测
      world.expect[key] = Math.min(1, 0.55 * (world.expect[key] || 0) + (0.5 + cult) * used * (0.4 + 0.6 * ability));
      world.eraUses[key] = 0;
      const prep = world.expect[key] * ability;
      if (prep > 0.25) {
        const k = 0.22 * prep, v = omenVec(world, key);
        for (let i = 0; i < D; i++) world.z[i] -= k * v[i];
        preps.push({ id: key, level: prep });
      }
    }
    return { ability, preps };
  }

  // 愿望的暗示：越往后越明确（第 3 条几乎直说）
  function wishHint(world, era, rng) {
    const h = world.wish.hints;
    const tier = era < 4 ? 0 : era < 8 ? 1 : 2;
    return h[Math.max(0, tier - (rng() < 0.3 ? 1 : 0))];
  }

  // 繁荣度：既要人口，也要和平，也不能把一切押在单一维度上
  function flourishing(o) {
    const vals = OBS.map((ob) => o[ob.key]);
    const mean = vals.reduce((a, b) => a + b, 0) / D;
    const spread = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / D);
    return Math.max(0, o.pop * (1 - o.strife) * (0.6 + 0.4 * o.soil) * (1 - 0.6 * spread));
  }

  // 预言：从当前（或假设的）状态出发采样 N 条未来，返回分位数扇形
  function prophecy(world, { years = 25, samples = 32, omen = null, second = null, strength = 1, seed = 1 } = {}) {
    const rng = mulberry32(seed);
    const runs = [];
    for (let k = 0; k < samples; k++) {
      const w = cloneWorld(world);
      if (omen) intervene(w, omen, strength, second);
      const traj = [decode(w)];
      for (let t = 0; t < years; t++) { step(w, rng); traj.push(decode(w)); }
      runs.push(traj);
    }
    return runs;
  }

  root.Sim = { OBS, OMENS, WISHES, WISH_YEARS, COMPOUND_NAMES, REVERSE_NAMES, REVERSE_UNLOCK, ANCHOR_YEARS, isReverse, anchor, residentNudge, compoundKey, omenName, wishHint, createWorld, cloneWorld, decode, step, intervene, anticipate, scholarAbility,
    flourishing, prophecy, mulberry32, gauss };
  if (typeof module !== 'undefined') module.exports = root.Sim;
})(typeof window !== 'undefined' ? window : globalThis);
