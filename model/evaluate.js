// 对比规则引擎与学习引擎：用同样的种子、同样的策略，看学到的世界模型是否保留了规则世界的"性格"。
// 用法：node model/evaluate.js [weights=prototype/learned_weights.js] [seeds=12]
const path = require('path');
const Sim = require('../prototype/sim.js');
const weightsPath = path.resolve(process.argv[2] || path.join(__dirname, '../prototype/learned_weights.js'));
Sim.setLearnedWeights(require(weightsPath));
const SEEDS = +(process.argv[3] || 12);
const ids = Sim.OMENS.map((o) => o.id);
const fmt = (x) => x.toFixed(3);

function play(engine, seed, policy) {
  const w = Sim.createWorld(seed, { engine });
  const rng = Sim.mulberry32(seed * 7);
  let F = 0, E = 0, dis = 0, wish = 0;
  for (let era = 0; era < 12; era++) {
    if (policy) for (const [k, s, k2] of policy(w)) Sim.intervene(w, k, s, k2);
    for (let t = 0; t < 25; t++) { Sim.step(w, rng); F += Sim.flourishing(Sim.decode(w)); E += w.erosion; dis += w.disagreement || 0; }
    Sim.anticipate(w);
  }
  if (w.wishDone) wish = 1;
  return { F: F / 300, E: E / 300, dis: dis / 300, wish };
}
function greedy(w) {
  const acts = []; const ww = Sim.cloneWorld(w);
  for (let k = 0; k < 2; k++) {
    let best = null, bv = -1;
    for (const o of [null, ...ids]) {
      const v = Sim.prophecy(ww, { omen: o, samples: 8, seed: 3 }).reduce((a, r) => a + r.reduce((b, x) => b + Sim.flourishing(x), 0), 0);
      if (v > bv) { bv = v; best = o; }
    }
    if (!best) break; acts.push([best, 1]); Sim.intervene(ww, best, 1);
  }
  return acts;
}
const spam = () => [['ore', 1.6], ['rain', 1.6]];

console.log(`seeds: ${SEEDS}`);
for (const [name, pol] of [['不干预', null], ['贪心规划', greedy], ['强度1.6乱推', spam]]) {
  const row = {};
  for (const eng of ['rule', 'learned']) {
    const acc = { F: 0, E: 0, dis: 0, wish: 0 };
    for (let s = 1; s <= SEEDS; s++) { const r = play(eng, s, pol); for (const k in acc) acc[k] += r[k] / SEEDS; }
    row[eng] = acc;
  }
  console.log(`${name.padEnd(8)} 繁荣 rule ${fmt(row.rule.F)} / learned ${fmt(row.learned.F)} | 梦蚀 ${fmt(row.rule.E)} / ${fmt(row.learned.E)} | 模型分歧 ${fmt(row.learned.dis)} | 愿望实现 ${fmt(row.rule.wish)} / ${fmt(row.learned.wish)}`);
}

// 征兆的即时效果：同一个起始状态，降下征兆后过 1 年，与不降下相比各指标的平均变化
console.log('\n征兆效果（1 年后相对不干预的指标变化，rule → learned）');
for (const id of ids) {
  const delta = {};
  for (const eng of ['rule', 'learned']) {
    const d = Object.fromEntries(Sim.OBS.map((b) => [b.key, 0]));
    const n = 200;
    for (let s = 0; s < n; s++) {
      const base = Sim.createWorld(500 + s, { engine: eng });
      const pre = Sim.mulberry32(s); for (let t = 0; t < 30; t++) Sim.step(base, pre); // 走到不同状态
      const a = Sim.cloneWorld(base), b = Sim.cloneWorld(base);
      a.expect[id] = 0; b.expect[id] = 0;
      const ra = Sim.mulberry32(s + 99), rb = Sim.mulberry32(s + 99);
      Sim.intervene(a, id, 1); a.erosion = b.erosion; // 只看方向本身
      Sim.step(a, ra, 0); Sim.step(b, rb, 0);
      const oa = Sim.decode(a), ob = Sim.decode(b);
      for (const k in d) d[k] += (oa[k] - ob[k]) / n;
    }
    delta[eng] = d;
  }
  const line = Sim.OBS.map((b) => `${b.name} ${delta.rule[b.key] >= 0 ? '+' : ''}${delta.rule[b.key].toFixed(2)}→${delta.learned[b.key] >= 0 ? '+' : ''}${delta.learned[b.key].toFixed(2)}`).join('  ');
  console.log(`${Sim.omenName(id)}  ${line}`);
}
