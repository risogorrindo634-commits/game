// 潜流 · LLM 居民层（可选）
// 每个纪元结束时，把"居民能感知到的东西"交给 Claude，换回结构化的：
//   编年史条目、各派系对征兆的解读、居民对"守望者"的集体理论、以及至多 2 条集体行动。
// 集体行动只包含"指标 + 方向"，幅度由 sim.js 的 residentNudge 固定并限幅。
// API 密钥由玩家提供，只保存在本机浏览器，并且只发送给 api.anthropic.com。
// 普通脚本（非模块），以便直接双击 index.html 也能加载；SDK 在启用时才动态 import。
(function (root) {
  'use strict';

const SDK_URL = 'https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@1.1.0/+esm';
const MODEL = 'claude-opus-5-5';

const INDICATORS = ['rain', 'soil', 'pop', 'faith', 'strife', 'craft'];
const FACTIONS = ['祭司', '学者', '商团', '异端'];
const STANCES = ['崇拜', '研究', '利用', '反抗', '漠视'];

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['chronicle', 'factions', 'theory', 'nudges'],
  properties: {
    chronicle: { type: 'array', items: { type: 'string' } },
    factions: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['faction', 'stance', 'interpretation'],
        properties: {
          faction: { type: 'string', enum: FACTIONS },
          stance: { type: 'string', enum: STANCES },
          interpretation: { type: 'string' },
        },
      },
    },
    theory: { type: 'string' },
    nudges: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['indicator', 'direction', 'reason'],
        properties: {
          indicator: { type: 'string', enum: INDICATORS },
          direction: { type: 'string', enum: ['up', 'down'] },
          reason: { type: 'string' },
        },
      },
    },
  },
};

const SYSTEM = `你是一个模拟世界里全体居民的集体声音。这个世界由一个潜空间世界模型推进；一个看不见的"守望者"偶尔降下征兆。
居民看不到守望者，也看不到世界的真实状态，只能感受到天气、收成、人口、信仰、纷争、技艺的变化，以及征兆本身。

每个纪元（25 年）结束时，你会收到这一纪元的 JSON 摘要。请据此写出：
- chronicle：2–4 条编年史，每条不超过 50 个汉字，用史官的口吻，具体、有画面感，必须与指标的变化一致，不要编造摘要里没有的征兆。
- factions：祭司、学者、商团、异端中 2–4 个派系，各自的立场（崇拜/研究/利用/反抗/漠视）与对本纪元征兆的解读，每条不超过 60 个汉字。没有征兆时，解读他们对"沉默"的看法。
- theory：居民此刻对"守望者"的集体理论，一句话，不超过 60 个汉字。应当延续或修正之前的理论；守望者越有规律，理论越准确；越随机，越像迷信。
- nudges：0–2 条居民的集体行动带来的影响（如祭司修建神庙→信仰 up）。只能是对现状的合理反应，reason 不超过 30 个汉字。不确定时宁可留空。

指标取值 0–1。erosion（梦蚀）高时世界在"做梦"，居民会感到现实不稳定。scholarAbility 越高，学者越能识破守望者的规律。
全部用简体中文。`;

let clientPromise = null;
async function getClient(apiKey) {
  if (!clientPromise) {
    clientPromise = import(SDK_URL).then((m) => {
      const Anthropic = m.default || m.Anthropic;
      // 浏览器直连：密钥由玩家自己提供并只保存在本机，因此显式允许浏览器环境。
      return new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
    });
  }
  return clientPromise;
}
function resetClient() { clientPromise = null; }

const clip = (s, n) => String(s || '').replace(/\s+/g, ' ').trim().slice(0, n);

// 校验并裁剪模型输出：结构化输出保证了形状，这里再限制条数与长度
function sanitize(out) {
  if (!out || typeof out !== 'object') throw new Error('empty output');
  return {
    chronicle: (Array.isArray(out.chronicle) ? out.chronicle : []).slice(0, 4).map((t) => clip(t, 80)).filter(Boolean),
    factions: (Array.isArray(out.factions) ? out.factions : [])
      .filter((f) => FACTIONS.includes(f.faction) && STANCES.includes(f.stance)).slice(0, 4)
      .map((f) => ({ faction: f.faction, stance: f.stance, interpretation: clip(f.interpretation, 90) })),
    theory: clip(out.theory, 90),
    nudges: (Array.isArray(out.nudges) ? out.nudges : [])
      .filter((n) => INDICATORS.includes(n.indicator) && (n.direction === 'up' || n.direction === 'down')).slice(0, 2)
      .map((n) => ({ indicator: n.indicator, direction: n.direction, reason: clip(n.reason, 40) })),
  };
}

// context: 由 index.html 组装的纪元摘要（只含居民可感知的信息，不含世界愿望）
// client: 可注入（测试用）；否则用玩家的密钥创建
async function consult(context, { apiKey, client, signal } = {}) {
  const c = client || (await getClient(apiKey));
  const response = await c.beta.messages.create({
    model: MODEL,
    max_tokens: 16000,
    betas: ['server-side-fallback-2026-07-01'],
    fallbacks: 'default', // 安全分类器误拒时由服务端换模型重试
    output_config: { effort: 'low', format: { type: 'json_schema', schema: SCHEMA } },
    system: SYSTEM,
    messages: [{ role: 'user', content: JSON.stringify(context) }],
  }, { signal, timeout: 90_000 });
  if (response.stop_reason === 'refusal') throw new Error('refusal');
  if (response.stop_reason === 'max_tokens') throw new Error('truncated');
  const text = response.content.filter((b) => b.type === 'text').map((b) => b.text).join('');
  return sanitize(JSON.parse(text));
}

  root.Residents = { consult, resetClient, sanitize, MODEL, SCHEMA, SYSTEM };
  if (typeof module !== 'undefined') module.exports = root.Residents;
})(typeof window !== 'undefined' ? window : globalThis);
