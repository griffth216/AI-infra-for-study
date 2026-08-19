const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "AI Infra Study";
pres.title = "GPU存储层次：从硬件到算法的学习汇报";

// ============================================================
// Color Palette — Deep Tech
// ============================================================
const C = {
  bg:       "0A0E17",
  surface:  "131926",
  card:     "161C28",
  border:   "1E293B",
  text:     "E2E8F0",
  muted:    "8899AA",
  white:    "FFFFFF",
  accent:   "00D4AA",   // teal — highlights / success
  blue:     "4D9DE0",   // blue — secondary
  orange:   "F09B59",   // orange — warnings
  red:      "EF4444",   // red — problems / danger
  purple:   "A78BFA",   // purple — additional
};

// ============================================================
// Helpers
// ============================================================
const FONT = "Microsoft YaHei";
const FONT_BOLD = "Microsoft YaHei";

function makeShadow() {
  return { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.35 };
}

function addPageNum(slide, num) {
  slide.addText(String(num), {
    x: 9.2, y: 5.2, w: 0.6, h: 0.3,
    fontSize: 9, fontFace: FONT, color: C.muted, align: "right",
    margin: 0,
  });
}

function addSlideTitle(slide, title, subtitle) {
  // Thin accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.45, w: 0.06, h: 0.45,
    fill: { color: C.accent },
  });
  slide.addText(title, {
    x: 0.75, y: 0.35, w: 8.5, h: 0.55,
    fontSize: 26, fontFace: FONT, color: C.white, bold: true,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.75, y: 0.85, w: 8.5, h: 0.35,
      fontSize: 12, fontFace: FONT, color: C.muted,
      margin: 0,
    });
  }
}

function addCard(slide, x, y, w, h, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: color || C.card },
    shadow: makeShadow(),
    rectRadius: 0.06,
  });
}

function addTopAccent(slide, x, y, w, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: 0.04,
    fill: { color: color || C.accent },
  });
}

// ============================================================
// SLIDE 1 — Title
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Decorative top-right glow
  s.addShape(pres.shapes.OVAL, {
    x: 6.5, y: -1.5, w: 5, h: 5,
    fill: { color: C.accent, transparency: 94 },
  });
  // Decorative bottom-left glow
  s.addShape(pres.shapes.OVAL, {
    x: -1.5, y: 3.5, w: 4, h: 4,
    fill: { color: C.blue, transparency: 94 },
  });

  // Accent line
  s.addShape(pres.shapes.RECTANGLE, {
    x: 1.2, y: 1.6, w: 1.2, h: 0.05,
    fill: { color: C.accent },
  });

  s.addText("GPU 存储层次", {
    x: 1.2, y: 1.8, w: 8, h: 1.0,
    fontSize: 44, fontFace: FONT, color: C.white, bold: true,
    margin: 0,
  });
  s.addText("从硬件到算法的学习汇报  ·  Week 2", {
    x: 1.2, y: 2.8, w: 8, h: 0.6,
    fontSize: 20, fontFace: FONT, color: C.accent,
    margin: 0,
  });
  s.addText("AI Infra 学习计划  |  初学者学习汇报", {
    x: 1.2, y: 3.6, w: 8, h: 0.4,
    fontSize: 13, fontFace: FONT, color: C.muted,
    margin: 0,
  });

  // Bottom info bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.0, w: 10, h: 0.625,
    fill: { color: C.surface },
  });
  s.addText("2026.06  |  AI Infra Study Group", {
    x: 1.2, y: 5.05, w: 8, h: 0.5,
    fontSize: 11, fontFace: FONT, color: C.muted,
    margin: 0,
  });
}

// ============================================================
// SLIDE 2 — Agenda
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "汇报大纲", "16 分钟  ·  四个部分  ·  从学习到思考");

  const items = [
    { num: "01", title: "这周学了什么", desc: "GPU 四层存储金字塔\nHBM 与 SRAM 深度解析", time: "~4 min", color: C.accent },
    { num: "02", title: "发现了什么问题", desc: "内存墙 · SRAM 太小\nHBM 容量涨不过模型规模", time: "~4 min", color: C.blue },
    { num: "03", title: "怎么解决的", desc: "FlashAttention 的 IO-Aware 思想\nblock_size 的硬件约束推演", time: "~4 min", color: C.purple },
    { num: "04", title: "当前挑战与思考", desc: "SRAM 瓶颈 · HBM 带宽瓶颈\nTMA · HBM4 · PIM · 低精度", time: "~4 min", color: C.orange },
  ];

  items.forEach((item, i) => {
    const x = 0.4 + i * 2.35;
    const y = 1.6;
    const w = 2.15;
    const h = 3.2;
    addCard(s, x, y, w, h);
    addTopAccent(s, x, y, w, item.color);

    s.addText(item.num, {
      x: x + 0.25, y: y + 0.2, w: 0.8, h: 0.7,
      fontSize: 36, fontFace: FONT, color: item.color, bold: true,
      margin: 0,
    });
    s.addText(item.title, {
      x: x + 0.25, y: y + 0.95, w: w - 0.5, h: 0.5,
      fontSize: 15, fontFace: FONT, color: C.white, bold: true,
      margin: 0,
    });
    s.addText(item.desc, {
      x: x + 0.25, y: y + 1.55, w: w - 0.5, h: 1.0,
      fontSize: 11, fontFace: FONT, color: C.muted,
      margin: 0, lineSpacingMultiple: 1.6,
    });
    // Time badge
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.25, y: y + h - 0.55, w: 0.95, h: 0.3,
      fill: { color: item.color, transparency: 80 },
      rectRadius: 0.15,
    });
    s.addText(item.time, {
      x: x + 0.25, y: y + h - 0.55, w: 0.95, h: 0.3,
      fontSize: 10, fontFace: FONT, color: item.color, align: "center", valign: "middle",
      margin: 0,
    });
  });

  addPageNum(s, 2);
}

// ============================================================
// SLIDE 3 — 四层金字塔
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "GPU 存储系统：四层金字塔", "越往上越快但越小，越往下越大但越慢 —— 这是物理约束，不是设计偏好");

  // Pyramid — stacked horizontal bars (bottom = widest = HBM)
  const layers = [
    { name: "寄存器 (Register)",   cap: "256 KB / SM",     lat: "~0 cycle",    analogy: "工人手心里的纸条",     color: C.accent,  width: 2.8, yOff: 1.7,  h: 0.55  },
    { name: "共享内存 / SRAM",     cap: "192 KB / SM",     lat: "~20 cycles",  analogy: "小组眼前的工作台",       color: C.blue,   width: 4.5, yOff: 2.35, h: 0.55  },
    { name: "L2 Cache",             cap: "40 MB / 全卡",    lat: "~200 cycles", analogy: "车间公共货架",           color: C.purple, width: 6.2, yOff: 3.0,  h: 0.55  },
    { name: "HBM / 显存",          cap: "80 GB / 全卡",    lat: "~600 cycles", analogy: "园区外的大仓库",         color: C.orange, width: 7.9, yOff: 3.65, h: 0.55  },
  ];

  layers.forEach((l) => {
    const x = (10 - l.width) / 2;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: l.yOff, w: l.width, h: l.h,
      fill: { color: l.color, transparency: 80 },
      line: { color: l.color, width: 1.2 },
      rectRadius: 0.06,
    });
    // Layer name + analogy
    s.addText(l.name, {
      x: x + 0.2, y: l.yOff + 0.05, w: l.width * 0.5, h: 0.25,
      fontSize: 12, fontFace: FONT, color: l.color, bold: true,
      margin: 0,
    });
    s.addText(l.analogy, {
      x: x + 0.2, y: l.yOff + 0.28, w: l.width * 0.5, h: 0.22,
      fontSize: 9, fontFace: FONT, color: C.muted,
      margin: 0,
    });
    // Capacity + Latency on right
    s.addText(l.cap, {
      x: x + l.width * 0.52, y: l.yOff + 0.05, w: l.width * 0.45, h: 0.25,
      fontSize: 10, fontFace: FONT, color: C.white, align: "right",
      margin: 0,
    });
    s.addText(l.lat, {
      x: x + l.width * 0.52, y: l.yOff + 0.28, w: l.width * 0.45, h: 0.22,
      fontSize: 9, fontFace: FONT, color: C.muted, align: "right",
      margin: 0,
    });
  });

  // Arrow labels
  for (let i = 0; i < layers.length - 1; i++) {
    s.addText("▼ 容量 ×200  延迟 ×30", {
      x: 3.5, y: layers[i].yOff + layers[i].h + 0.02, w: 3, h: 0.18,
      fontSize: 7, fontFace: FONT, color: C.muted, align: "center",
      margin: 0,
    });
  }

  // Bottom insight box
  addCard(s, 0.5, 4.45, 9.0, 0.55, C.card);
  s.addText([
    { text: "💡 ", options: { fontSize: 12 } },
    { text: "SRAM 一个 bit 的芯片面积是 DRAM 的 5-7 倍。如果把 80GB 全换成 SRAM，芯片面积需要 16000mm²，一片晶圆只能切三四颗——物理上不可行。", options: { fontSize: 10, color: C.muted } },
  ], {
    x: 0.7, y: 4.48, w: 8.6, h: 0.5,
    fontFace: FONT, color: C.muted, valign: "middle",
    margin: 0,
  });

  addPageNum(s, 3);
}

// ============================================================
// SLIDE 4 — HBM vs GDDR
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "HBM：靠「加车道」而不是「踩油门」提速", "传统 GDDR vs HBM 的架构对比");

  // GDDR card (left)
  const leftX = 0.4, cardY = 1.5, cardW = 4.3, cardH = 3.5;
  addCard(s, leftX, cardY, cardW, cardH);
  addTopAccent(s, leftX, cardY, cardW, C.orange);

  s.addText("GDDR — 传统显存", {
    x: leftX + 0.25, y: cardY + 0.2, w: cardW - 0.5, h: 0.35,
    fontSize: 16, fontFace: FONT, color: C.orange, bold: true, margin: 0,
  });
  s.addText("「平房」— 内存颗粒平铺在 PCB 板上", {
    x: leftX + 0.25, y: cardY + 0.55, w: cardW - 0.5, h: 0.3,
    fontSize: 11, fontFace: FONT, color: C.muted, margin: 0,
  });

  const gddrSpecs = [
    { label: "位宽", val: "32-64 bit" },
    { label: "提速方式", val: "提高频率 → 功耗爆炸" },
    { label: "典型带宽", val: "272 GB/s (RTX 4060)" },
    { label: "价格", val: "消费级，整机 ~¥7000" },
    { label: "封装", val: "PCB 平面布局" },
  ];
  gddrSpecs.forEach((spec, i) => {
    const yy = cardY + 1.05 + i * 0.42;
    s.addText(spec.label, {
      x: leftX + 0.25, y: yy, w: 1.2, h: 0.35,
      fontSize: 11, fontFace: FONT, color: C.white, bold: true, margin: 0,
    });
    s.addText(spec.val, {
      x: leftX + 1.5, y: yy, w: cardW - 1.8, h: 0.35,
      fontSize: 11, fontFace: FONT, color: C.muted, margin: 0,
    });
  });

  // HBM card (right)
  const rightX = 5.3;
  addCard(s, rightX, cardY, cardW, cardH);
  addTopAccent(s, rightX, cardY, cardW, C.accent);

  s.addText("HBM — 高带宽内存", {
    x: rightX + 0.25, y: cardY + 0.2, w: cardW - 0.5, h: 0.35,
    fontSize: 16, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });
  s.addText("「摩天大楼」— 多层 DRAM 垂直堆叠 + TSV 贯通", {
    x: rightX + 0.25, y: cardY + 0.55, w: cardW - 0.5, h: 0.3,
    fontSize: 11, fontFace: FONT, color: C.muted, margin: 0,
  });

  const hbmSpecs = [
    { label: "位宽", val: "1024 bit (HBM3e)" },
    { label: "提速方式", val: "增加并行通道，低频低压" },
    { label: "典型带宽", val: "2.0 TB/s (A100)" },
    { label: "价格", val: "数据中心级，单卡 ~¥100,000" },
    { label: "封装", val: "3D 堆叠 + 硅中介层" },
  ];
  hbmSpecs.forEach((spec, i) => {
    const yy = cardY + 1.05 + i * 0.42;
    s.addText(spec.label, {
      x: rightX + 0.25, y: yy, w: 1.2, h: 0.35,
      fontSize: 11, fontFace: FONT, color: C.white, bold: true, margin: 0,
    });
    s.addText(spec.val, {
      x: rightX + 1.5, y: yy, w: cardW - 1.8, h: 0.35,
      fontSize: 11, fontFace: FONT, color: C.muted, margin: 0,
    });
  });

  // Bottom insight
  addCard(s, 0.4, 5.1, 9.2, 0.4, C.surface);
  s.addText("关键洞察：HBM 不靠提高频率，靠增加并行通道。代价是贵——数据中心卡用 HBM，消费级用 GDDR。但「内存墙」只是被推远，核心矛盾并未消失。", {
    x: 0.6, y: 5.12, w: 8.8, h: 0.35,
    fontSize: 10, fontFace: FONT, color: C.muted, valign: "middle", margin: 0,
  });

  addPageNum(s, 4);
}

// ============================================================
// SLIDE 5 — SRAM Deep Dive
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "SRAM：程序员手动管理的工作台", "三层核心用途 + Bank Conflict 详解");

  // Left: Three uses
  const uses = [
    { title: "① 分块计算 (Tiling)", desc: "把大矩阵切成 Tile，每次只搬一小块到 SRAM 计算。HBM 访问从 O(N³) 降到 O(N³/Tile)。节省 30-100× 带宽。" },
    { title: "② 线程间通信 + 同步", desc: "同一 Block 内的线程通过 SRAM 交换数据。__syncthreads() 确保数据一致性。没有它 → 数据竞争。" },
    { title: "③ 数据重组 / 挽救非合并访问", desc: "乱序数据先搬进 SRAM 重新排列，再用合并访问写回 HBM。SRAM 随机访问快，不经过 cache line。" },
  ];

  uses.forEach((u, i) => {
    const yy = 1.5 + i * 1.1;
    addCard(s, 0.4, yy, 4.8, 1.0);
    s.addText(u.title, {
      x: 0.6, y: yy + 0.1, w: 4.4, h: 0.3,
      fontSize: 13, fontFace: FONT, color: C.accent, bold: true, margin: 0,
    });
    s.addText(u.desc, {
      x: 0.6, y: yy + 0.42, w: 4.4, h: 0.5,
      fontSize: 10, fontFace: FONT, color: C.muted, margin: 0, lineSpacingMultiple: 1.4,
    });
  });

  // Right: Bank Conflict
  addCard(s, 5.5, 1.5, 4.1, 3.3);
  addTopAccent(s, 5.5, 1.5, 4.1, C.red);
  s.addText("⚠ Bank Conflict", {
    x: 5.75, y: 1.65, w: 3.6, h: 0.35,
    fontSize: 15, fontFace: FONT, color: C.red, bold: true, margin: 0,
  });
  s.addText("SRAM 硬件上被分为 32 个 Bank。多个线程同时访问同一 Bank → 排队串行。", {
    x: 5.75, y: 2.05, w: 3.6, h: 0.45,
    fontSize: 10, fontFace: FONT, color: C.muted, margin: 0, lineSpacingMultiple: 1.4,
  });

  // Bank Conflict example
  s.addText("列访问 → 32 路冲突：", {
    x: 5.75, y: 2.55, w: 3.6, h: 0.25,
    fontSize: 10, fontFace: FONT, color: C.orange, bold: true, margin: 0,
  });
  s.addText("__shared__ float tile[32][32];\nval = tile[threadIdx.x][col];\n→ 全部 32 线程命中同一 Bank！", {
    x: 5.75, y: 2.8, w: 3.6, h: 0.7,
    fontSize: 9, fontFace: "Consolas", color: C.red, margin: 0, lineSpacingMultiple: 1.5,
  });

  // Padding fix
  s.addText("Padding 解法：", {
    x: 5.75, y: 3.55, w: 3.6, h: 0.25,
    fontSize: 10, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });
  s.addText("__shared__ float tile[32][32+1];\n// 每行多一列，让行间 Bank 号错开\n// 性能改善 5%-35%", {
    x: 5.75, y: 3.8, w: 3.6, h: 0.65,
    fontSize: 9, fontFace: "Consolas", color: C.accent, margin: 0, lineSpacingMultiple: 1.5,
  });

  addPageNum(s, 5);
}

// ============================================================
// SLIDE 6 — Hardware Trends
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "硬件趋势：金字塔结构不变，比例在微调", "Ampere → Hopper → Blackwell 三代演进");

  // Stats row
  const stats = [
    { num: "+25-33%", label: "SRAM 每代增长", sub: "A100 192KB → H100 256KB/SM", color: C.accent },
    { num: "×4", label: "HBM 带宽增长", sub: "A100 2TB/s → B200 8TB/s", color: C.blue },
    { num: "×7", label: "算力增长 (同期)", sub: "312 → 2250 TFLOPS", color: C.orange },
    { num: "×60", label: "模型规模增长", sub: "GPT-3 175B → GPT-4 ~1.7T", color: C.red },
  ];

  stats.forEach((st, i) => {
    const x = 0.4 + i * 2.35;
    addCard(s, x, 1.5, 2.15, 1.8);
    addTopAccent(s, x, 1.5, 2.15, st.color);
    s.addText(st.num, {
      x: x + 0.15, y: 1.65, w: 1.85, h: 0.55,
      fontSize: 28, fontFace: FONT, color: st.color, bold: true, align: "center", margin: 0,
    });
    s.addText(st.label, {
      x: x + 0.15, y: 2.25, w: 1.85, h: 0.35,
      fontSize: 12, fontFace: FONT, color: C.white, bold: true, align: "center", margin: 0,
    });
    s.addText(st.sub, {
      x: x + 0.15, y: 2.6, w: 1.85, h: 0.5,
      fontSize: 9, fontFace: FONT, color: C.muted, align: "center", margin: 0, lineSpacingMultiple: 1.3,
    });
  });

  // Key insight card
  addCard(s, 0.4, 3.6, 9.2, 1.7);
  addTopAccent(s, 0.4, 3.6, 9.2, C.purple);
  s.addText("核心矛盾：算力涨得比带宽快，差距在拉大", {
    x: 0.65, y: 3.75, w: 8.7, h: 0.35,
    fontSize: 15, fontFace: FONT, color: C.white, bold: true, margin: 0,
  });

  const trendData = [
    ["", "A100 (2020)", "H100 (2022)", "B200 (2025)", "增长倍数"],
    ["FP16 算力", "312 TFLOPS", "989 TFLOPS", "2250 TFLOPS", "×7.2"],
    ["HBM 带宽", "2.0 TB/s", "3.35 TB/s", "8.0 TB/s", "×4.0"],
    ["算力/带宽比", "156 FLOPS/Byte", "295 FLOPS/Byte", "281 FLOPS/Byte", "矛盾加剧"],
  ];

  s.addTable(trendData, {
    x: 0.65, y: 4.15, w: 8.7, h: 1.0,
    fontFace: FONT, fontSize: 10,
    color: C.text,
    border: { pt: 0.5, color: C.border },
    fill: { color: C.card },
    rowH: [0.25, 0.25, 0.25, 0.25],
    colW: [1.8, 1.8, 1.8, 1.8, 1.5],
    autoPage: false,
  });

  addPageNum(s, 6);
}

// ============================================================
// SLIDE 7 — Memory Wall
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "问题一：算力过剩，带宽不足 ——「内存墙」", "以 RTX 4060 为例，量化算力与带宽的剪刀差");

  // Big stat
  addCard(s, 0.4, 1.5, 4.2, 2.2);
  s.addText("889", {
    x: 0.6, y: 1.6, w: 3.8, h: 0.9,
    fontSize: 60, fontFace: FONT, color: C.orange, bold: true, align: "center", margin: 0,
  });
  s.addText("FLOPS / Byte", {
    x: 0.6, y: 2.5, w: 3.8, h: 0.4,
    fontSize: 18, fontFace: FONT, color: C.white, align: "center", margin: 0,
  });
  s.addText("每从 HBM 读 1 个数字，需要做 889 次运算才能喂饱 GPU", {
    x: 0.6, y: 2.95, w: 3.8, h: 0.5,
    fontSize: 11, fontFace: FONT, color: C.muted, align: "center", margin: 0, lineSpacingMultiple: 1.3,
  });

  // Calculation
  addCard(s, 5.0, 1.5, 4.6, 2.2);
  s.addText("推演过程", {
    x: 5.2, y: 1.6, w: 4.2, h: 0.3,
    fontSize: 14, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const calcLines = [
    "GPU 算力 (FP16) = 121 TFLOPS = 121 万亿次/秒",
    "HBM 带宽 (GDDR6) = 272 GB/s = 2720 亿字节/秒",
    "每个 FP16 = 2 字节 → 每秒最多搬 1360 亿个数字",
    "",
    "121 万亿 ÷ 1360 亿 = 889 FLOPS/Byte",
    "",
    "但 LLM Decode 阶段：1-2 FLOPS/Byte",
    "→ GPU 实际算力利用率 < 0.2%",
  ];

  s.addText(calcLines.map((line, i) => ({
    text: line,
    options: {
      fontSize: 9.5,
      color: line.startsWith("→") ? C.red : (line === "" ? C.muted : C.muted),
      bold: line.startsWith("→"),
      breakLine: i < calcLines.length - 1,
    },
  })), {
    x: 5.2, y: 2.0, w: 4.2, h: 1.6,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.25,
  });

  // Cross-GPU comparison
  addCard(s, 0.4, 3.95, 9.2, 1.45);
  s.addText("这不是 RTX 4060 的问题——所有 GPU 都面临同样的矛盾", {
    x: 0.6, y: 4.05, w: 8.8, h: 0.3,
    fontSize: 13, fontFace: FONT, color: C.white, bold: true, margin: 0,
  });

  const cmpData = [
    ["", "RTX 4060", "A100", "H100", "B200"],
    ["算力 (TFLOPS)", "121", "312", "989", "2,250"],
    ["带宽 (TB/s)", "0.27", "2.0", "3.35", "8.0"],
    ["FLOPS/Byte", "445", "156", "295", "281"],
    ["Decode 实际需求", "1~2", "1~2", "1~2", "1~2"],
  ];

  s.addTable(cmpData, {
    x: 0.6, y: 4.35, w: 8.8, h: 0.9,
    fontFace: FONT, fontSize: 9,
    color: C.text,
    border: { pt: 0.5, color: C.border },
    fill: { color: C.surface },
    rowH: [0.2, 0.18, 0.18, 0.18, 0.18],
    colW: [2.0, 1.7, 1.7, 1.7, 1.7],
    autoPage: false,
  });

  addPageNum(s, 7);
}

// ============================================================
// SLIDE 8 — SRAM too small + Model growth
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "问题二 & 三：SRAM 太小 + 显存涨不过模型", "三个问题的共同根源：数据和算力不在同一个地方");

  // Problem 2: SRAM too small
  addCard(s, 0.4, 1.5, 4.3, 2.6);
  addTopAccent(s, 0.4, 1.5, 4.3, C.orange);
  s.addText("问题二：SRAM 太小", {
    x: 0.6, y: 1.6, w: 3.9, h: 0.35,
    fontSize: 16, fontFace: FONT, color: C.orange, bold: true, margin: 0,
  });

  s.addText("A100 SRAM:     ~20 MB", {
    x: 0.6, y: 2.05, w: 3.9, h: 0.35,
    fontSize: 18, fontFace: FONT, color: C.white, bold: true, margin: 0,
  });
  s.addText("A100 HBM:      80 GB", {
    x: 0.6, y: 2.4, w: 3.9, h: 0.35,
    fontSize: 18, fontFace: FONT, color: C.white, bold: true, margin: 0,
  });

  s.addText("工作台只有大仓库的", {
    x: 0.6, y: 2.85, w: 3.9, h: 0.3,
    fontSize: 12, fontFace: FONT, color: C.muted, margin: 0,
  });
  s.addText("0.025%", {
    x: 0.6, y: 3.1, w: 3.9, h: 0.5,
    fontSize: 40, fontFace: FONT, color: C.red, bold: true, margin: 0,
  });
  s.addText("但所有计算都必须在工作台上完成", {
    x: 0.6, y: 3.65, w: 3.9, h: 0.3,
    fontSize: 11, fontFace: FONT, color: C.muted, margin: 0,
  });

  // Problem 3: Model growth
  addCard(s, 5.3, 1.5, 4.3, 2.6);
  addTopAccent(s, 5.3, 1.5, 4.3, C.red);
  s.addText("问题三：HBM 容量涨不过模型规模", {
    x: 5.5, y: 1.6, w: 3.9, h: 0.35,
    fontSize: 14, fontFace: FONT, color: C.red, bold: true, margin: 0,
  });

  const growthData = [
    ["年份", "显存", "模型规模"],
    ["2020 (A100)", "80 GB", "GPT-3: 175B"],
    ["2024 (H200)", "141 GB", "GPT-4: ~1.7T"],
    ["2025 (B200)", "192 GB", "下一代: ~10T+"],
  ];

  s.addTable(growthData, {
    x: 5.5, y: 2.1, w: 3.9, h: 0.95,
    fontFace: FONT, fontSize: 10,
    color: C.text,
    border: { pt: 0.5, color: C.border },
    fill: { color: C.surface },
    rowH: [0.22, 0.22, 0.22, 0.22],
    colW: [1.3, 1.3, 1.3],
    autoPage: false,
  });

  // Growth comparison stats
  s.addText("×2.4", {
    x: 5.5, y: 3.2, w: 1.95, h: 0.5,
    fontSize: 32, fontFace: FONT, color: C.blue, bold: true, align: "center", margin: 0,
  });
  s.addText("显存增长", {
    x: 5.5, y: 3.65, w: 1.95, h: 0.3,
    fontSize: 10, fontFace: FONT, color: C.muted, align: "center", margin: 0,
  });

  s.addText("×60", {
    x: 7.45, y: 3.2, w: 1.95, h: 0.5,
    fontSize: 32, fontFace: FONT, color: C.red, bold: true, align: "center", margin: 0,
  });
  s.addText("模型增长", {
    x: 7.45, y: 3.65, w: 1.95, h: 0.3,
    fontSize: 10, fontFace: FONT, color: C.muted, align: "center", margin: 0,
  });

  // Common root cause
  addCard(s, 0.4, 4.35, 9.2, 1.05);
  addTopAccent(s, 0.4, 4.35, 9.2, C.purple);
  s.addText("共同根源", {
    x: 0.6, y: 4.45, w: 1.5, h: 0.3,
    fontSize: 13, fontFace: FONT, color: C.purple, bold: true, margin: 0,
  });
  s.addText([
    { text: "数据和算力不在同一个地方。", options: { bold: true, fontSize: 13, color: C.white } },
    { text: "  数据在 HBM（慢但大），算力在 SM（快但只能用小数据）。", options: { fontSize: 12, color: C.muted } },
    { text: " 搬数据的时间 ≫ 算数据的时间。", options: { bold: true, fontSize: 12, color: C.orange, breakLine: true } },
    { text: "→ 核心优化思路：既然搬数据是瓶颈，那就少搬。", options: { fontSize: 11, color: C.accent } },
  ], {
    x: 0.6, y: 4.75, w: 8.8, h: 0.55,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.3,
  });

  addPageNum(s, 8);
}

// ============================================================
// SLIDE 9 — Bridge: Week1→Week2
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "回到 Week 1：用硬件知识重新理解 FlashAttention 代码", "每一行代码对应一个硬件操作");

  // Mapping table
  const mapData = [
    [
      { text: "Week1 代码", options: { bold: true, color: C.white, fontSize: 10 } },
      { text: "硬件位置", options: { bold: true, color: C.white, fontSize: 10 } },
      { text: "具体物理过程", options: { bold: true, color: C.white, fontSize: 10 } },
    ],
    [
      { text: "Qi = Q[i_start:i_end]", options: { fontSize: 9, fontFace: "Consolas", color: C.accent } },
      { text: "HBM → L2 → SRAM", options: { fontSize: 10, color: C.blue } },
      { text: "从 HBM 读取 Q 块。L2 命中=200cyc, Miss=600cyc", options: { fontSize: 10, color: C.muted } },
    ],
    [
      { text: "S = Qi @ Kj.T", options: { fontSize: 9, fontFace: "Consolas", color: C.accent } },
      { text: "SRAM + 寄存器", options: { fontSize: 10, color: C.blue } },
      { text: "Tensor Core 执行 MMA，数据全在 SRAM 中", options: { fontSize: 10, color: C.muted } },
    ],
    [
      { text: "m_new = max(m, ...)", options: { fontSize: 9, fontFace: "Consolas", color: C.accent } },
      { text: "寄存器", options: { fontSize: 10, color: C.blue } },
      { text: "Online Softmax 状态更新，0 cycle 延迟", options: { fontSize: 10, color: C.muted } },
    ],
    [
      { text: "\"阅后即焚\" 注释", options: { fontSize: 9, fontFace: "Consolas", color: C.orange } },
      { text: "SRAM 被覆盖", options: { fontSize: 10, color: C.orange } },
      { text: "下一个 K,V 块直接覆盖同一块 SRAM 空间", options: { fontSize: 10, color: C.muted } },
    ],
    [
      { text: "O[i_start:i_end] = O/l", options: { fontSize: 9, fontFace: "Consolas", color: C.accent } },
      { text: "SRAM → HBM", options: { fontSize: 10, color: C.blue } },
      { text: "唯一一次将最终结果写回 HBM（不是中间结果！）", options: { fontSize: 10, color: C.muted } },
    ],
  ];

  s.addTable(mapData, {
    x: 0.4, y: 1.5, w: 9.2, h: 2.2,
    fontFace: FONT, fontSize: 10,
    color: C.text,
    border: { pt: 0.5, color: C.border },
    fill: { color: C.card },
    rowH: [0.28, 0.32, 0.32, 0.32, 0.32, 0.32],
    colW: [2.7, 1.7, 4.8],
    autoPage: false,
  });

  // Insight box
  addCard(s, 0.4, 3.95, 9.2, 1.45);
  addTopAccent(s, 0.4, 3.95, 9.2, C.accent);
  s.addText("关键理解", {
    x: 0.6, y: 4.05, w: 8.8, h: 0.3,
    fontSize: 14, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });
  s.addText([
    { text: "Week 1 反复提到的「从 HBM 搬进 SRAM」「阅后即焚」这些概念，", options: { fontSize: 12, color: C.muted } },
    { text: "学完存储层次之后完全理解了。", options: { fontSize: 12, color: C.white, bold: true, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "本质：把数据放在 SRAM（20 cycles）上算，而不是在 HBM（600 cycles）上反复读写。", options: { fontSize: 13, color: C.accent, bold: true, breakLine: true } },
    { text: "FlashAttention 没有改变数学公式，它改变的是数据的物理位置。", options: { fontSize: 12, color: C.muted } },
  ], {
    x: 0.6, y: 4.4, w: 8.8, h: 0.9,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.45,
  });

  addPageNum(s, 9);
}

// ============================================================
// SLIDE 10 — FlashAttention Solution
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "核心思路：既然搬数据是瓶颈，那就少搬", "FlashAttention — 不是数学创新，是硬件调度创新");

  // Before (Standard)
  addCard(s, 0.4, 1.5, 4.5, 3.1);
  addTopAccent(s, 0.4, 1.5, 4.5, C.red);
  s.addText("✕  标准 Attention", {
    x: 0.6, y: 1.6, w: 4.1, h: 0.35,
    fontSize: 16, fontFace: FONT, color: C.red, bold: true, margin: 0,
  });

  const beforeSteps = [
    "① 算出完整 N×N 注意力矩阵 (~128MB, N=4096)",
    "② 整个存在 HBM 里",
    "③ 读出来做 Softmax → 写回 HBM",
    "④ 读出来乘 V → 写回 HBM",
    "⑤ 每一步都是 HBM 级读写 (600 cycles/次)",
    "",
    "HBM 读写量：O(N²) — 毁灭性的",
  ];
  s.addText(beforeSteps.map((line, i) => ({
    text: line,
    options: {
      fontSize: 10,
      color: line.startsWith("HBM") ? C.red : C.muted,
      bold: line.startsWith("HBM"),
      breakLine: i < beforeSteps.length - 1,
    },
  })), {
    x: 0.6, y: 2.05, w: 4.1, h: 2.4,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.35,
  });

  // After (FlashAttention)
  addCard(s, 5.1, 1.5, 4.5, 3.1);
  addTopAccent(s, 5.1, 1.5, 4.5, C.accent);
  s.addText("✓  FlashAttention", {
    x: 5.3, y: 1.6, w: 4.1, h: 0.35,
    fontSize: 16, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  const afterSteps = [
    "① 切成小块，每次只搬一小块到 SRAM",
    "② 在 SRAM 里完成 Q×Kᵀ → Softmax → ×V",
    "③ 算完直接扔，不写回 HBM",
    "④ 只把很小的最终结果写回 HBM",
    "⑤ 大部分操作在 SRAM (20 cycles) 完成",
    "",
    "HBM 读写量：O(N) — 线性增长",
  ];
  s.addText(afterSteps.map((line, i) => ({
    text: line,
    options: {
      fontSize: 10,
      color: line.startsWith("HBM") ? C.accent : C.muted,
      bold: line.startsWith("HBM"),
      breakLine: i < afterSteps.length - 1,
    },
  })), {
    x: 5.3, y: 2.05, w: 4.1, h: 2.4,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.35,
  });

  // Bottom key insight
  addCard(s, 0.4, 4.85, 9.2, 0.55);
  s.addText([
    { text: "💡 ", options: { fontSize: 14 } },
    { text: "计算量一点没少，SRAM 也没变快。快在哪里？", options: { fontSize: 13, color: C.white, bold: true } },
    { text: " 快在少跑了仓库。", options: { fontSize: 14, color: C.accent, bold: true } },
    { text: " 这就是 IO-Aware 算法的含义。", options: { fontSize: 13, color: C.muted } },
  ], {
    x: 0.6, y: 4.88, w: 8.8, h: 0.5,
    fontFace: FONT, valign: "middle", margin: 0,
  });

  addPageNum(s, 10);
}

// ============================================================
// SLIDE 11 — block_size Calculation
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "block_size 怎么定的？—— SRAM 容量的物理约束", "为什么 block_size 不能设成 256？算一笔 SRAM 预算账");

  const blocks = [
    {
      size: "block_size = 64",
      items: "Q[64,64]=8KB + K[64,64]=8KB\n+ V[64,64]=8KB + S[64,64]=8KB\n+ 状态 ~0.5KB",
      total: "≈ 33 KB",
      limit: "< 128 KB ✓",
      status: "安全",
      color: C.accent,
      statusColor: C.accent,
    },
    {
      size: "block_size = 128",
      items: "Q[128,64]=16KB + K[128,64]=16KB\n+ V[128,64]=16KB + S[128,128]=32KB\n+ 状态 ~1KB",
      total: "≈ 81 KB",
      limit: "> 128 KB (4060) ✗\n< 164 KB (A100) ✓",
      status: "紧/溢出",
      color: C.orange,
      statusColor: C.orange,
    },
    {
      size: "block_size = 256",
      items: "Q[256,64]=32KB + K[256,64]=32KB\n+ V[256,64]=32KB + S[256,256]=128KB\n+ 状态 ~2KB",
      total: "≈ 226 KB",
      limit: "> 任何 GPU ✗",
      status: "溢出！",
      color: C.red,
      statusColor: C.red,
    },
  ];

  blocks.forEach((b, i) => {
    const x = 0.4 + i * 3.15;
    addCard(s, x, 1.5, 2.95, 3.0);
    addTopAccent(s, x, 1.5, 2.95, b.color);

    s.addText(b.size, {
      x: x + 0.15, y: 1.6, w: 2.65, h: 0.35,
      fontSize: 15, fontFace: FONT, color: b.color, bold: true, margin: 0,
    });
    s.addText(b.items, {
      x: x + 0.15, y: 2.05, w: 2.65, h: 1.1,
      fontSize: 9.5, fontFace: "Consolas", color: C.muted, margin: 0, lineSpacingMultiple: 1.45,
    });
    s.addText(b.total, {
      x: x + 0.15, y: 3.2, w: 2.65, h: 0.35,
      fontSize: 17, fontFace: FONT, color: C.white, bold: true, margin: 0,
    });
    s.addText(b.limit, {
      x: x + 0.15, y: 3.55, w: 2.65, h: 0.5,
      fontSize: 10, fontFace: FONT, color: b.statusColor, margin: 0, lineSpacingMultiple: 1.3,
    });

    // Status badge
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.15, y: 4.15, w: 1.2, h: 0.28,
      fill: { color: b.color, transparency: 80 },
      rectRadius: 0.14,
    });
    s.addText(b.status, {
      x: x + 0.15, y: 4.15, w: 1.2, h: 0.28,
      fontSize: 10, fontFace: FONT, color: b.color, align: "center", valign: "middle", margin: 0,
    });
  });

  // Bottom insight
  addCard(s, 0.4, 4.75, 9.2, 0.65);
  s.addText([
    { text: "结论：", options: { fontSize: 13, color: C.white, bold: true } },
    { text: "block_size 被 SRAM 的物理容量卡死。这就是为什么 H100 的 SRAM 从 192KB → 256KB 后，FlashAttention 天然更快——分块更大，搬运次数更少。", options: { fontSize: 11, color: C.muted } },
  ], {
    x: 0.6, y: 4.78, w: 8.8, h: 0.6,
    fontFace: FONT, valign: "middle", margin: 0, lineSpacingMultiple: 1.3,
  });

  addPageNum(s, 11);
}

// ============================================================
// SLIDE 12 — Bottleneck 1: SRAM
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "当前瓶颈一：SRAM 太小，卡住分块算法上限", "一个 SRAM bit = 6T，面积是 DRAM 的 5-7 倍，物理上限无法突破");

  // Problem
  addCard(s, 0.4, 1.5, 4.5, 2.0);
  addTopAccent(s, 0.4, 1.5, 4.5, C.red);
  s.addText("为什么是问题", {
    x: 0.6, y: 1.6, w: 4.1, h: 0.3,
    fontSize: 14, fontFace: FONT, color: C.red, bold: true, margin: 0,
  });
  s.addText([
    { text: "• block_size 能设多大，完全取决于 SRAM 有多大", options: { fontSize: 11, color: C.muted, breakLine: true } },
    { text: "• SRAM 每代只涨 25-33%（192KB → 256KB/SM）", options: { fontSize: 11, color: C.muted, breakLine: true } },
    { text: "• 根本原因：6T SRAM cell 占面积是 DRAM 的 5-7×", options: { fontSize: 11, color: C.muted, breakLine: true } },
    { text: "• 芯片总面积是硬约束，只要还用硅就不可能有质的突破", options: { fontSize: 11, color: C.muted } },
  ], {
    x: 0.6, y: 2.0, w: 4.1, h: 1.3,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.5,
  });

  // Solutions
  addCard(s, 5.1, 1.5, 4.5, 2.0);
  addTopAccent(s, 5.1, 1.5, 4.5, C.accent);
  s.addText("当前优化方向", {
    x: 5.3, y: 1.6, w: 4.1, h: 0.3,
    fontSize: 14, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });

  // TMA
  s.addText("硬件侧 — TMA", {
    x: 5.3, y: 2.0, w: 4.1, h: 0.25,
    fontSize: 12, fontFace: FONT, color: C.blue, bold: true, margin: 0,
  });
  s.addText("H100 新增的 Tensor Memory Accelerator，硬件自动在后台搬运 HBM→SRAM 数据。线程腾出手专心计算。FlashAttention-3 基于 TMA 进一步加速。", {
    x: 5.3, y: 2.25, w: 4.1, h: 0.55,
    fontSize: 10, fontFace: FONT, color: C.muted, margin: 0, lineSpacingMultiple: 1.4,
  });

  // Autotuning
  s.addText("软件侧 — 编译器自动分块", {
    x: 5.3, y: 2.85, w: 4.1, h: 0.25,
    fontSize: 12, fontFace: FONT, color: C.blue, bold: true, margin: 0,
  });
  s.addText("Triton 等编译器自动搜索最优 block_size。程序员只写计算逻辑，编译器为不同 GPU 自动适配 SRAM 分配方案。", {
    x: 5.3, y: 3.1, w: 4.1, h: 0.4,
    fontSize: 10, fontFace: FONT, color: C.muted, margin: 0, lineSpacingMultiple: 1.4,
  });

  // Transition
  addCard(s, 0.4, 3.75, 9.2, 1.65);
  addTopAccent(s, 0.4, 3.75, 9.2, C.purple);
  s.addText("瓶颈二预览：HBM 带宽不够，Memory-bound 不会自动消失", {
    x: 0.6, y: 3.85, w: 8.8, h: 0.35,
    fontSize: 15, fontFace: FONT, color: C.white, bold: true, margin: 0,
  });
  s.addText([
    { text: "B200 的 HBM3e 带宽 8 TB/s，但算力 2250 TFLOPS → 要求每 byte 做 281 次运算才喂饱 GPU。", options: { fontSize: 11, color: C.muted, breakLine: true } },
    { text: "Decode 阶段每 byte 只做 1-2 次运算。", options: { fontSize: 11, color: C.orange, bold: true, breakLine: true } },
    { text: "HBM 带宽从 A100 2TB/s → B200 8TB/s，涨了 4 倍。但算力涨了 7 倍。", options: { fontSize: 11, color: C.red, bold: true, breakLine: true } },
    { text: "矛盾不是缩小了，是更大了。", options: { fontSize: 12, color: C.red, bold: true } },
  ], {
    x: 0.6, y: 4.25, w: 8.8, h: 1.0,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.4,
  });

  addPageNum(s, 12);
}

// ============================================================
// SLIDE 13 — Bottleneck 2: HBM Bandwidth Solutions
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addSlideTitle(s, "当前瓶颈二：HBM 带宽优化 — 三条路线并行", "硬件微调比例 + 架构重构 + 软件压缩精度");

  // Route 1: HBM4
  addCard(s, 0.4, 1.5, 2.9, 2.8);
  addTopAccent(s, 0.4, 1.5, 2.9, C.blue);
  s.addText("路线一", {
    x: 0.6, y: 1.6, w: 2.5, h: 0.25,
    fontSize: 10, fontFace: FONT, color: C.muted, margin: 0,
  });
  s.addText("HBM4", {
    x: 0.6, y: 1.85, w: 2.5, h: 0.45,
    fontSize: 26, fontFace: FONT, color: C.blue, bold: true, margin: 0,
  });
  s.addText("硬件侧微调比例", {
    x: 0.6, y: 2.3, w: 2.5, h: 0.25,
    fontSize: 10, fontFace: FONT, color: C.muted, margin: 0,
  });

  const hbm4Specs = [
    "位宽: 1024→2048-bit",
    "单 Stack 带宽: 6-8+ TB/s",
    "堆叠 16 层 DRAM",
    "预计 2026-2027",
    "",
    "挑战: B200 已 1000W",
    "功耗和散热是巨大挑战",
  ];
  s.addText(hbm4Specs.map((line, i) => ({
    text: line,
    options: { fontSize: 10, color: C.muted, breakLine: i < hbm4Specs.length - 1 },
  })), {
    x: 0.6, y: 2.6, w: 2.5, h: 1.5,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.3,
  });

  // Route 2: PIM
  addCard(s, 3.55, 1.5, 2.9, 2.8);
  addTopAccent(s, 3.55, 1.5, 2.9, C.purple);
  s.addText("路线二", {
    x: 3.75, y: 1.6, w: 2.5, h: 0.25,
    fontSize: 10, fontFace: FONT, color: C.muted, margin: 0,
  });
  s.addText("近存计算", {
    x: 3.75, y: 1.85, w: 2.5, h: 0.45,
    fontSize: 22, fontFace: FONT, color: C.purple, bold: true, margin: 0,
  });
  s.addText("PIM — 架构重构", {
    x: 3.75, y: 2.3, w: 2.5, h: 0.25,
    fontSize: 10, fontFace: FONT, color: C.muted, margin: 0,
  });

  const pimSpecs = [
    "在 HBM 堆栈内部加入",
    "简单处理单元",
    "计算直接在内存侧完成",
    "从根本上减少搬数据需求",
    "",
    "三星、SK 海力士在推",
    "成熟度不高",
    "编程模型不完善",
  ];
  s.addText(pimSpecs.map((line, i) => ({
    text: line,
    options: { fontSize: 10, color: C.muted, breakLine: i < pimSpecs.length - 1 },
  })), {
    x: 3.75, y: 2.6, w: 2.5, h: 1.5,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.3,
  });

  // Route 3: Low Precision
  addCard(s, 6.7, 1.5, 2.9, 2.8);
  addTopAccent(s, 6.7, 1.5, 2.9, C.accent);
  s.addText("路线三", {
    x: 6.9, y: 1.6, w: 2.5, h: 0.25,
    fontSize: 10, fontFace: FONT, color: C.muted, margin: 0,
  });
  s.addText("低精度计算", {
    x: 6.9, y: 1.85, w: 2.5, h: 0.45,
    fontSize: 20, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });
  s.addText("软件侧压缩精度", {
    x: 6.9, y: 2.3, w: 2.5, h: 0.25,
    fontSize: 10, fontFace: FONT, color: C.muted, margin: 0,
  });

  const lpSpecs = [
    "FP8 → FP4 → INT4",
    "数据精度砍半",
    "等效带宽翻倍",
    "",
    "B200 已支持 FP4",
    "短期见效最快",
    "成本最低的方向",
    "",
    "缺点: 精度损失",
    "不是所有任务都能用",
  ];
  s.addText(lpSpecs.map((line, i) => ({
    text: line,
    options: { fontSize: 10, color: C.muted, breakLine: i < lpSpecs.length - 1 },
  })), {
    x: 6.9, y: 2.6, w: 2.5, h: 1.5,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.3,
  });

  // Bottom: Two paths summary
  addCard(s, 0.4, 4.6, 9.2, 0.8);
  addTopAccent(s, 0.4, 4.6, 9.2, C.orange);
  s.addText([
    { text: "两条路并行：", options: { fontSize: 13, color: C.white, bold: true } },
    { text: "硬件侧微调比例", options: { fontSize: 13, color: C.blue, bold: true } },
    { text: "（HBM4、近存计算）→ 每代 +30-50%　|　", options: { fontSize: 12, color: C.muted } },
    { text: "软件侧重写 IO 模式", options: { fontSize: 13, color: C.accent, bold: true } },
    { text: "（FlashAttention、低精度）→ 一次 +10-100×", options: { fontSize: 12, color: C.muted } },
  ], {
    x: 0.6, y: 4.63, w: 8.8, h: 0.75,
    fontFace: FONT, valign: "middle", margin: 0,
  });

  addPageNum(s, 13);
}

// ============================================================
// SLIDE 14 — Summary & Personal Thoughts
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Decorative glows
  s.addShape(pres.shapes.OVAL, {
    x: 7, y: -1.5, w: 5, h: 5,
    fill: { color: C.accent, transparency: 95 },
  });
  s.addShape(pres.shapes.OVAL, {
    x: -2, y: 3.5, w: 4, h: 4,
    fill: { color: C.blue, transparency: 95 },
  });

  // Section title
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.4, w: 0.06, h: 0.4,
    fill: { color: C.accent },
  });
  s.addText("总结与思考", {
    x: 0.75, y: 0.35, w: 4, h: 0.5,
    fontSize: 28, fontFace: FONT, color: C.white, bold: true, margin: 0,
  });

  // Four key takeaways in 2x2 grid
  const takeaways = [
    { title: "一个核心矛盾", desc: "算力增长远超带宽增长。从 A100 到 B200，FLOPS 涨 7×，带宽涨 4×。差距不会消失——物理定律决定的。", color: C.red },
    { title: "一个不变框架", desc: "四层金字塔（寄存器→SRAM→L2→HBM）是 GPU 存储的不变结构。每代只在比例上微调，形状不会变。", color: C.blue },
    { title: "一个核心技能", desc: "用手动管理的 SRAM（__shared__）替代自动管理的 L2/HBM。FlashAttention = 把 Attention 中间计算从 HBM 搬进 SRAM。", color: C.accent },
    { title: "一个未来趋势", desc: "硬件微调比例（+30%），软件重写 IO（+10-100×）。AI Infra 的价值：不是等硬件变快，是让软件跑得更聪明。", color: C.purple },
  ];

  takeaways.forEach((t, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.4 + col * 4.7;
    const y = 1.15 + row * 1.55;
    const w = 4.5;
    const h = 1.4;

    addCard(s, x, y, w, h);
    addTopAccent(s, x, y, w, t.color);
    s.addText(t.title, {
      x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.3,
      fontSize: 13, fontFace: FONT, color: t.color, bold: true, margin: 0,
    });
    s.addText(t.desc, {
      x: x + 0.2, y: y + 0.5, w: w - 0.4, h: 0.8,
      fontSize: 10, fontFace: FONT, color: C.muted, margin: 0, lineSpacingMultiple: 1.5,
    });
  });

  // Quote at bottom
  addCard(s, 0.4, 4.35, 9.2, 1.05);
  addTopAccent(s, 0.4, 4.35, 9.2, C.accent);
  s.addText([
    { text: "来课题组之前，我以为 AI 慢是因为 GPU 算得不够快。学完这两周我意识到：", options: { fontSize: 12, color: C.muted, breakLine: true } },
    { text: "瓶颈不在算力，在搬运。", options: { fontSize: 15, color: C.accent, bold: true, breakLine: true } },
    { text: "AI Infra 的魅力：不是在硬件和算法之间做选择，而是理解两者的约束后，找到让它们更好配合的方式。", options: { fontSize: 12, color: C.white } },
  ], {
    x: 0.6, y: 4.4, w: 8.8, h: 0.95,
    fontFace: FONT, valign: "middle", margin: 0, lineSpacingMultiple: 1.5,
  });

  addPageNum(s, 14);
}

// ============================================================
// Output
// ============================================================
pres.writeFile({ fileName: "C:/Users/19374/Desktop/AI Infra/GPU存储层次_Week2_汇报.pptx" })
  .then(() => console.log("PPT saved successfully!"))
  .catch(err => console.error("Error:", err));
