// Build the research report .docx from the JSON payload produced by
// scripts/12_build_report.py. docx-js is the reliable way to create Word files.
//
// Usage:  node build_report.js report/_report_data.json

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  ImageRun, PageBreak, PageNumber, Header, Footer,
} = require("docx");

const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

const INK = "1a1a2e";
const ACCENT = "c1440e";
const GREY = "6b7280";
const LIGHT = "eef1f5";

// ---- helpers ----------------------------------------------------------------
const fmt = (x, d = 2) =>
  x == null || isNaN(x) ? "—" : (x >= 0 ? "+" : "") + Number(x).toFixed(d);

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 120, line: 276, ...(opts.spacing || {}) },
    alignment: opts.align,
    children: Array.isArray(text)
      ? text
      : [new TextRun({ text, size: opts.size ?? 21, color: opts.color ?? INK,
                       bold: opts.bold, italics: opts.italics, font: "Calibri" })],
  });
}

function h(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: 260, after: 130 },
    children: [new TextRun({ text, font: "Calibri Light", color: INK })],
  });
}

function img(path, w, hh) {
  if (!path || !fs.existsSync(path)) return p("[figure unavailable]", { italics: true, color: GREY });
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(path),
      transformation: { width: w, height: hh },
    })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text, italics: true, size: 17, color: GREY, font: "Calibri" })],
  });
}

function cell(text, { bold, align, shade, width, color } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align || AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), bold, size: 18,
                               color: color || INK, font: "Calibri" })],
    })],
  });
}

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((hd, i) =>
      cell(hd, { bold: true, shade: INK, color: "ffffff",
                 align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
                 width: widths[i] })),
  });
  const bodyRows = rows.map((r, ri) =>
    new TableRow({
      children: r.map((c, i) =>
        cell(c.text ?? c, {
          bold: c.bold, color: c.color,
          shade: c.shade || (ri % 2 ? LIGHT : undefined),
          align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
          width: widths[i],
        })),
    }));
  return new Table({
    columnWidths: widths,
    width: { size: total, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "cccccc" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "cccccc" },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "e5e5e5" },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [headerRow, ...bodyRows],
  });
}

function bullet(text) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 80, line: 276 },
    children: [new TextRun({ text, size: 21, color: INK, font: "Calibri" })],
  });
}

// ---- content ----------------------------------------------------------------
const live = payload.live || {};
const board = payload.leaderboard || [];
const best = board[0] || {};
const benchmark = board.find((b) => b.model === "rolling_mean") || {};
const bridge = board.find((b) => b.model === "bridge_average") || {};
const dfm = board.find((b) => b.model && b.model.startsWith("dfm")) || {};

const kids = [];

// Title block
kids.push(new Paragraph({
  spacing: { after: 60 },
  children: [new TextRun({ text: "Nowcasting Australian GDP Growth",
    bold: true, size: 40, color: INK, font: "Calibri Light" })],
}));
kids.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "A point-in-time evaluation of bridge equations and a dynamic factor model",
    size: 24, color: ACCENT, font: "Calibri Light" })],
}));
kids.push(new Paragraph({
  spacing: { after: 260 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "cccccc", space: 8 } },
  children: [new TextRun({ text: "Independent research project · generated " + (live.generated || ""),
    size: 18, color: GREY, italics: true, font: "Calibri" })],
}));

// Abstract
kids.push(h("Abstract", HeadingLevel.HEADING_2));
const improveTxt = bridge.improvement != null
  ? `${bridge.improvement}%` : "a small margin";
kids.push(p(
  `This project builds a point-in-time nowcasting system for Australian quarterly ` +
  `real GDP growth using monthly indicators published by the ABS and RBA. Because ` +
  `official GDP is released roughly nine weeks after a quarter ends, the current ` +
  `quarter is systematically unmeasured; the system estimates it from labour-market, ` +
  `trade, construction and financial data that arrive earlier. The central ` +
  `methodological commitment is a strictly point-in-time backtest: every observation ` +
  `carries the date it was actually published, so no forecast can use information ` +
  `unavailable at the time. Across a 1993–2019 out-of-sample evaluation, a ` +
  `rolling-window average of bridge equations was the most accurate model, improving ` +
  `on a rolling-mean benchmark by ${improveTxt}, with accuracy rising as monthly data ` +
  `accrued through the quarter. The improvement was not statistically significant ` +
  `(Diebold–Mariano test), and a mixed-frequency dynamic factor model did not ` +
  `outperform the simpler combination. The result is a credible negative finding: ` +
  `standard indicators provide limited, non-significant uplift for Australian GDP at ` +
  `a one-quarter horizon, and model complexity is not rewarded on a short sample.`,
  { after: 200 }));

// 1. Introduction
kids.push(h("1. Introduction and motivation", HeadingLevel.HEADING_2));
kids.push(p(
  `Gross domestic product is the headline measure of economic activity, but it is ` +
  `published with a long delay and at quarterly frequency only. In Australia the ABS ` +
  `releases the national accounts approximately 65 days after the reference quarter ` +
  `closes, so at any given moment the most recent completed quarter has no official ` +
  `GDP figure. Policymakers, markets and forecasters therefore rely on "nowcasting": ` +
  `inferring current activity from higher-frequency indicators that are published ` +
  `sooner — employment, hours worked, retail and household spending, building ` +
  `approvals, trade flows and financial variables.`));
kids.push(p(
  `This project implements and rigorously evaluates such a system for Australia. The ` +
  `emphasis is not on a single model but on an honest evaluation framework: the ` +
  `central question is whether monthly indicators genuinely improve on naive ` +
  `benchmarks once look-ahead bias is eliminated. The distinction matters because ` +
  `nowcasting results are unusually easy to inflate; a one-line timing error that ` +
  `lets a model see data from the future can produce large but entirely spurious ` +
  `accuracy gains.`));

// 2. Data
kids.push(h("2. Data", HeadingLevel.HEADING_2));
kids.push(p(
  `The target is quarter-on-quarter growth in real GDP (chain volume measures, ` +
  `seasonally adjusted; ABS 5206.0). Predictors are drawn from monthly ABS and RBA ` +
  `series. All data is publicly available and retrieved programmatically via the ` +
  `readabs package. Table 1 lists the series, their native frequency, the ` +
  `stationarity transform applied, and, critically, the publication lag used to ` +
  `determine when each observation became known.`));

const seriesRows = (payload.series || []).map((s) => [
  s.name.replace(/_/g, " "),
  s.series_id, s.collection, s.freq, String(s.lag_days), s.transform,
]);
kids.push(table(
  ["Series", "ABS ID", "Cat.", "Freq", "Lag (d)", "Transform"],
  seriesRows,
  [1700, 1300, 1000, 700, 900, 1800]));
kids.push(caption("Table 1. Series registry. Publication lags are conservative upper bounds verified against the ABS release calendar."));

kids.push(p(
  `Publication lags were verified against the ABS release calendar and set to ` +
  `conservative upper bounds: where a release date varied month to month, the ` +
  `longest recent delay plus a small buffer was used. This biases the system against ` +
  `itself, it occasionally discards data it could legitimately have used, so ` +
  `measured accuracy is a lower bound on what a real-time forecaster could achieve. ` +
  `The balanced-panel sample begins ${payload.balanced_start}, constrained by the ` +
  `shortest long-history series; the Monthly Household Spending Indicator ` +
  `(from 2012) is used only by the factor model, which handles a ragged left edge.`,
  { after: 200 }));

// 3. Method
kids.push(h("3. Methodology", HeadingLevel.HEADING_2));
kids.push(h("3.1 Point-in-time data and the ragged edge", HeadingLevel.HEADING_3));
kids.push(p(
  `Every observation is stored in a long panel stamped with an availability date ` +
  `equal to the end of its reference period plus its publication lag. A "snapshot" ` +
  `as of any date is formed by keeping only rows whose availability date precedes it. ` +
  `Because series publish at different speeds, the most recent months form a ` +
  `staircase pattern, the ragged edge, in which faster indicators (labour force) ` +
  `extend further than slower ones (trade, construction). This construction makes ` +
  `look-ahead bias structurally impossible rather than a matter of discipline, and ` +
  `is enforced by an assertion in the backtest and by unit tests that deliberately ` +
  `inject leakage and confirm it is caught.`));

kids.push(h("3.2 Stationarity", HeadingLevel.HEADING_3));
kids.push(p(
  `Level series were transformed to stationary form — percentage changes for ` +
  `volume/value series, first differences for rates. Augmented Dickey–Fuller tests ` +
  `confirm stationarity of the resulting regressors (Table 2).`));

const adfRows = (payload.adf || []).map((r) => [
  r.series.replace(/_/g, " "),
  r.adf_stat == null ? "—" : r.adf_stat.toFixed(2),
  r.p_value == null ? "—" : r.p_value.toFixed(4),
  { text: r.stationary ? "yes" : "no", color: r.stationary ? "2f7d4f" : ACCENT, bold: true },
]);
kids.push(table(
  ["Series", "ADF stat", "p-value", "Stationary (5%)"],
  adfRows, [2400, 1400, 1400, 1600]));
kids.push(caption("Table 2. Augmented Dickey–Fuller tests. The null of a unit root is rejected for every regressor."));

kids.push(h("3.3 Bridge equations", HeadingLevel.HEADING_3));
kids.push(p(
  `A bridge equation regresses quarterly GDP growth on the quarterly reading of a ` +
  `monthly indicator, using whatever months of the target quarter have been ` +
  `published. Two construction choices matter. First, monthly indicators enter as ` +
  `levels and the regressor is the change in the quarterly average level — the same ` +
  `quantity GDP is measured on — rather than an average of monthly growth rates; a ` +
  `controlled test showed this recovers true quarterly growth with roughly 60% lower ` +
  `error when only one month is available. Second, estimation uses a 40-quarter ` +
  `rolling window, because Australian trend growth has declined over the sample and ` +
  `an expanding window anchors the intercept on a stale, higher average — the same ` +
  `mechanism that makes a rolling mean outperform an expanding mean here.`));

kids.push(h("3.4 Dynamic factor model", HeadingLevel.HEADING_3));
kids.push(p(
  `The bridge equations read each indicator in isolation. A mixed-frequency dynamic ` +
  `factor model instead treats all indicators as noisy signals of a common latent ` +
  `factor: the business cycle, estimated by the Kalman filter and EM algorithm ` +
  `(statsmodels DynamicFactorMQ). It ingests monthly and quarterly series at their ` +
  `native frequencies, handles the ragged edge natively, and can incorporate the ` +
  `short-history spending series. This is the architecture underlying the New York ` +
  `Fed Staff Nowcast.`));

kids.push(h("3.5 Evaluation", HeadingLevel.HEADING_3));
kids.push(p(
  `Models are evaluated by expanding-window pseudo-out-of-sample forecasting: one ` +
  `nowcast per GDP release, the model refit from scratch on exactly the data visible ` +
  `at that vintage. To trace how accuracy improves within a quarter, the backtest is ` +
  `run at 0, 30, 60 and 80 days after each release. Accuracy is summarised by RMSE ` +
  `and MAE, and differences are tested with the Diebold–Mariano test rather than ` +
  `read off raw RMSE, since with roughly one hundred forecasts small gaps are not ` +
  `statistically meaningful.`, { after: 200 }));

// 4. Results
kids.push(new Paragraph({ children: [new PageBreak()] }));
kids.push(h("4. Results", HeadingLevel.HEADING_2));

kids.push(h("4.1 Model comparison", HeadingLevel.HEADING_3));
kids.push(p(
  `Table 3 ranks all models at the richest information set (80 days after the prior ` +
  `release), on the 1993–2019 sample that excludes both the volatile 1970s–80s and ` +
  `the COVID shock. ${best.model ? best.model.replace(/_/g, " ") : "The leading model"} ` +
  `achieved the lowest RMSE.`));

const lbRows = board.slice(0, 12).map((r, i) => [
  { text: r.model.replace(/_/g, " "), bold: i === 0, color: i === 0 ? ACCENT : INK },
  r.rmse.toFixed(4), r.mae.toFixed(4), fmt(r.bias, 3),
  r.improvement == null ? "—"
    : r.improvement > 0 ? `${r.improvement}% better`
    : r.improvement < 0 ? `${Math.abs(r.improvement)}% worse` : "baseline",
]);
kids.push(table(
  ["Model", "RMSE", "MAE", "Bias", "vs benchmark"],
  lbRows, [2200, 1300, 1300, 1200, 1700]));
kids.push(caption("Table 3. Out-of-sample accuracy, 1993–2019, 80 days after the prior GDP release. RMSE and MAE in percentage points of quarterly growth."));

kids.push(p([
  new TextRun({ text: "The ranking carries the project's main message. ", size: 21, font: "Calibri", color: INK }),
  new TextRun({ text: "The simple forecast combination leads; the dynamic factor model does not improve on it; and the flexible ridge regression is among the worst.",
    size: 21, font: "Calibri", color: INK, bold: true }),
  new TextRun({ text: " This is the expected outcome on a short macroeconomic sample, where richly-parameterised models pay an estimation-variance cost that their flexibility does not recoup.",
    size: 21, font: "Calibri", color: INK }),
], { after: 160 }));

kids.push(h("4.2 Accuracy improves as data arrives", HeadingLevel.HEADING_3));
kids.push(p(
  `Figure 1 plots RMSE against how many days into the forecasting cycle the nowcast ` +
  `is formed. The headline model's error falls as more monthly data accrues, the ` +
  `signature of a nowcast that is extracting genuine signal, not merely fitting the ` +
  `unconditional mean. Models that ignore the monthly indicators appear as flat lines.`,
  { after: 160 }));
kids.push(img(payload.figures?.vintage_curve, 520, 304));
kids.push(caption("Figure 1. Nowcast RMSE by days after the prior GDP release (1993–2019). The bridge combination improves monotonically as data arrives."));

kids.push(h("4.3 Nowcast versus outcome", HeadingLevel.HEADING_3));
kids.push(p(
  `Figure 2 overlays the headline nowcast on realised GDP growth across the full ` +
  `evaluation period. The model tracks the broad path of activity but, as the ` +
  `statistics indicate, adds only modest information beyond a slowly-moving mean, ` +
  `and no model anticipates the COVID collapse of 2020Q2, an approximately ten-sigma ` +
  `event that no indicator-based system could have foreseen.`));
kids.push(img(payload.figures?.actual_vs_pred, 520, 272));
kids.push(caption("Figure 2. Point-in-time nowcast versus actual quarterly GDP growth."));

if (payload.figures?.news) {
  kids.push(h("4.4 News decomposition", HeadingLevel.HEADING_3));
  kids.push(p(
    `A distinctive capability of the factor model is attributing a change in the ` +
    `nowcast to the specific data releases that caused it. As each indicator is ` +
    `published, its surprise relative to the model's expectation moves the nowcast by ` +
    `an amount equal to that surprise times the weight the Kalman filter assigns it. ` +
    `Figure 3 shows this decomposition for the current quarter. The model rendered ` +
    `as an explanation rather than a black box, which is precisely what a central-bank ` +
    `nowcasting desk publishes.`));
  kids.push(img(payload.figures.news, 520, 256));
  kids.push(caption("Figure 3. News decomposition: contribution of each indicator's latest surprise to the current-quarter nowcast."));
}

// 5. Live nowcast
kids.push(h("5. Current nowcast", HeadingLevel.HEADING_2));
if (live.headline_nowcast != null) {
  kids.push(p([
    new TextRun({ text: `As of ${live.as_of}, the system's nowcast for ${live.target_quarter} — `,
      size: 21, font: "Calibri", color: INK }),
    new TextRun({ text: `a quarter not yet published by the ABS is ${fmt(live.headline_nowcast)}% `,
      size: 21, font: "Calibri", color: INK, bold: true }),
    new TextRun({ text: `quarter-on-quarter, from the rolling-window bridge combination. The most recently ` +
      `published figure was ${fmt(live.latest_published_value)}% for ${live.latest_published_quarter}. ` +
      `Because the target quarter has no official outcome yet, this is a genuine forward ` +
      `statement whose accuracy will be revealed at the next ABS release; the test of any nowcasting system.`,
      size: 21, font: "Calibri", color: INK }),
  ], { after: 160 }));
}

// 6. Limitations
kids.push(h("6. Limitations", HeadingLevel.HEADING_2));
kids.push(bullet(
  "Final-vintage data. The system uses the latest revised figures rather than the " +
  "first prints a real-time forecaster would have seen. GDP revisions in Australia " +
  "can reach several tenths of a percentage point, so measured accuracy is optimistic " +
  "relative to a true real-time system. Incorporating genuine data vintages is the " +
  "single most valuable extension."));
kids.push(bullet(
  "Constant publication lags. Lags are held fixed across the sample, though ABS " +
  "releases were slower historically; the assumption is mildly favourable to the " +
  "model on older data."));
kids.push(bullet(
  "Short sample. With roughly 130 usable quarters, flexible models are structurally " +
  "disadvantaged, and the failure of the factor model to win should be read in that light."));
kids.push(bullet(
  "Narrow indicator set. The system omits business and consumer surveys, financial " +
  "conditions indices and commodity-price detail that richer nowcasting systems exploit; " +
  "these are plausible sources of the uplift the current indicators do not provide."));

// 7. Conclusion
kids.push(h("7. Conclusion", HeadingLevel.HEADING_2));
kids.push(p(
  `A rigorous point-in-time framework shows that standard monthly indicators offer ` +
  `only limited, statistically insignificant improvement over a rolling-mean benchmark ` +
  `for one-quarter-ahead Australian GDP growth, and that a mixed-frequency dynamic ` +
  `factor model does not beat a simple combination of bridge equations. Far from a ` +
  `disappointment, this is the credible outcome the framework was built to detect: ` +
  `the value of the project lies in the honesty of the evaluation, not in an inflated ` +
  `accuracy figure. The most promising directions is real-time data vintages and a ` +
  `broader indicator set including survey and financial data which follows directly from ` +
  `the limitations above.`, { after: 160 }));

kids.push(new Paragraph({
  spacing: { before: 200 },
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: "cccccc", space: 8 } },
  children: [new TextRun({
    text: "Code, tests and an interactive dashboard accompany this report. Data: ABS and RBA, " +
      "retrieved via the readabs package. This is a personal research project and not investment advice.",
    size: 16, italics: true, color: GREY, font: "Calibri" })],
}));

// ---- document ---------------------------------------------------------------
const doc = new Document({
  creator: "Independent research project",
  title: "Nowcasting Australian GDP Growth",
  styles: {
    paragraphStyles: [
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: INK, font: "Calibri Light" } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, color: "374151", font: "Calibri" } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES],
          size: 16, color: GREY, font: "Calibri" })],
      })] }),
    },
    children: kids,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(payload.out_path, buf);
  console.log("  wrote " + payload.out_path);
});
