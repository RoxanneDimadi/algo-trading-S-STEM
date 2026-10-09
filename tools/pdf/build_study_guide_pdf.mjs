import { readFileSync, writeFileSync, mkdirSync, cpSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

import { createRequire } from 'node:module';

// Playwright may be installed locally or globally; resolve either.
const require_ = createRequire(import.meta.url);
let pw;
for (const spec of ['playwright', 'playwright-core',
                    '/opt/node-tools/node_modules/playwright/index.js']) {
  try { pw = require_(spec); break; } catch { /* try the next one */ }
}
if (!pw) throw new Error('playwright not found (npm i playwright)');
const { chromium } = pw;

const katex = require_('katex');
const BUILD = path.dirname(new URL(import.meta.url).pathname);
const SRC = process.argv[2];
const OUT = process.argv[3];

// ---------------------------------------------------------------- 1. preprocess
let md = readFileSync(SRC, 'utf8');
// print the self-test answers rather than hiding them behind a disclosure widget
md = md.replace(/<details>/g, '<details open>');
// the in-document TOC is a web convenience; the PDF gets page-numbered nav instead
md = md.replace(/\n\*\*Contents\*\*\n\n- \[Part 0 —[\s\S]*?\[Appendix F — Reading list\]\(#appendix-f--reading-list\)\n/,
                '\n<!--TOC_PLACEHOLDER-->\n');
const stage = path.join(BUILD, 'stage.md');
writeFileSync(stage, md);

// ---------------------------------------------------------------- 2. pandoc
const body = execFileSync('pandoc', [
  '-f', 'gfm+tex_math_dollars+raw_html+attributes',
  '-t', 'html5', '--katex', '--wrap=preserve', stage,
], { encoding: 'utf8', maxBuffer: 1 << 28 });

// ---------------------------------------------------------------- 3. render math
const unesc = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>')
                    .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
                    .replace(/&amp;/g, '&');
let mathCount = 0, mathFail = 0;
const renderMath = (html) => html.replace(
  /<span class="math (inline|display)">([\s\S]*?)<\/span>/g,
  (_m, kind, tex) => {
    mathCount++;
    try {
      // the source escapes the superscript star as \* so GitHub's markdown
      // pass does not read it as emphasis; KaTeX wants a bare *
      const clean = unesc(tex).trim().replace(/\\\*/g, '*');
      return katex.renderToString(clean, {
        displayMode: kind === 'display', throwOnError: true,
        strict: false, output: 'html', trust: false,
      });
    } catch (e) {
      mathFail++;
      console.error(`MATH FAIL (${kind}): ${clean.slice(0, 140)}\n   -> ${e.message}`);
      return `<code class="mathfail">${tex}</code>`;
    }
  });
let rendered = renderMath(body);
console.error(`math spans: ${mathCount}, failures: ${mathFail}`);

// ---------------------------------------------------------------- 4. front matter + TOC
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
// drop the markdown H1 (it becomes the title page) and its intro paragraph lead
rendered = rendered.replace(/^<h1[^>]*>Multi-Signal Alpha — Complete Study Guide<\/h1>\s*/, '');

// collect the Part/Appendix headings for the contents page
const tocRows = [...rendered.matchAll(/<h1 id="([^"]+)">([\s\S]*?)<\/h1>/g)]
  .map(m => ({ id: m[1], text: m[2].replace(/<[^>]+>/g, '') }));
const tocHtml = `
<div class="contents">
  <h1 class="nobreak">Contents</h1>
  <ol class="toc">
    ${tocRows.map(r => `<li><a href="#${r.id}"><span class="t">${esc(r.text)}</span>`
      + `<span class="pg" data-head="${esc(r.text)}">PAGE_${r.id}</span></a></li>`).join('\n    ')}
  </ol>
  <p class="tocnote">Section cross-references in the text (written as §4.7, §8.3 and so on)
  point to the numbered sub-sections inside these parts. Every internal link in this PDF is
  clickable.</p>
</div>`;
rendered = rendered.replace('<!--TOC_PLACEHOLDER-->', tocHtml);

const today = new Date().toISOString().slice(0, 10);
const commit = execFileSync('git', ['-C', path.dirname(path.dirname(SRC)), 'rev-parse', '--short', 'HEAD'],
  { encoding: 'utf8' }).trim();
const words = readFileSync(SRC, 'utf8').split(/\s+/).filter(Boolean).length;

const titlepage = `
<section class="titlepage">
  <div class="eyebrow">Multi-Signal Alpha</div>
  <h1 class="doctitle">Complete Study Guide</h1>
  <div class="rule"></div>
  <p class="subtitle">Project overview, every formula and what it means,
  the four predictive models, the differentiable trading agent,
  the full results, the analysis and the conclusions.</p>
  <p class="meta">
    <strong>A cross-sectional equity return-prediction research platform</strong><br/>
    Synthetic planted-truth validation &nbsp;·&nbsp; 212-factor real-data run<br/><br/>
    Written in layers: Part 0 assumes no finance or statistics;<br/>
    Part 5 onward is theorems.<br/><br/>
    Generated ${today} &nbsp;·&nbsp; repository commit <code>${commit}</code><br/>
    ${words.toLocaleString('en-US')} words &nbsp;·&nbsp; 9 parts &nbsp;·&nbsp; 6 appendices
  </p>
</section>`;

// ---------------------------------------------------------------- 5. assemble
mkdirSync(path.join(BUILD, 'out'), { recursive: true });
cpSync(path.join(BUILD, 'node_modules/katex/dist/katex.min.css'), path.join(BUILD, 'out/katex.min.css'));
cpSync(path.join(BUILD, 'node_modules/katex/dist/fonts'), path.join(BUILD, 'out/fonts'), { recursive: true });
cpSync(path.join(BUILD, 'study_guide_print.css'), path.join(BUILD, 'out/style.css'));

const html = `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>Multi-Signal Alpha — Complete Study Guide</title>
<link rel="stylesheet" href="katex.min.css"/>
<link rel="stylesheet" href="style.css"/>
<style>
.contents{ page-break-before:always; }
.contents h1{ page-break-before:avoid; margin-bottom:5mm; }
ol.toc{ list-style:none; padding:0; margin:6mm 0 8mm;
  font-family:Inter,'Liberation Sans',sans-serif; font-size:10.5pt; }
ol.toc li{ padding:1.9mm 0; border-bottom:.5px dotted var(--rule); }
ol.toc a{ color:var(--ink); display:flex; justify-content:space-between; align-items:baseline; gap:4mm; }
ol.toc .pg{ font-variant-numeric:tabular-nums; color:var(--muted); flex:0 0 auto; }
.tocnote{ font-size:9.2pt; color:var(--muted); font-style:italic; }
code.mathfail{ background:#ffe9e9; color:#a00; }
</style>
</head><body>
${titlepage}
${rendered}
</body></html>`;
const htmlPath = path.join(BUILD, 'out/guide.html');

// ---------------------------------------------------------------- 6. print
const foot = `<div style="font-family:Inter,'Liberation Sans',sans-serif;font-size:7.4pt;
  color:#6b7280;width:100%;padding:0 16mm;display:flex;justify-content:space-between;">
  <span>Multi-Signal Alpha — Complete Study Guide</span>
  <span class="pageNumber"></span>
</div>`;
const PDF_OPTS = {
  format: 'A4', printBackground: true,
  margin: { top: '17mm', bottom: '18mm', left: '16mm', right: '16mm' },
  displayHeaderFooter: true, headerTemplate: '<div></div>', footerTemplate: foot,
  preferCSSPageSize: false,
};

const browser = await chromium.launch();
const render = async (htmlText, outPath) => {
  writeFileSync(htmlPath, htmlText);
  const page = await browser.newPage();
  await page.goto('file://' + htmlPath, { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(900);
  await page.pdf({ ...PDF_OPTS, path: outPath });
  await page.close();
};

// pass 1: placeholders in the contents, used only to learn where each part lands
const tmpPdf = path.join(BUILD, 'out/pass1.pdf');
await render(html, tmpPdf);

const pageText = JSON.parse(execFileSync('python3', ['-I', '-c', `
import json,sys
from pypdf import PdfReader
r = PdfReader(${JSON.stringify(tmpPdf)})
print(json.dumps([(p.extract_text() or '') for p in r.pages]))
`], { encoding: 'utf8', maxBuffer: 1 << 28 }));

const norm = t => t.replace(/[\s ]+/g, ' ').trim();
let html2 = html;
for (const r of tocRows) {
  const needle = norm(r.text);
  let found = 0;
  for (let i = 0; i < pageText.length; i++) {
    if (norm(pageText[i]).includes(needle)) { found = i + 1; break; }
  }
  html2 = html2.replace(`PAGE_${r.id}`, found ? String(found) : '');
}
// pass 2: the real document, with page numbers in the contents
await render(html2, OUT);
await browser.close();
console.error('wrote ' + OUT + ` (${pageText.length} pages)`);
