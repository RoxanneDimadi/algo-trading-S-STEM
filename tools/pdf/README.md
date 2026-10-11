# Building `docs/STUDY_GUIDE.pdf`

`docs/STUDY_GUIDE.md` is the source of truth. This directory regenerates the
PDF from it; do not hand-edit the PDF.

## What the pipeline does

1. **Preprocess** — opens the Appendix E answer boxes (`<details>` →
   `<details open>`) so they print, and swaps the in-document table of
   contents for a page-numbered one.
2. **pandoc** (`gfm+tex_math_dollars`, `--katex`) → HTML, with every `$…$` and
   `$$…$$` passed through verbatim rather than pandoc's own TeX converter.
3. **KaTeX, server-side** — each math span is rendered to HTML at build time,
   so nothing depends on JavaScript or a CDN at print time. The source escapes
   the superscript star as `\*` for GitHub's markdown pass; the builder strips
   that escape before handing the TeX to KaTeX.
4. **Chromium via Playwright** → A4 PDF with `study_guide_print.css`, printed
   **twice**: the first pass reveals which page each part starts on, the second
   bakes those numbers into the contents page.

## Requirements

- `pandoc`
- `node` with `playwright` (local or global) and a Chromium build
- `npm install` in this directory (pulls KaTeX, which ships its own fonts)

No LaTeX distribution is needed.

## Run

```bash
cd tools/pdf
npm install
node build_study_guide_pdf.mjs ../../docs/STUDY_GUIDE.md ../../docs/STUDY_GUIDE.pdf
```

It prints the math-span count and the number of rendering failures. **A clean
build reports `failures: 0`** — anything else means a formula silently fell
back to monospace source and needs looking at.
