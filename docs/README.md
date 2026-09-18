# Documentation

Everything written *about* this project, collected in one place. Code and
configuration live in `multisignal-alpha/` (research agent) and `real-data/`
(ingest).

| File | What it is |
|---|---|
| `USER_GUIDE.md` | The complete use guide: setup, both sub-projects, the test suite, the synthetic demo, getting real data without WRDS, running on real data, how to read every output table and figure, the four models, the trading agent, the config reference, what the results do and don't claim, troubleshooting. **Start here.** |
| `math/` | Proof-driven lesson plan, chapter per file — every formula in the repository derived from first principles, assuming no finance background. Start at `math/00_index.md`. |
| `mathematical_foundations_lesson_plan.md` | The same lesson plan as one continuous document, for reading or printing end-to-end. |

Design and research notes stay next to the code they describe, in
`multisignal-alpha/multisignal-alpha/docs/` (research findings, roadmap,
improvement backlog, model proposals). `research/documents/` used to hold a
second, byte-identical copy of those; it now holds only the original project
archive and a pointer.
