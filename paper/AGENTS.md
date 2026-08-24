# Paper-scoped editing instructions

These instructions apply recursively to every file under `paper/`, including
`submission/iclr2027/main.tex`, appendices, rebuttals, captions, tables,
result summaries, the dashboard, and review responses.

Before making any paper-facing edit, read `PAPER_WRITING_STYLE.md` completely
and follow it. Do not substitute a summary for the complete guide.

In particular:

- keep methods about design and results about findings;
- move useful material instead of deleting it;
- make only targeted edits unless the user requests a rewrite;
- state the exact comparator and scope for every performance claim;
- preserve negative and partial findings;
- use direct, terse prose without marketing language; and
- use `\utkarsh{TODO: ...}` for visible notes to self;
- use descriptive `\citep{TODO_xxx}` placeholders for citations Utkarsh will
  supply;
- never invent a bibliography entry or add a real citation unless asked; and
- keep `main.tex` within the ICLR page budget by moving secondary detail to an
  appendix with a concrete forward reference.

Raw measurements remain under `results/raw/`; derived tables remain under
`results/`; paper-facing claims must be traceable to those artifacts. Treat
generated LaTeX files (`main.aux`, `main.bbl`, `main.blg`, `main.fdb_latexmk`,
`main.fls`, `main.log`, `main.out`, and `main.pdf`) as build products unless the
user explicitly asks to commit them.
