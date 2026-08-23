# ICLR 2027 submission

The unmodified official style archive supplied by the user is extracted under
`iclr2027/`. Our anonymous working draft is `iclr2027/main.tex`; the official
formatting-instructions shell remains `iclr2027/iclr2027_conference.tex`.

Important constraints from the official template:

- Initial submission: 9 pages of main text; citations may use additional pages.
- Rebuttal/camera ready: 10 pages of main text.
- Leave `\iclrfinalcopy` disabled and authors anonymous for submission.
- A reproducibility statement is required and does not count toward the limit.
- An ethics statement is recommended and does not count toward the limit.

Build from `paper/submission/iclr2027` with:

```bash
latexmk -pdf main.tex
```

LaTeX is not currently installed in the login environment, so the draft has
not yet been rendered locally.
