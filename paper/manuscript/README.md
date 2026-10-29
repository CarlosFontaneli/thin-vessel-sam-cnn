# LaTeX source of the paper

The paper as submitted for publication (`revtex4-1`), with its comments removed.

```
main.tex          preamble, title, authors, abstract; inputs the sections
sections/         01_introduction … 06_appendix
references.bib    bibliography (BibTeX)
figures/          figure PDFs and table .tex files, generated (see below)
```

**Compile** from the repository root with `make paper` (writes `outputs/paper/main.pdf`), or here with
`latexmk -pdf main.tex`. It needs TeX Live with `latexmk`, BibTeX, the `revtex4-1` class and the packages loaded by
`main.tex`.

**`figures/` is generated.** `make figures` (`../figures_tables/run_all.sh`) writes every file in it from the tables
in `results/`; do not edit them by hand. They are committed so that the paper compiles from a clone. Which script
writes which float: [`../README.md`](../README.md).

**Licence.** The text and figures of the paper are not under the MIT licence of the code; see
[`../README.md`](../README.md#licence-of-the-paper).
