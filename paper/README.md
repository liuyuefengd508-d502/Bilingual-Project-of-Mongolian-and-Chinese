# Paper — Cold-Start Initialization Unlocks Asymmetric UDA in Mongolian Text Detection

Target: **Neurocomputing** (Elsevier, SCI Q2, IF~6).
Template: `elsarticle` (already loaded by `main.tex`).

## Layout

```
paper/
├── main.tex                # entry point
├── references.bib
├── sections/
│   ├── 00_abstract.tex     ✅ first draft
│   ├── 01_introduction.tex ⚠ contributions list ✅; prose body TODO
│   ├── 02_related_work.tex ⚠ subsection skeleton ✅; prose TODO
│   ├── 03_method.tex       ⚠ key cold-start fix paragraph ✅; equations TODO
│   ├── 04_experiments.tex  ⚠ main results table ✅; other tables/figs TODO
│   ├── 05_discussion.tex   ⚠ skeleton with TODOs
│   └── 06_conclusion.tex   ⚠ skeleton TODO
└── figures/                # to be populated from docs/figures/
```

## Build

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

(Requires a TeX distribution. On macOS: `brew install --cask mactex-no-gui`.)

## Status

**~85% drafted -- complete first draft compiles to 13-page PDF.**
Remaining work is Related Work prose, citation verification, affiliation,
and template switch to elsarticle for final submission.

### Done
- Document skeleton with all sections wired up
- Abstract: complete (one full paragraph capturing all three contributions)
- Introduction: complete (4 paragraphs + contributions list)
- Method: complete (DBNet/EMA/DANN/cold-start/pseudo-weighting with equations)
- Experiments 4.1 Datasets: complete
- Experiments 4.2 Implementation details: complete
- Experiments 4.3 Cross-domain motivation table + prose: complete
- Experiments 4.4 Main results table: complete
- Experiments 4.5 Asymmetric gain analysis: complete
- Experiments 4.6 Long-training analysis + 5-ep argument: complete
- Experiments 4.7 Hyperparameter ablation table + prose: complete
- Experiments 4.8 Qualitative results: complete
- Discussion: complete (4 subsections)
- Conclusion: complete
- Figures: long-run curves PDF generated, dataset overview JPG generated,
  comparison grids copied from `docs/figures/`
- References: stub with DBNet, FCENet, PSENet, DANN, AdaptiveTeacher, MIC,
  USTB Mongolian work
- LaTeX compiles cleanly: 13 pages, no float-too-large warnings.

### TODO before submission

| Section | Item |
|---|---|
| Related work | Fill prose under each subsection (3 TODOs) |
| References | Verify USTB-related citations (author/title/DOI); add cite keys |
| Title page | Replace `\TODO{Affiliation}` with institution |
| Template | Switch \documentclass to `elsarticle` (see comment in main.tex) |
| Acknowledgments | Add if applicable |
| Polish | Single proofreading pass for tone/typos |

## Source data for tables

All numbers in the paper are sourced from `mn-uda-textdet/docs/baseline_results.md`
and the per-experiment JSON metric dumps under `mn-uda-textdet/work_dirs/`.
Cross-reference each table caption against those files when filling in TODOs.
