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
│   ├── 01_introduction.tex ✅ four-paragraph intro + contributions list
│   ├── 02_related_work.tex ✅ prose complete
│   ├── 03_method.tex       ✅ equations + cold-start fix complete
│   ├── 04_experiments.tex  ✅ all 8 subsections + 4 figures + 3 tables
│   ├── 05_discussion.tex   ✅ four discussion subsections
│   └── 06_conclusion.tex   ✅ three-contribution summary
└── figures/                ✅ 4 figures generated/copied
```

## Build

Two compilable manuscripts are maintained in parallel:

| File | Purpose | Class | Layout |
|---|---|---|---|
| `main.tex` | Development / proof-reading | `article` | single column, 14 pages |
| `main_neurocomputing.tex` | **Submission to Neurocomputing** | `elsarticle` (vendored) | two column, 10 pages |

Both share the same `sections/`, `figures/` and `references.bib`. Either compiles
with the standard MacTeX BasicTeX distribution -- elsarticle.cls is vendored
locally so no `sudo tlmgr install` is required.

```bash
cd paper
# Development build
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Submission build (Neurocomputing two-column)
pdflatex main_neurocomputing.tex && bibtex main_neurocomputing && \
  pdflatex main_neurocomputing.tex && pdflatex main_neurocomputing.tex
```

(Requires a TeX distribution. On macOS: `brew install --cask mactex-no-gui`.)

## Status

**Submission-ready first draft.**
- `main_neurocomputing.tex` compiles to 10 pages in Neurocomputing two-column format with all sections filled and references resolved.
- `main.tex` keeps the article-class single-column build for proof-reading.
- 0 remaining `\TODO` markers in body sections.

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
