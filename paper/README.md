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

**~30% drafted (skeleton + abstract + main table + key method paragraph).**

### Done
- Document skeleton with all sections wired up
- Abstract first draft (one full paragraph, captures all three contributions)
- Contributions list (introduction)
- Cold-start initialization key paragraph (method)
- Main results table (Table 2)
- Cross-domain motivation table (Table 1)
- Bibliography stub with key references

### TODO before submission

| Section | Item |
|---|---|
| Intro | 4-paragraph prose body (motivation → why UDA → our findings → outline) |
| Related work | Fill prose under each subsection; verify all `\cite` keys |
| Method | DBNet/EMA/DANN equations; pseudo-label confidence filter formal definition |
| Experiments 4.1 | Dataset table + 1 illustrative image per domain |
| Experiments 4.2 | Implementation details paragraph (env, optimiser, schedule, walltime) |
| Experiments 4.5 | Asymmetric gain analysis paragraphs (currently TODO block) |
| Experiments 4.6 | Long-training plot (PNG export from log files) |
| Experiments 4.7 | v1/v2/v3 ablation table |
| Experiments 4.8 | Insert + caption qualitative figures from `docs/figures/` |
| Discussion | All four subsections' prose |
| Conclusion | Final paragraph (1 page max) |
| References | Verify USTB-related citations; complete author names/titles |
| Figures | Convert `compare_*.jpg` to `figures/` (rename + maybe shrink) |
| Acknowledgments | If applicable |
| Affiliation | Replace `\TODO{Affiliation}` with the correct department/institution |

## Source data for tables

All numbers in the paper are sourced from `mn-uda-textdet/docs/baseline_results.md`
and the per-experiment JSON metric dumps under `mn-uda-textdet/work_dirs/`.
Cross-reference each table caption against those files when filling in TODOs.
