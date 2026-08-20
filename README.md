# LUCID-Drive

**LUCID-Drive: A Large-Scale Dataset for Language-grounded Causal Inference
and Decision-Making in Autonomous Driving**

Wei Zhang, Xinyu Liu, Sarah A. Johnson, Marc Hofmann, Kai Chen

---

## Overview
LUCID-Drive is a multimodal driving dataset collected across **6 global cities**
on **3 continents** over 18 months.  It comprises **12,847 richly annotated
driving scenarios** (2,312,460 frames) captured by an eight-camera surround-view
system, LiDAR, radar, and HD maps under seven weather conditions and four
lighting states.  The dataset contains **847,293 language annotations** organised
into five causal/counterfactual task types.

## Repository Contents
| Path | Description |
|------|-------------|
| `main.tex` | Main LaTeX source |
| `references.bib` | Bibliography |
| `sections/` | Individual section `.tex` files |
| `figures/` | PNG figures used in the paper |
| `generate_figures.py` | Script to regenerate matplotlib figures |

## Compile
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## License
CC BY-NC 4.0
