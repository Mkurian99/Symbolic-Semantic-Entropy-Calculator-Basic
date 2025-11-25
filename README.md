# Symbolic-Semantic-Entropy-Calculator-Basic
Mathematical framework for quantifying meaning density in text. Users assemble motif categories based on relevant tokens. System evaluates motif presence in excess to a global baseline average. Outputs heatmaps and line graphs which measure semantic activity. Theoretically foregrounded by Lubomír Doležel, Claude Shannon, and Carl Jung. 


# Symbolic Entropy (SE) Calculator

**Quantifying meaning density in text through information theory**

---

## What is Symbolic Entropy?

Symbolic Entropy extends Shannon's 1948 information theory to measure **semantic content** in text—what Shannon deliberately excluded as the "Level B problem."

SE measures two dimensions:
```
SE = (H, Σ)
```

- **H (Shannon Entropy)** — Lexical diversity (bits/token)
  - High H = rich vocabulary, unpredictable language
  - Low H = repetitive, simple language

- **Σ (Sigma/KL Divergence)** — Archetypal pattern concentration (bits/token)
  - High Σ = dense clustering of meaningful motifs
  - Low Σ = scattered, unfocused content spanning few motifs

**Key insight:** Meaning emerges where linguistic variety (H) intersects with semantic clustering (Σ).

---

## Why This Matters

- **Falsifiable:** Σ collapses on shuffled text (proves we're measuring structure, not just word frequency)
- **Information-theoretic:** Uses proper Shannon entropy, and Sigma KL divergence with consistent units
- **Interpretable:** Can trace exactly which motifs drive each peak
- **Universal:** Works on literature, philosophy, religious texts, clinical language

---

## Installation
```bash
pip install numpy pandas matplotlib scipy python-docx
```

---

## Usage
```bash
python SE_Master_Calculator.py your_text_file.txt
```

**Outputs:**
- `your_text_se_heatmap.png` — Visual map of motif clustering
- `your_text_se_timeseries.png` — H and Σ over text progression
- `your_text_peaks_valleys.png` — Top semantic moments with text excerpts
- `your_text_se_results.csv` — Complete numerical data

---

## How It Works

1. **Text is tokenized** into semantic units (multi-word phrases like "tree of life" = 1 token)

2. **Sliding windows** analyze local regions with 50% overlap

3. **For each window:**
   - Calculate H (Shannon entropy) from word probabilities
   - Calculate Σ (KL divergence) comparing local motif distribution to global baseline

4. **Peaks identify** narratively significant moments where archetypal patterns cluster

---

## The Motif Dictionary

The calculator uses **archetypal or motif categories** to detect semantic patterns. Default example analyzes Genesis 1-3 with categories like:

- Tree/Axis-Mundi (tree of life, tree of knowledge, fruit of life)
- Waters/Sea (primordial chaos, deep)
- Light/Fire (cosmic order, day/night)
- Serpent/Beast (chaos creatures)
- Divine/Sacred (God, holy, eternal)
- Etc. 

Motifs are defined by users and tokens and phrases that index them are assigned to motif categories 
**To analyze your own text:** Edit the `motif_dict` section (line 110) with categories relevant to your domain.

---

## Validation

**The Falsification Test:**

When text is word-shuffled (destroying narrative structure while preserving vocabulary):
- H remains approximately the same
- **Σ collapses toward zero** (typically 10-20x reduction)

This proves SE measures semantic architecture, not mere word frequency.

---



---

## Academic Context

Developed by Kurian, M.A. (MA in Religion, Rice University) 

**Theoretical grounding:**
- Shannon (1948) — Information theory foundation
- Lubomír Doležel (1981) — Information-theoretic semantics
- Jung — Archetypal psychology
- German Idealism (Goethe, Hegel) — Pattern and meaning

---

## Citation
```bibtex
@software{kurian2025symbolic,
  author = {Kurian, Michael A.},
  title = {Symbolic Entropy: Quantifying Meaning Density in Text},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/symbolic-entropy},
  version = {2.0.0}
}
```

---

## License

MIT License — Free to use with attribution

*"Completing Shannon's project: measuring not just information transmission, but meaning itself."*
