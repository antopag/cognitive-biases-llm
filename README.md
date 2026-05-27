# Cognitive Biases in Large Language Models

Companion repository for the paper:

> **Cognitive Biases in Large Language Models: A Decision-Theoretic Framework with Empirical Evidence**
> Antonio Pagliaro (2026). *Electronics*.

This repository contains all prompts, raw model responses, analysis code, and the **Bias Strength Index (BSI)** computation framework used in the study. It enables full reproduction of every figure, table, and statistic in the paper.

---

## What is in this repository

```
.
├── code/                          # Python implementation
│   ├── biases.py                  # Definitions of the 11 cognitive biases and their prompt variants
│   ├── metrics.py                 # BSI metric + uncertainty decomposition
│   ├── models.py                  # Unified API client for the 8 evaluated LLMs
│   ├── experiment.py              # Main experiment driver
│   ├── analysis.py                # Result aggregation and figure generation
│   ├── glmm_analysis.py           # Trial-level GLMM significance tests
│   ├── bsi_sensitivity_analysis.py  # Robustness checks for the BSI definition
│   ├── debiasing.py               # CoT / counter-prompting / role-based debiasing
│   ├── run_*.py                   # Reproducible entry-points per experiment
│   ├── environment.yml            # Conda environment specification
│   ├── requirements.txt           # pip alternative
│   └── .env.example               # API-key template (copy to .env and fill in)
└── results/
    ├── raw_responses/             # JSON of every API call (>70,000 responses)
    ├── figures/                   # All figures used in the paper (PDF)
    ├── bsi_summary.csv            # Aggregated BSI per (bias, model, strategy)
    ├── bsi_table.tex              # LaTeX table source used in the paper
    ├── glmm_analysis/             # GLMM numerical/categorical results
    ├── bsi_sensitivity_analysis/  # Sensitivity to BSI parameter choices
    ├── overdispersion_analysis/   # ICC and sensitivity analysis
    └── parse_failure_analysis/    # Parse failure rates per (model, condition)
```

## Cognitive biases evaluated

The benchmark covers eleven biases, organised into four functional categories:

| Category | Biases |
|---|---|
| **Probability / frequency estimation** | anchoring, availability heuristic, representativeness |
| **Choice under uncertainty** | framing effect, sunk cost fallacy, status quo bias, decoy effect |
| **Information integration** | confirmation bias, bandwagon effect, authority bias |
| **Sequence / position effects** | primacy/recency |

Each bias is probed with several semantically equivalent prompt variants and N = 100 trials per variant, enabling decomposition of total uncertainty into a **statistical** component (finite sampling) and a **systematic** component (prompt sensitivity).

## Models evaluated

GPT-4.1 Mini, Claude 3.5 Sonnet, Gemini 2.5 Flash, Llama 3.3 70B, Llama 3.1 8B, Mistral Large, DeepSeek V3, MiniMax M2.5.

## Reproduction

### 1. Environment

```bash
conda env create -f code/environment.yml
conda activate cognitive-biases-llm
```

or with pip:

```bash
pip install -r code/requirements.txt
```

### 2. API keys

Copy the template and fill in your own keys for the providers you want to query:

```bash
cp code/.env.example code/.env
# then edit code/.env
```

You only need keys for the providers you intend to call (e.g. `OPENAI_API_KEY` for GPT-4.1 Mini, `ANTHROPIC_API_KEY` for Claude, etc.). Without keys you can still re-analyse the bundled raw responses.

### 3. Reproduce the analysis from cached responses

All raw model outputs are bundled under `results/raw_responses/`, so the figures and tables can be regenerated **without re-running the API calls**:

```bash
cd code
python analysis.py                       # main figures
python glmm_analysis.py                  # GLMM trial-level tests
python bsi_sensitivity_analysis.py       # BSI robustness checks
python analyze_debiasing.py              # Debiasing figure + table
python parse_failure_analysis.py         # Parse-failure diagnostics
```

### 4. Re-run the experiment (optional, requires API keys)

```bash
cd code
python run_production.py                 # baseline runs (≈70k API calls)
python run_debias_cot.py                 # CoT debiasing on Llama 3.1 8B
python run_debias_counter.py             # counter-prompting debiasing
python run_debias_role.py                # role-based debiasing
```

## Bias Strength Index (BSI)

The BSI is a normalised, continuous metric in `[0, 1]` quantifying the magnitude of a bias as the standardised deviation of model behaviour from the rational baseline. Statistical and systematic uncertainties are reported alongside every point estimate. See `code/metrics.py` and Section 3 of the paper for the full definition.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{pagliaro2026cognitivebiasllm,
  title   = {Cognitive Biases in Large Language Models: A Decision-Theoretic Framework with Empirical Evidence},
  author  = {Pagliaro, Antonio},
  journal = {Electronics},
  year    = {2026}
}
```

The repository is also permanently archived on Zenodo (DOI to be issued upon release tagging).

## License

- **Code** (`code/`): MIT License — see [LICENSE](LICENSE).
- **Data** (`results/`, prompts in `code/biases.py`): Creative Commons Attribution 4.0 International (CC BY 4.0) — see [LICENSE-DATA](LICENSE-DATA).

## Contact

Antonio Pagliaro — [antonio.pagliaro@inaf.it](mailto:antonio.pagliaro@inaf.it) — INAF IASF Palermo
