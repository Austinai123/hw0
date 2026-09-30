# HW 0: Portfolio Selection (Markowitz, 1952)

This is the practice homework for *FINM 32800: Data Pipelines for Quantitative
Research*. It is ungraded. It accompanies
[Lecture 0 in the course textbook](https://finm-32800.github.io/overview_w0.html),
and the full instructions are on the
[HW 0 page](https://finm-32800.github.io/HW0.html).

It is the pattern the rest of the course repeats at larger scale: a data pull,
an analysis in a notebook, the same logic moved into tested functions, and an
automated check that runs every time you push to GitHub.

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/hw0.git
cd hw0
conda create -n finm python=3.12
conda activate finm
pip install -r requirements.txt
doit                       # download the data and build the notebooks
streamlit run src/app.py   # open the dashboard
pytest                     # run the tests (they fail until you finish the homework)
```

`doit` downloads a cached extract of monthly stock returns from CRSP to
`_data/crsp_monthly_returns.csv`, then executes the notebooks and saves HTML
copies in `_output/`. The notebooks need the data. The dashboard and the tests
do not, so you can run those two commands in any order. Without the data, the
dashboard shows simulated returns.

## What to do

1. Read and run the notebook, `src/01_markowitz.ipynb.py`. It is a Python
   script in the "percent" format, which VS Code runs cell by cell. The
   appendix notebook, `src/02_markowitz_derivation.ipynb.py`, derives the
   formulas.
2. Fill in the two functions marked `TODO` in `src/port_opt.py`.
3. Run `pytest`. When the tests pass, commit and push. The same tests run
   automatically on GitHub Actions; look for the green check mark next to your
   commit.

Do not edit the test files (`src/test_*.py`).

## Pulling the data from WRDS

The cached extract needs no account, so `doit` works before your WRDS access
is approved. To pull the data fresh from WRDS instead, copy `.env.example` to
`.env`, fill in your `WRDS_USERNAME`, set `NO_CACHE=True`, and run `doit`
again. The queries that produced the extract are in `src/pull_crsp.py`.

## What is in this repository

| Path | Purpose |
|---|---|
| `requirements.txt` | The exact package versions this project was tested with |
| `src/pull_crsp.py` | Gets the data: a cached download by default, WRDS with `NO_CACHE=True` |
| `src/01_markowitz.ipynb.py` | The notebook: mean-variance analysis on real data |
| `src/02_markowitz_derivation.ipynb.py` | Appendix notebook: derivation of the formulas |
| `src/mean_variance.py` | Frontier calculations, with and without short sales or a risk-free asset |
| `src/port_opt.py` | The functions you complete |
| `src/test_*.py` | Unit tests |
| `src/app.py` | Streamlit dashboard |
| `src/config.py` | Paths and settings, read from an optional `.env` file |
| `dodo.py` | Task runner file (`doit`): gets the data and builds the notebooks |
| `.github/workflows/tests.yml` | Runs the tests on GitHub Actions on every push |
| `_data/`, `_output/` | Downloaded data and generated files. Safe to delete; never committed |
