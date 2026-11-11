# Online Localized Conformal Prediction

Code to reproduce the experiments in **Online Localized Conformal Prediction**  
Yuheng Lai and Garvesh Raskutti · NeurIPS 2026

## Structure

```text
cp_methods.py                    Conformal prediction methods
notebooks/simulations.ipynb       Synthetic scenarios A–C
notebooks/elec2.ipynb             Electricity transfer
notebooks/ilinet.ipynb            Influenza-like illness
notebooks/etf_volatility.ipynb    ETF volatility and VIX regimes
```

The nine methods are **CP, LCP, ACI, DtACI, COP, LQT, SPCI, OLCP, and OLCP-Hedge**.

## Installation

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab
```

Run any notebook from `notebooks/` (or the project root). The notebooks locate `cp_methods.py` and `data/` automatically. The ILINet dataset is loaded by Darts. Notebook files include embedded tables and figures from the completed experiments and can be viewed directly on GitHub. To reproduce them, run the notebooks top to bottom with the documented dependencies and datasets. Figures and result tables are displayed in notebook cells.

## Experimental settings

All experiments target 90% coverage and use chronological evaluation.

| Dataset | Base predictor | Calibration window | Evaluation |
|---|---|---:|---|
| Simulations A–C | Linear regression | 200 | 1,500 observations, first 500 for training, 100 repetitions |
| ELEC2 | Gradient-boosted regression tree | 100 | 70% training / 30% online test |
| ILINet | Temporal convolutional network (26 lags) | 52 | 70% training / 10% validation / 20% test |
| ETF volatility | EWMA(20) of squared returns | 400 | Five ETFs, testing from January 2020 |


**Evaluation:** Simulations use 100 Monte Carlo repetitions. Real-data uncertainty estimates use 1,000 block-bootstrap replicates, with block lengths 48 (ELEC2), 26 (ILINet), and 20 (ETF). Rolling coverage and interval-width diagnostics and VIX-quartile comparisons are included.


## Data

The fixed benchmark CSVs are in the [`data/` directory of the OLCP repository](https://github.com/yuhenglai/OLCP/tree/main/data). Keep those files when updating this repository. ILINet is loaded with `darts.datasets.ILINetDataset`. See [`data/README.md`](data/README.md) for required file names and sources.

## Citation

If you use this code, please cite:

> Y. Lai and G. Raskutti. *Online Localized Conformal Prediction*. NeurIPS 2026.
