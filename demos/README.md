# Demos

These scripts are short, runnable examples for the public API. Each file is
standalone and writes outputs under `demos/demo_outputs/`.

Install plotting support before running demos that save figures:

```bash
pip install -e ".[plot]"
```

Run from the project root:

```bash
python demos/01_basic_regression.py
python demos/02_pinn_monotonicity_regularization.py
python demos/03_classification.py
python demos/04_cross_validation.py
python demos/05_prediction_from_saved_model.py
```

## Files

- `01_basic_regression.py`: data loading, training, saving, prediction, and plotting.
- `02_pinn_monotonicity_regularization.py`: custom PINN-style regularization with first and second derivatives.
- `03_classification.py`: multi-class classification and probability/class prediction.
- `04_cross_validation.py`: grid search with K-fold cross-validation.
- `05_prediction_from_saved_model.py`: loading a saved model for NumPy and torch predictions.
