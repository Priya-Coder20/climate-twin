"""SHAP explanations for any model (sklearn / XGBoost / Keras / PyTorch LSTM).
`predict_fn` must take a 2-D numpy array (n_samples, n_features) and return (n_samples,).
For an LSTM with input (n, T, F): flatten to (n, T*F) and reshape inside the wrapper."""
import numpy as np, shap, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
 
def make_wrapper(model, T=None, F=None, target_index=0):
    def f(X):
        X = np.asarray(X, dtype="float32")
        if T and F: X = X.reshape(len(X), T, F)
        y = model.predict(X, verbose=0) if hasattr(model, "predict") else model(X)
        y = np.asarray(y)
        return y[:, target_index] if y.ndim > 1 else y
    return f
 
def explain(predict_fn, X_background, X_explain, feature_names, out_prefix="shap_tmax"):
    bg = shap.sample(X_background, 50)                    # keep small: slow for LSTM
    explainer = shap.Explainer(predict_fn, bg, feature_names=feature_names)
    sv = explainer(X_explain[:100])                        # explain <=100 rows
    shap.summary_plot(sv, X_explain[:100], feature_names=feature_names, show=False)
    plt.tight_layout(); plt.savefig(f"{out_prefix}_summary.png", dpi=150); plt.close()
    shap.plots.waterfall(sv[0], show=False)
    plt.tight_layout(); plt.savefig(f"{out_prefix}_local.png", dpi=150); plt.close()
    return sv
 
def top_drivers(sv, feature_names, row=0, k=3):
    """Text-ready top-k drivers for one prediction (feeds the recommendation engine)."""
    vals = sv.values[row]; idx = np.argsort(-np.abs(vals))[:k]
    return [(feature_names[i], float(vals[i])) for i in idx]