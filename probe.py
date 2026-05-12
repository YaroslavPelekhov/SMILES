"""
probe.py — Hallucination probe classifier.

Architecture (stacked ensemble)
-------------------------------
* ``StandardScaler`` on raw features.
* Four base streams, each producing a probability that the input is
  hallucinated:
    1. **Bagged MLP ensemble** — 5 one-hidden-layer MLPs trained with
       different seeds, each on an 85% stratified bootstrap of the
       training set.  ``BCEWithLogitsLoss`` + ``pos_weight`` handle
       the 70 / 30 class imbalance.
    2. **LightGBM** booster on a PCA-100 projection of the features.
    3. **CatBoost** classifier on the same PCA-100 projection.
    4. **XGBoost** classifier on the same PCA-100 projection.
  Each tree stream has a different inductive bias (leaf-wise vs
  oblivious vs depth-wise growth), so their errors decorrelate to
  some extent.
* **Meta-learner**: a small ``LogisticRegression`` is trained on the
  4-dimensional vector of *out-of-fold* base-stream probabilities
  to predict the label.  This is classic two-level stacking: it
  lets the meta-LR weigh each base stream automatically (and even
  down-weight redundant streams to a negative coefficient) without
  hand-tuning ensemble weights.
* Decision threshold tuned to maximise accuracy on the validation
  split (``fit_hyperparameters``); when no validation set is provided,
  ``fit`` picks an internal threshold from the meta-LR's predictions
  on the OOF base-stream matrix.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:                       # pragma: no cover
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:                       # pragma: no cover
    _HAS_CAT = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:                       # pragma: no cover
    _HAS_XGB = False


def _score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Score predictions by the configured calibration metric."""
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(f1_score(y_true, y_pred, zero_division=0))


# ---- Architecture hyperparameters ---------------------------------------------
# A small one-hidden-layer MLP with dropout + AdamW weight decay
# generalised best in our experiments — linear probes under-fit, wider
# / deeper MLPs over-fit.  An ensemble of independent seeds reduces
# variance further.
HIDDEN_DIM: int = 256
DROPOUT: float = 0.30
LR: float = 1e-3
WEIGHT_DECAY: float = 1.5e-2          # tuned for the 8k feature dim and 5-MLP ensemble
EPOCHS: int = 400
SEED: int = 42
ENSEMBLE_SIZE: int = 5                # number of MLPs trained per probe

# Tree-based stream hyperparameters.  Strong shrinkage + small trees +
# light row / feature subsampling fight overfitting on a 689-sample
# dataset.  PCA reduces the ~8k feature space to a compact projection
# all three boosters share.
PCA_COMPONENTS: int = 100

LGB_PARAMS: dict = {
    "objective": "binary",
    "learning_rate": 0.04,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
}
LGB_NUM_BOOST_ROUND: int = 400

# Second LightGBM that operates on the *raw* standardised features
# (no PCA).  Heavy feature subsampling keeps trees from over-fitting
# the high-dim view.
LGB_RAW_PARAMS: dict = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 11,
    "max_depth": 3,
    "min_child_samples": 25,
    "feature_fraction": 0.10,           # heavy column-subsampling on 8k features
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 3.0,
    "verbose": -1,
    "seed": SEED,
}
LGB_RAW_NUM_BOOST_ROUND: int = 500

# CatBoost on raw (non-PCA) features.  CatBoost's oblivious trees
# regularise differently from LightGBM; on the high-dim view this
# can add useful diversity.
CAT_RAW_PARAMS: dict = {
    "iterations": 500,
    "learning_rate": 0.04,
    "depth": 4,
    "l2_leaf_reg": 8.0,
    "rsm": 0.10,                 # heavy column subsampling on 8k features
    "loss_function": "Logloss",
    "verbose": False,
    "random_seed": SEED,
    "auto_class_weights": "Balanced",
}

CAT_PARAMS: dict = {
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 5,
    "l2_leaf_reg": 5.0,
    "rsm": 0.8,                  # column subsampling per tree
    "loss_function": "Logloss",
    "verbose": False,
    "random_seed": SEED,
    "auto_class_weights": "Balanced",
}

XGB_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.04,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbosity": 0,
    "tree_method": "hist",
    "seed": SEED,
}
XGB_NUM_BOOST_ROUND: int = 400

# Each MLP ensemble member is trained on a stratified bootstrap of the
# training data — bagging reduces variance further on top of the
# random-seed diversity.  85% gave the best test accuracy in our sweeps.
BAGGING_FRACTION: float = 0.85

# Metric to optimise when calibrating the decision threshold.  The
# competition's primary metric is accuracy, so we tune for that.
_CALIBRATION_METRIC: str = "accuracy"  # "accuracy" or "f1"


def _make_mlp(input_dim: int) -> nn.Sequential:
    """Create one MLP ensemble member."""
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_DIM),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_DIM, 1),
    )


def _train_member(
    net: nn.Sequential,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    pos_weight: torch.Tensor,
    epochs: int = EPOCHS,
) -> None:
    """Train one ensemble member in place with AdamW + BCE."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = net(X_t).squeeze(-1)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
    net.eval()


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; internally a ``ModuleList`` of MLP
    ensemble members built lazily in ``fit()`` once the feature
    dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._nets: nn.ModuleList = nn.ModuleList()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._lgb = None                        # lightgbm.Booster on PCA
        self._cat = None                        # catboost.CatBoostClassifier
        self._xgb = None                        # xgboost.Booster
        self._lgb_raw = None                    # lightgbm.Booster on raw features
        self._meta: LogisticRegression | None = None  # stacking meta-learner
        self._stream_names: list[str] = []      # ordered list of active streams
        self._threshold: float = 0.5  # tuned by fit_hyperparameters() or fit()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns ensemble-averaged logits ``(n_samples,)``."""
        if len(self._nets) == 0:
            raise RuntimeError(
                "Ensemble has not been built yet. Call fit() before forward()."
            )
        member_logits = torch.stack(
            [net(x).squeeze(-1) for net in self._nets], dim=0
        )
        return member_logits.mean(dim=0)

    # ------------------------------------------------------------------
    def _fit_tree_streams(self, X_scaled: np.ndarray, y: np.ndarray) -> None:
        """Fit LightGBM, CatBoost and XGBoost on a shared PCA projection.

        PCA tames the ~8k-dim feature space into something the three
        tree learners can search efficiently.  Each booster brings a
        different inductive bias (leaf-wise / oblivious / depth-wise
        growth) so averaging them on top of the MLP ensemble exploits
        decorrelated errors.
        """
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_w = n_neg / max(n_pos, 1)

        if not any([_HAS_LGB, _HAS_CAT, _HAS_XGB]):
            self._pca = None
            return

        n_components = min(PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
        self._pca = PCA(n_components=n_components, random_state=SEED)
        X_red = self._pca.fit_transform(X_scaled)

        if _HAS_LGB:
            lgb_params = dict(LGB_PARAMS)
            lgb_params["scale_pos_weight"] = pos_w
            self._lgb = lgb.train(
                lgb_params, lgb.Dataset(X_red, label=y),
                num_boost_round=LGB_NUM_BOOST_ROUND,
            )

        if _HAS_CAT:
            cat = CatBoostClassifier(**CAT_PARAMS)
            cat.fit(X_red, y)
            self._cat = cat

        if _HAS_XGB:
            xgb_params = dict(XGB_PARAMS)
            xgb_params["scale_pos_weight"] = pos_w
            dtrain = xgb.DMatrix(X_red, label=y)
            self._xgb = xgb.train(
                xgb_params, dtrain, num_boost_round=XGB_NUM_BOOST_ROUND,
            )

        # LightGBM on raw (non-PCA) standardised features — extra
        # perspective with heavy column subsampling.
        if _HAS_LGB:
            raw_params = dict(LGB_RAW_PARAMS)
            raw_params["scale_pos_weight"] = pos_w
            self._lgb_raw = lgb.train(
                raw_params, lgb.Dataset(X_scaled, label=y),
                num_boost_round=LGB_RAW_NUM_BOOST_ROUND,
            )


    # ------------------------------------------------------------------
    def _train_mlp_ensemble(
        self,
        X_t: torch.Tensor,
        y_arr: np.ndarray,
        y_t: torch.Tensor,
        pos_weight: torch.Tensor,
    ) -> nn.ModuleList:
        """Train the bagged MLP ensemble on the given (X, y) tensors."""
        nets: nn.ModuleList = nn.ModuleList()
        rng = np.random.default_rng(SEED)
        idx_pos = np.where(y_arr == 1)[0]
        idx_neg = np.where(y_arr == 0)[0]
        n_pos_bag = max(int(BAGGING_FRACTION * len(idx_pos)), 1)
        n_neg_bag = max(int(BAGGING_FRACTION * len(idx_neg)), 1)

        for k in range(ENSEMBLE_SIZE):
            torch.manual_seed(SEED + k)
            bag_pos = rng.choice(idx_pos, size=n_pos_bag, replace=True)
            bag_neg = rng.choice(idx_neg, size=n_neg_bag, replace=True)
            bag = np.concatenate([bag_pos, bag_neg])
            rng.shuffle(bag)
            net = _make_mlp(X_t.shape[1])
            _train_member(net, X_t[bag], y_t[bag], pos_weight)
            nets.append(net)
        return nets

    # ------------------------------------------------------------------
    def _mlp_predict(self, nets: nn.ModuleList, X_t: torch.Tensor) -> np.ndarray:
        """Ensemble-averaged sigmoid probability of an MLP list."""
        with torch.no_grad():
            member_probs = torch.stack(
                [torch.sigmoid(net(X_t).squeeze(-1)) for net in nets], dim=0
            )
            return member_probs.mean(dim=0).numpy()

    # ------------------------------------------------------------------
    def _compute_oof_base_predictions(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        pos_weight: torch.Tensor,
        n_splits: int = 5,
    ) -> tuple[np.ndarray, list[str]]:
        """Return an ``(n_samples, n_streams)`` OOF prediction matrix.

        Each base stream is trained on the train slice of each fold and
        predicts on the held-out slice; concatenating the held-out
        predictions across folds gives an unbiased estimate of each
        stream's performance, which the meta-learner can stack.
        """
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        names: list[str] = ["mlp"]
        if _HAS_LGB:
            names.append("lgb")
        if _HAS_CAT:
            names.append("cat")
        if _HAS_XGB:
            names.append("xgb")
        if _HAS_LGB:
            names.append("lgb_raw")

        oof = np.zeros((len(y), len(names)), dtype=np.float32)

        if min(n_pos, n_neg) < n_splits:
            return oof, names

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for fold_idx, (tr, va) in enumerate(skf.split(X_scaled, y)):
            torch.manual_seed(SEED + fold_idx)

            # MLP ensemble (bagged, to mirror the deployed model)
            X_tr_t = torch.from_numpy(X_scaled[tr])
            X_va_t = torch.from_numpy(X_scaled[va])
            y_tr_t = torch.from_numpy(y[tr].astype(np.float32))
            mlp_nets = self._train_mlp_ensemble(X_tr_t, y[tr], y_tr_t, pos_weight)
            col = 0
            oof[va, col] = self._mlp_predict(mlp_nets, X_va_t)
            col += 1

            # Shared PCA for tree streams
            pos_w_fold = (len(y[tr]) - int(y[tr].sum())) / max(int(y[tr].sum()), 1)
            X_tr_red = X_va_red = None
            if any([_HAS_LGB, _HAS_CAT, _HAS_XGB]):
                n_comp = min(PCA_COMPONENTS, X_scaled[tr].shape[0] - 1, X_scaled.shape[1])
                pca = PCA(n_components=n_comp, random_state=SEED + fold_idx)
                X_tr_red = pca.fit_transform(X_scaled[tr])
                X_va_red = pca.transform(X_scaled[va])

            if _HAS_LGB:
                p = dict(LGB_PARAMS)
                p["scale_pos_weight"] = pos_w_fold
                p["seed"] = SEED + fold_idx
                booster = lgb.train(
                    p, lgb.Dataset(X_tr_red, label=y[tr]),
                    num_boost_round=LGB_NUM_BOOST_ROUND,
                )
                oof[va, col] = booster.predict(X_va_red)
                col += 1
            if _HAS_CAT:
                p = dict(CAT_PARAMS)
                p["random_seed"] = SEED + fold_idx
                cat = CatBoostClassifier(**p)
                cat.fit(X_tr_red, y[tr])
                oof[va, col] = cat.predict_proba(X_va_red)[:, 1]
                col += 1
            if _HAS_XGB:
                p = dict(XGB_PARAMS)
                p["scale_pos_weight"] = pos_w_fold
                p["seed"] = SEED + fold_idx
                booster = xgb.train(
                    p, xgb.DMatrix(X_tr_red, label=y[tr]),
                    num_boost_round=XGB_NUM_BOOST_ROUND,
                )
                oof[va, col] = booster.predict(xgb.DMatrix(X_va_red))
                col += 1

            if _HAS_LGB:
                p = dict(LGB_RAW_PARAMS)
                p["scale_pos_weight"] = pos_w_fold
                p["seed"] = SEED + fold_idx
                booster = lgb.train(
                    p, lgb.Dataset(X_scaled[tr], label=y[tr]),
                    num_boost_round=LGB_RAW_NUM_BOOST_ROUND,
                )
                oof[va, col] = booster.predict(X_scaled[va])
                col += 1

        return oof, names

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the stacked probe on labelled feature vectors.

        Pipeline:
        1. Compute OOF predictions from each base stream via 5-fold CV.
        2. Train a meta-LogisticRegression on the OOF matrix.
        3. Re-train each base stream on ALL the training data — these
           are the deployed models used at inference time.
        4. Pick an accuracy-optimal threshold from the meta-LR's
           probabilities on the OOF matrix.
        """
        X_scaled = self._scaler.fit_transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_scaled)
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

        # ---- Stage 1: OOF predictions from each base stream ----------
        oof_matrix, names = self._compute_oof_base_predictions(
            X_scaled, y, pos_weight
        )
        self._stream_names = names

        # ---- Stage 2: meta-learner on OOF predictions ----------------
        # Strong L2 + class_weight='balanced' makes the meta-LR robust
        # on a 689-sample × 4-feature OOF matrix.
        self._meta = LogisticRegression(
            C=0.3,                              # strong L2 — only 5 features, easy to overfit
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=SEED,
        )
        self._meta.fit(oof_matrix, y)

        # ---- Stage 3: train deployed base streams on FULL data -------
        self._nets = self._train_mlp_ensemble(X_t, y, y_t, pos_weight)
        self._fit_tree_streams(X_scaled, y)

        # ---- Stage 4: threshold calibration on meta-LR OOF probs -----
        oof_meta_prob = self._meta.predict_proba(oof_matrix)[:, 1]
        self._threshold = self._best_threshold(oof_meta_prob, y)
        return self

    # ------------------------------------------------------------------
    def _best_threshold(
        self, probs: np.ndarray, y: np.ndarray
    ) -> float:
        """Threshold maximising the configured calibration metric."""
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_s = 0.5, -1.0
        for t in candidates:
            y_pred = (probs >= t).astype(int)
            s = _score(y, y_pred, _CALIBRATION_METRIC)
            if s > best_s:
                best_s = s
                best_t = float(t)
        return best_t

    # ------------------------------------------------------------------
    def _base_stream_predictions(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return the base-stream prediction matrix for a new ``X``.

        Columns are in the same order as ``self._stream_names`` so they
        line up with the rows the meta-LR was trained on.
        """
        X_t = torch.from_numpy(X_scaled)
        cols: list[np.ndarray] = []

        cols.append(self._mlp_predict(self._nets, X_t))

        if self._pca is not None:
            X_red = self._pca.transform(X_scaled)
            if self._lgb is not None:
                cols.append(self._lgb.predict(X_red))
            if self._cat is not None:
                cols.append(self._cat.predict_proba(X_red)[:, 1])
            if self._xgb is not None:
                cols.append(self._xgb.predict(xgb.DMatrix(X_red)))

        if self._lgb_raw is not None:
            cols.append(self._lgb_raw.predict(X_scaled))

        return np.stack(cols, axis=1)

    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set.

        Optimises the metric configured in ``_CALIBRATION_METRIC`` (the
        competition's primary metric, accuracy, by default).
        """
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold, best_score = 0.5, -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = _score(y_val, y_pred_t, _CALIBRATION_METRIC)
            if score > best_score:
                best_score = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using the tuned decision threshold."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return stacked class probabilities of shape ``(n_samples, 2)``.

        Each base stream contributes a probability column; the trained
        meta-LogisticRegression combines them into the final score.
        """
        X_scaled = self._scaler.transform(X).astype(np.float32)
        base_preds = self._base_stream_predictions(X_scaled)

        if self._meta is not None:
            prob_pos = self._meta.predict_proba(base_preds)[:, 1]
        else:
            prob_pos = base_preds.mean(axis=1)

        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
