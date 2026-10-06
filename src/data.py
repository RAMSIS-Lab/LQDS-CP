from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ALIASES = {
    "casP": "bio",
    "casp": "bio",
    "blog_data": "blog",
    "facebook_1": "fb1",
    "facebook_2": "fb2",
    "meps_19": "meps19",
    "meps_20": "meps20",
    "meps_21": "meps21",
}


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    x_scaler: StandardScaler
    y_scaler: MinMaxScaler
    dataset: str
    feature_names: list[str]
    target_name: str

    @property
    def y_scale(self) -> float:
        return float(self.y_scaler.data_range_[0])

    @property
    def y_offset(self) -> float:
        return float(self.y_scaler.data_min_[0])

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:

    df = pd.get_dummies(df, dummy_na=False, dtype=float)
    return df.apply(pd.to_numeric, errors="coerce")


def load_dataset(
    name: str, data_dir: str | Path
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    name = ALIASES.get(name.lower(), name.lower())
    root = Path(data_dir)

    if name == "synth_hard":
        from .synthetic import make_dataset

        values, y = make_dataset()
        X = pd.DataFrame(values, columns=[f"x{i + 1}" for i in range(values.shape[1])])
    elif name.startswith("meps"):
        year = name[-2:]
        df = pd.read_csv(root / f"meps_{year}_reg.csv")
        y = df.pop("UTILIZATION_reg").to_numpy()
        X = df.drop(columns=["Unnamed: 0"], errors="ignore")
    elif name == "bio":
        df = pd.read_csv(root / "CASP.csv")
        y, X = df.iloc[:, 0].to_numpy(), df.iloc[:, 1:]
    elif name == "blog":
        df = pd.read_csv(root / "blogData_train.csv", header=None)
        y, X = df.iloc[:, -1].to_numpy(), df.iloc[:, :-1]
    elif name in {"fb1", "fb2"}:
        variant = 1 if name == "fb1" else 2
        df = pd.read_csv(
            root / "facebook" / f"Features_Variant_{variant}.csv", header=None
        )
        y, X = df.iloc[:, -1].to_numpy(), df.iloc[:, :-1]
    elif name == "bike":
        df = pd.read_csv(root / "bike_train.csv")
        dt = pd.to_datetime(df.pop("datetime"))
        df["hour"], df["day"], df["month"], df["year"] = (
            dt.dt.hour,
            dt.dt.dayofweek,
            dt.dt.month,
            dt.dt.year - dt.dt.year.min(),
        )
        y = df.pop("count").to_numpy()
        X = df.drop(columns=["casual", "registered"], errors="ignore")
        X = pd.get_dummies(
            X, columns=[c for c in ["season", "weather"] if c in X], dtype=float
        )
    elif name == "star":
        df = pd.read_csv(root / "STAR.csv")
        score_cols = [
            f"{subject}{grade}"
            for subject in ("read", "math")
            for grade in ("k", "1", "2", "3")
        ]

        df = df.dropna().reset_index(drop=True)
        y = df[score_cols].sum(axis=1).to_numpy()
        X = df.drop(columns=score_cols + ["Unnamed: 0", "lunchk"], errors="ignore")
    elif name == "community":
        attributes = pd.read_csv(root / "communities_attributes.csv")[
            "attributes"
        ].str.strip()
        df = pd.read_csv(root / "communities.data", names=attributes, na_values="?")
        df = df.drop(
            columns=["state", "county", "community", "communityname", "fold"],
            errors="ignore",
        )
        y = df.pop("ViolentCrimesPerPop").to_numpy()
        X = df.dropna(axis=1, how="all")
    else:
        raise ValueError(f"Unknown dataset {name!r}")

    X = _numeric_frame(pd.DataFrame(X)).replace([np.inf, -np.inf], np.nan)
    feature_names = [str(column) for column in X.columns]
    valid = np.isfinite(np.asarray(y, dtype=float))
    X, y = X.loc[valid].reset_index(drop=True), np.asarray(y, dtype=np.float32)[valid]
    return X.to_numpy(dtype=np.float32), y, feature_names


def make_splits(
    name: str, data_dir: str | Path, seed: int = 2026, max_rows: int | None = None,
) -> SplitData:
    X, y, feature_names = load_dataset(name, data_dir)
    if max_rows is not None and len(y) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(y), size=max_rows, replace=False))
        X, y = X[idx], y[idx]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.10, random_state=seed
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_dev, y_dev, test_size=2 / 9, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_fit, y_fit, test_size=1 / 7, random_state=seed
    )

    try:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(
            X_train
        )
    except TypeError:

        imputer = SimpleImputer(strategy="median").fit(X_train)
    X_train = imputer.transform(X_train)
    X_val, X_cal, X_test = (imputer.transform(z) for z in (X_val, X_cal, X_test))
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
    tx = lambda z: x_scaler.transform(z).astype(np.float32)
    ty = lambda z: y_scaler.transform(z.reshape(-1, 1)).ravel().astype(np.float32)
    canonical_name = ALIASES.get(name.lower(), name.lower())
    target_names = {
        "star": "total reading + math score",
        "bike": "count",
        "bio": "RMSD",
        "blog": "comments in next 24 hours",
        "community": "ViolentCrimesPerPop",
        "fb1": "Facebook comment count",
        "fb2": "Facebook comment count",
        "meps19": "UTILIZATION_reg",
        "meps20": "UTILIZATION_reg",
        "meps21": "UTILIZATION_reg",
        "synth_hard": "y",
    }
    return SplitData(
        tx(X_train),
        ty(y_train),
        tx(X_val),
        ty(y_val),
        tx(X_cal),
        ty(y_cal),
        tx(X_test),
        ty(y_test),
        x_scaler,
        y_scaler,
        canonical_name,
        feature_names,
        target_names.get(canonical_name, "target"),
    )


def available_datasets() -> list[str]:
    return [
        "bike",
        "bio",
        "blog",
        "fb1",
        "fb2",
        "meps19",
        "meps20",
        "meps21",
        "star",
        "community",
    ]
