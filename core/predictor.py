from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class InflationPredictor(ABC):
    """Contract that every inflation predictor model must fulfill."""

    @abstractmethod
    def predict(self, country: str, year: int, month: int) -> float | None: ...

    @abstractmethod
    def countries(self) -> list[dict]: ...


class FoodInflationPredictor(InflationPredictor):
    """
    Wraps a trained model. Accepts (country, year, month) and returns real inflation.
    Lag computation and log-inverse transform are handled internally.
    """

    def __init__(self, model, le, features, df_feat, tails, last_dates):
        self._model = model
        self._le = le
        self._features = features
        self._df_feat = df_feat
        self._tails = tails            # {country: last 12 inflation values}
        self._last_dates = last_dates  # {country: last known Timestamp}

    @classmethod
    def build(cls, model, le, features, df):
        tails = {}
        last_dates = {}
        for country, group in df.groupby("country"):
            g = group.sort_values("date")
            tails[country] = list(g["inflation"].tail(12).values)
            last_dates[country] = g["date"].max()
        return cls(model, le, features, df, tails, last_dates)

    def last_date(self, country: str) -> pd.Timestamp | None:
        return self._last_dates.get(country)

    def last_value(self, country: str) -> float | None:
        tail = self._tails.get(country)
        return float(tail[-1]) if tail else None

    def countries(self) -> list[dict]:
        mapping = (
            self._df_feat[["country", "country_name"]]
            .drop_duplicates()
            .sort_values("country_name")
        )
        return [{"code": r.country, "name": r.country_name} for r in mapping.itertuples()]

    def predict(self, country: str, year: int, month: int) -> float | None:
        # Si target_date está en los datos históricos, predice directamente con las features precalculadas.
        # Si está en el futuro, predice mes a mes desde last_date hasta target_date,
        # usando cada predicción como lag del mes siguiente — necesario porque el modelo
        # requiere features de lag (lag1..lag12, rolling means, yoy) que aún no existen.
        if country not in self._le.classes_:
            return None

        country_enc = int(self._le.transform([country])[0])
        target_date = pd.Timestamp(year=year, month=month, day=1)
        last_date = self._last_dates.get(country)

        if last_date is None:
            return None

        if target_date <= last_date:
            row = self._df_feat[
                (self._df_feat["country"] == country) &
                (self._df_feat["date"] == target_date)
            ]
            if row.empty:
                return None
            pred_log = float(self._model.predict(row[self._features].values)[0])
            return round(float(np.expm1(pred_log)), 2)

        recent = list(self._tails[country])
        cursor = last_date + pd.DateOffset(months=1)
        while cursor <= target_date:
            n = len(recent)
            lag1  = recent[-1]  if n >= 1  else 0
            lag2  = recent[-2]  if n >= 2  else 0
            lag3  = recent[-3]  if n >= 3  else 0
            lag6  = recent[-6]  if n >= 6  else 0
            lag12 = recent[-12] if n >= 12 else 0
            rm3   = np.mean(recent[-3:])  if n >= 3  else np.mean(recent)
            rm12  = np.mean(recent[-12:]) if n >= 12 else np.mean(recent)
            yoy   = recent[-1] - recent[-12] if n >= 12 else 0

            X = np.array([[country_enc, cursor.year, cursor.month,
                           lag1, lag2, lag3, lag6, lag12, rm3, rm12, yoy]])
            pred_log = float(self._model.predict(X)[0])
            recent.append(float(np.expm1(pred_log)))
            cursor += pd.DateOffset(months=1)

        return round(recent[-1], 2)
