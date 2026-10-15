"""Synthetic issue triage using TF-IDF similarity and ridge regression."""
from __future__ import annotations

import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SEED = 42
TOPICS = {
    "authentication": ["login timeout after password reset", "sign in request times out", "authentication fails after reset", "user cannot log in after reset", "session expires during login"],
    "billing": ["invoice total is incorrect", "billing page shows wrong amount", "charged twice for monthly plan", "duplicate subscription payment", "invoice tax calculation is wrong"],
    "api": ["rate limit header missing", "api returns 429 too early", "request quota is not reset", "rate limiter ignores plan", "api throttles valid requests"],
    "database": ["connection pool is exhausted", "database connections leak", "query times out under load", "pool does not release connections", "database reconnect loop"],
    "export": ["csv export omits final rows", "downloaded report is truncated", "export misses recent records", "csv file has incomplete data", "report download ends early"],
    "notifications": ["email notification arrives twice", "duplicate alert emails sent", "webhook event is delivered twice", "notification retry creates duplicate", "same message is sent two times"],
    "permissions": ["viewer can edit project settings", "read only user changes configuration", "role permissions are ignored", "member gets admin access", "unauthorized settings update succeeds"],
    "search": ["search results ignore filters", "date filter returns old records", "search query drops status filter", "results do not match selected range", "filter is missing from search request"],
}


def _issue_data(seed: int = SEED):
    rng = random.Random(seed)
    texts, hours = [], []
    for topic, phrases in TOPICS.items():
        base = {"authentication": 5, "billing": 8, "api": 6, "database": 12, "export": 4, "notifications": 5, "permissions": 7, "search": 4}[topic]
        for i in range(32):
            phrase = rng.choice(phrases)
            texts.append(f"{phrase}; build {i % 7}; environment {('linux', 'windows', 'cloud')[i % 3]}")
            hours.append(max(0.5, base + 0.22 * len(phrase.split()) + rng.gauss(0, 1.8)))
    return texts, np.asarray(hours)


def _pairs(seed: int = SEED):
    rng = random.Random(seed + 1)
    items = list(TOPICS.items())
    pairs, labels = [], []
    for _ in range(480):
        if rng.random() < 0.5:
            _, phrases = rng.choice(items)
            a, b = rng.sample(phrases, 2)
            label = 1
        else:
            (_, first), (_, second) = rng.sample(items, 2)
            a, b = rng.choice(first), rng.choice(second)
            label = 0
        pairs.append((a, b))
        labels.append(label)
    return pairs, np.asarray(labels)


class IssuePairClassifier(ClassifierMixin, BaseEstimator):
    """Fit TF-IDF on individual issue texts and classify pair similarity."""
    def __init__(self, random_state: int = SEED):
        self.random_state = random_state

    @staticmethod
    def _features(pairs, vectorizer, fit=False):
        left, right = zip(*pairs)
        a = vectorizer.fit_transform(left) if fit else vectorizer.transform(left)
        b = vectorizer.transform(right)
        dot = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        na = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
        nb = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
        cosine = dot / np.maximum(na * nb, 1e-12)
        overlap = []
        for x, y in pairs:
            tx, ty = set(x.lower().split()), set(y.lower().split())
            overlap.append(len(tx & ty) / max(1, len(tx | ty)))
        return np.column_stack([cosine, overlap, np.minimum(na, nb) / np.maximum(np.maximum(na, nb), 1e-12)])

    def fit(self, X, y):
        self.vectorizer_ = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        features = self._features(X, self.vectorizer_, fit=True)
        self.classifier_ = LogisticRegression(class_weight="balanced", random_state=self.random_state)
        self.classifier_.fit(features, y)
        self.classes_ = self.classifier_.classes_
        return self

    def predict(self, X):
        return self.classifier_.predict(self._features(X, self.vectorizer_))


def sample_input():
    return {"issue_a": "login timeout after password reset", "issue_b": "sign in request times out", "issue": "query times out under load"}


def train_model(seed: int = SEED):
    pairs, labels = _pairs(seed)
    texts, hours = _issue_data(seed)
    duplicate = IssuePairClassifier().fit(pairs, labels)
    resolution = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), Ridge(alpha=2.0))
    resolution.fit(texts, hours)
    return {"duplicate": duplicate, "resolution": resolution}


def predict(model, payload):
    pair = [(payload["issue_a"], payload["issue_b"])]
    return {
        "duplicate": bool(model["duplicate"].predict(pair)[0]),
        "estimated_resolution_hours": round(float(model["resolution"].predict([payload["issue"]])[0]), 2),
    }


def evaluate(seed: int = SEED):
    pairs, labels = _pairs(seed)
    p_train, p_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.25, random_state=seed, stratify=labels)
    duplicate = IssuePairClassifier().fit(p_train, y_train)
    dup_pred = duplicate.predict(p_test)
    texts, hours = _issue_data(seed)
    x_train, x_test, r_train, r_test = train_test_split(texts, hours, test_size=0.25, random_state=seed)
    resolution = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), Ridge(alpha=2.0)).fit(x_train, r_train)
    pred = resolution.predict(x_test)
    return {
        "duplicate_accuracy": round(float(accuracy_score(y_test, dup_pred)), 4),
        "duplicate_f1": round(float(f1_score(y_test, dup_pred)), 4),
        "resolution_mae_hours": round(float(mean_absolute_error(r_test, pred)), 4),
        "resolution_rmse_hours": round(float(np.sqrt(mean_squared_error(r_test, pred))), 4),
        "resolution_r2": round(float(r2_score(r_test, pred)), 4),
        "test_rows": int(len(y_test) + len(r_test)),
    }
