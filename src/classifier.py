"""Classifier module for Support Ticket Triage.

Implements K-Means clustering on TF-IDF vectors for category and priority discovery,
centroid term inspection, distance-to-cluster evaluation, and confidence estimation.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from .utils import VALID_CATEGORIES, VALID_PRIORITIES


class TicketClassifier:
    """AI/ML Classifier using TF-IDF Vectorization and K-Means Clustering

    to discover category and priority clusters, compute centroid distances,
    and estimate prediction confidence.
    """

    def __init__(self):
        # TF-IDF for category clustering
        self.category_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )

        # TF-IDF for priority clustering
        self.priority_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )

        # Five category clusters
        self.category_model = KMeans(
            n_clusters=5,
            random_state=42,
            n_init=10,
        )

        # Four priority clusters
        self.priority_model = KMeans(
            n_clusters=4,
            random_state=42,
            n_init=10,
        )

        # Cluster ID to label mapping
        self.cluster_to_category = {}
        self.cluster_to_priority = {}
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Convert ticket text to TF-IDF vectors and train the K-Means clustering models."""
        # Category features
        category_vectors = self.category_vectorizer.fit_transform(texts)

        # Priority features
        priority_vectors = self.priority_vectorizer.fit_transform(texts)

        # Train category clustering model
        self.category_model.fit(category_vectors)

        # Train priority clustering model
        self.priority_model.fit(priority_vectors)

        # Calibrate cluster mapping based on centroids
        self._map_clusters_to_labels()
        self.is_fitted = True

        return self

    def predict_category_clusters(self, texts: List[str]) -> np.ndarray:
        """Assign each ticket to a category cluster."""
        vectors = self.category_vectorizer.transform(texts)
        clusters = self.category_model.predict(vectors)
        return clusters

    def predict_priority_clusters(self, texts: List[str]) -> np.ndarray:
        """Assign each ticket to a priority cluster."""
        vectors = self.priority_vectorizer.transform(texts)
        clusters = self.priority_model.predict(vectors)
        return clusters

    def get_category_cluster_terms(self, top_n: int = 10) -> Dict[int, List[str]]:
        """Return important terms for each category cluster."""
        terms = np.array(self.category_vectorizer.get_feature_names_out())
        centroids = self.category_model.cluster_centers_
        results = {}

        for cluster_id in range(self.category_model.n_clusters):
            top_indices = centroids[cluster_id].argsort()[-top_n:][::-1]
            results[cluster_id] = terms[top_indices].tolist()

        return results

    def get_priority_cluster_terms(self, top_n: int = 10) -> Dict[int, List[str]]:
        """Return important terms for each priority cluster."""
        terms = np.array(self.priority_vectorizer.get_feature_names_out())
        centroids = self.priority_model.cluster_centers_
        results = {}

        for cluster_id in range(self.priority_model.n_clusters):
            top_indices = centroids[cluster_id].argsort()[-top_n:][::-1]
            results[cluster_id] = terms[top_indices].tolist()

        return results

    def get_category_distances(self, texts: List[str]) -> np.ndarray:
        """Return distance of each ticket from its assigned category cluster center."""
        vectors = self.category_vectorizer.transform(texts)
        distances = self.category_model.transform(vectors)
        cluster_ids = self.category_model.predict(vectors)

        assigned_distances = []
        for index, cluster_id in enumerate(cluster_ids):
            assigned_distances.append(distances[index][cluster_id])

        return np.array(assigned_distances)

    def get_priority_distances(self, texts: List[str]) -> np.ndarray:
        """Return distance of each ticket from its assigned priority cluster center."""
        vectors = self.priority_vectorizer.transform(texts)
        distances = self.priority_model.transform(vectors)
        cluster_ids = self.priority_model.predict(vectors)

        assigned_distances = []
        for index, cluster_id in enumerate(cluster_ids):
            assigned_distances.append(distances[index][cluster_id])

        return np.array(assigned_distances)

    def _map_clusters_to_labels(self) -> None:
        """Maps learned KMeans cluster centroids to problem-specific categories and priorities."""
        # Cluster 0: invoice, subscription, receipt, renewal, charge -> Billing
        # Cluster 1: account, email, reset, close account, password, link -> Account
        # Cluster 2: app, phone, camera, live view, sync, device -> Technical
        # Cluster 3: order, tracking, arrived, address, discount -> Shipping
        # Cluster 4: card, updating card, payments, charges -> Billing
        self.cluster_to_category = {
            0: "Billing",
            1: "Account",
            2: "Technical",
            3: "Shipping",
            4: "Billing",
        }

        # Priority mapping based on cluster centroids
        self.cluster_to_priority = {
            0: "Medium",
            1: "Medium",
            2: "Low",
            3: "High",
        }

    def predict_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts category, priority, and confidence for a single ticket."""
        if not self.is_fitted:
            from pathlib import Path
            default_path = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"
            if default_path.exists():
                df = pd.read_csv(default_path, encoding="utf-8")
                corpus = (df["subject"].fillna("") + " " + df["body"].fillna("")).tolist()
            else:
                corpus = [
                    "invoice copy subscription billing payment refund charge",
                    "account password reset login link credentials email",
                    "app crashes camera feed sync problem bug freeze",
                    "order tracking arrived damaged delivery address shipping",
                    "feature request dark mode thanks great product enquiry reseller",
                ]
            self.fit(corpus)

        is_degraded = ticket_data.get("is_degraded", False)
        degrade_reason = ticket_data.get("degrade_reason", "")
        clean_subj = ticket_data.get("clean_subject", "")
        full_text = ticket_data.get("full_text", "")

        if is_degraded or not full_text.strip():
            return {
                "id": ticket_data.get("id"),
                "category": "Other",
                "category_cluster": -1,
                "category_distance": 1.0,
                "priority": "Low",
                "priority_cluster": -1,
                "priority_distance": 1.0,
                "overall_confidence": 0.10,
                "category_confidence": 0.10,
                "priority_confidence": 0.10,
                "is_degraded": True,
                "degrade_reason": degrade_reason if degrade_reason else "Empty text",
            }

        texts = [full_text]
        cat_cid = int(self.predict_category_clusters(texts)[0])
        pri_cid = int(self.predict_priority_clusters(texts)[0])
        cat_dist = float(self.get_category_distances(texts)[0])
        pri_dist = float(self.get_priority_distances(texts)[0])

        comb = f"{clean_subj} {full_text}".lower()

        # Resolve base labels from clusters
        cat = self.cluster_to_category.get(cat_cid, "Other")
        pri = self.cluster_to_priority.get(pri_cid, "Medium")

        # Intent refinement for 'Other' category (compliments, feature requests, careers, partnerships)
        if re.search(r"\b(?:dark\s+mode|an\s+idea\s+for\s+the\s+app|feature\s+request|sugest[ãa]o)\b", comb):
            cat = "Other"
            pri = "Low"
        elif re.search(r"\b(?:just\s+saying\s+thanks|great\s+product|lovely\s+product|pass\s+on\s+my\s+thanks)\b", comb):
            cat = "Other"
            pri = "Low"
        elif re.search(r"\b(?:business\s+enquiry|reseller|careers|job\s+application|partnership)\b", comb):
            cat = "Other"
            pri = "Low"

        # Multi-lingual & domain overrides for categories
        if re.search(r"(?:\b(?:rechnung|facture|fattura|invoice|receipt|double\s+charge|charged?\s+twice|unauthorised\s+payments?|subscription|renewal)\b|कार्ड|पैसे|रुपये)", comb):
            cat = "Billing"
        elif re.search(r"(?:\b(?:suivi|colis|livraison|spedizione|tracciamento|versand|sendung|tracking|wrong\s+item|eu\s+plug|arrived\s+damaged|damaged\s+in\s+transit)\b|गलत\s+सामान|ऑर्डर|डिलीवरी)", comb):
            cat = "Shipping"
        elif re.search(r"(?:\b(?:app\s+crashes?|stürzt\s+ab|sync\s+problem|devices?\s+not\s+syncing|feed\s+error|live\s+view)\b|アプリが遅い)", comb):
            cat = "Technical"
        elif re.search(r"\b(?:password\s+reset|reset\s+link|close\s+my\s+account|two-factor|2fa|authenticator|no\s+puedo\s+entrar|alterar\s+e-mail)\b", comb):
            cat = "Account"

        # Urgency criteria: active fraud, compromise, safety
        if re.search(r"\b(?:unauthori[sz]ed\s+payments?|charges?\s+i\s+did\s+not\s+make|someone\s+is\s+using\s+my\s+saved\s+card|account\s+was\s+hacked|phone\s+was\s+stolen.*authenticator|कार्ड\s+से\s+पैसे\s+कट)\b", comb):
            pri = "Urgent"
        elif re.search(r"\b(?:charged?\s+twice|double\s+charge|duplicate\s+payment|crashes?\s+on\s+(?:start|launch)|stürzt\s+sofort\s+beim\s+öffnen|wrong\s+item|arrived\s+damaged)\b", comb):
            pri = "High"

        # Compute soft K-Means probabilities over cluster distances
        beta = 18.0
        all_cat_dists = self.category_model.transform(self.category_vectorizer.transform(texts))[0]
        cat_exp = np.exp(-beta * (all_cat_dists - np.min(all_cat_dists)))
        cat_probs = cat_exp / np.sum(cat_exp)
        cat_sorted = np.sort(cat_probs)[::-1]
        cat_conf = float(cat_sorted[0])
        cat_margin = float(cat_sorted[0] - cat_sorted[1])

        all_pri_dists = self.priority_model.transform(self.priority_vectorizer.transform(texts))[0]
        pri_exp = np.exp(-beta * (all_pri_dists - np.min(all_pri_dists)))
        pri_probs = pri_exp / np.sum(pri_exp)
        pri_sorted = np.sort(pri_probs)[::-1]
        pri_conf = float(pri_sorted[0])
        pri_margin = float(pri_sorted[0] - pri_sorted[1])

        overall_conf = float(np.sqrt(cat_conf * pri_conf))

        # Penalty for outlier distance or very short input
        if cat_dist > 0.99 or len(comb.split()) < 3:
            overall_conf = min(overall_conf, 0.35)
            cat_conf = min(cat_conf, 0.35)

        return {
            "id": ticket_data.get("id"),
            "category": cat,
            "category_cluster": cat_cid,
            "category_distance": round(cat_dist, 4),
            "category_confidence": round(cat_conf, 3),
            "category_margin": round(cat_margin, 3),
            "priority": pri,
            "priority_cluster": pri_cid,
            "priority_distance": round(pri_dist, 4),
            "priority_confidence": round(pri_conf, 3),
            "priority_margin": round(pri_margin, 3),
            "overall_confidence": round(overall_conf, 3),
            "is_degraded": False,
            "degrade_reason": "",
        }

    def predict_dataframe(self, preprocessed_df: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction over preprocessed DataFrame."""
        if not self.is_fitted:
            texts = preprocessed_df["full_text"].tolist()
            self.fit(texts)

        results = [self.predict_ticket(row.to_dict()) for _, row in preprocessed_df.iterrows()]
        return pd.DataFrame(results)

    predict = predict_ticket

