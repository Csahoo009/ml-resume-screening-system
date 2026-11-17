"""
Unsupervised Clustering & Talent Pool Profiling
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


TALENT_POOL_NAMES = {
    0: "Senior Technical Leaders",
    1: "Specialized Engineering Experts",
    2: "Emerging High-Potential Talent",
    3: "Core Operations & Delivery"
}


class TalentPoolClusterer:
    """
    Unsupervised talent pool clustering using K-Means & Hierarchical Clustering.
    Evaluates clusters using Silhouette analysis and Elbow method inertia.
    """
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.cluster_profiles_ = {}
        self.cluster_names_ = {}

    def fit(self, X, original_df=None):
        self.kmeans.fit(X)
        kmeans_labels = self.kmeans.labels_
        
        # Profile clusters if original dataframe provided
        if original_df is not None:
            self._generate_profiles(original_df, kmeans_labels)
        return self

    def predict(self, X):
        return self.kmeans.predict(X)

    def _generate_profiles(self, df, labels):
        df_copy = df.copy()
        df_copy['cluster'] = labels
        
        profiles = {}
        # Order clusters by average experience and scores to assign descriptive persona names
        cluster_stats = df_copy.groupby('cluster').agg({
            'experience_years': 'mean',
            'technical_test_score': 'mean',
            'ats_score': 'mean',
            'interview_score': 'mean',
            'expected_salary': 'mean'
        })
        
        # Determine descriptive names dynamically based on stats
        sorted_by_exp = cluster_stats.sort_values(by='experience_years', ascending=False).index.tolist()
        
        name_mapping = {}
        if len(sorted_by_exp) == 4:
            name_mapping[sorted_by_exp[0]] = "Senior Technical Leaders"
            name_mapping[sorted_by_exp[1]] = "Specialized Engineering Experts"
            name_mapping[sorted_by_exp[2]] = "Core Delivery Contributors"
            name_mapping[sorted_by_exp[3]] = "Emerging High-Potential Talent"
        else:
            for i, c_idx in enumerate(sorted_by_exp):
                name_mapping[c_idx] = f"Talent Pool Group {i+1}"
                
        self.cluster_names_ = name_mapping
        
        for c in range(self.n_clusters):
            c_data = df_copy[df_copy['cluster'] == c]
            profiles[c] = {
                'cluster_id': int(c),
                'persona_name': name_mapping.get(c, f"Talent Pool {c}"),
                'size': int(len(c_data)),
                'percentage': round(len(c_data) / len(df_copy) * 100, 2),
                'avg_experience_years': round(float(c_data['experience_years'].mean()), 2),
                'avg_technical_score': round(float(c_data['technical_test_score'].mean()), 2),
                'avg_interview_score': round(float(c_data['interview_score'].mean()), 2),
                'avg_ats_score': round(float(c_data['ats_score'].mean()), 2),
                'avg_salary': round(float(c_data['expected_salary'].mean()), 2),
                'top_roles': c_data['job_role'].value_counts().head(3).to_dict() if 'job_role' in c_data.columns else {}
            }
        self.cluster_profiles_ = profiles


def evaluate_clustering_range(X, k_range=range(2, 9), sample_size=3000, random_state=42):
    """
    Evaluates K-Means clustering across a range of k values, computing
    inertia (elbow method) and silhouette scores.
    Also evaluates Hierarchical Clustering on a sample.
    """
    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        sample_indices = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X
        
    elbow_inertias = []
    kmeans_silhouettes = []
    hierarchical_silhouettes = []
    
    for k in k_range:
        # K-Means
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_sample)
        elbow_inertias.append(float(km.inertia_))
        km_sil = float(silhouette_score(X_sample, km.labels_))
        kmeans_silhouettes.append(km_sil)
        
        # Hierarchical
        hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        hc_labels = hc.fit_predict(X_sample)
        hc_sil = float(silhouette_score(X_sample, hc_labels))
        hierarchical_silhouettes.append(hc_sil)
        
    results = {
        'k_values': list(k_range),
        'inertias': elbow_inertias,
        'kmeans_silhouette': kmeans_silhouettes,
        'hierarchical_silhouette': hierarchical_silhouettes
    }
    return results
