"""
Preprocessing & Feature Engineering Pipeline for Resume Screening & Ranking
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


PII_COLUMNS = [
    'candidate_id', 'full_name', 'email', 'phone', 
    'github_profile', 'linkedin_profile'
]

EDUCATION_ORDER = ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']

NUMERICAL_COLS = [
    'age', 'cgpa', 'graduation_year', 'experience_years',
    'projects_completed', 'publications', 'communication_score',
    'problem_solving_score', 'technical_test_score', 'interview_score',
    'aptitude_score', 'expected_salary', 'resume_quality_score',
    'resume_length', 'keyword_match_percentage', 'ats_score'
]

CATEGORICAL_NOMINAL_COLS = [
    'job_role', 'employment_type', 'gender', 'field_of_study',
    'remote_preference', 'availability', 'internship_experience', 'leadership_experience'
]


class ResumeFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end feature pipeline:
    - Drops PII columns
    - Handles missing values (median for numerical, constant for categorical)
    - Engineers domain features (skills_count, has_certifications, has_previous_company)
    - Ordinal encodes education
    - One-hot encodes nominal categoricals
    - Scales numerical variables using StandardScaler
    - Provides optional PCA dimensionality reduction
    """
    def __init__(self, n_pca_components=15, apply_pca=False):
        self.n_pca_components = n_pca_components
        self.apply_pca = apply_pca
        
        self.num_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        self.edu_encoder = OrdinalEncoder(
            categories=[EDUCATION_ORDER],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.nominal_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        self.nominal_encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )
        self.pca = PCA(n_components=self.n_pca_components, random_state=42)
        self.pca_2d = PCA(n_components=2, random_state=42)
        
        self.feature_names_ = []
        self.fitted_ = False

    def _extract_engineered_features(self, df):
        """Extract domain features from text/composite fields if present."""
        df_out = pd.DataFrame(index=df.index)
        
        # Skills count
        if 'technical_skills' in df.columns:
            df_out['skills_count'] = df['technical_skills'].fillna('').apply(
                lambda s: 0 if s in ['', 'Not Applicable'] else len(str(s).split(','))
            )
        else:
            df_out['skills_count'] = 0
            
        # Has certifications
        if 'certifications' in df.columns:
            df_out['has_certifications'] = df['certifications'].fillna('').apply(
                lambda s: 0 if s in ['', 'Not Applicable'] else len(str(s).split(','))
            )
        else:
            df_out['has_certifications'] = 0
            
        # Has previous companies
        if 'previous_companies' in df.columns:
            df_out['has_previous_company'] = df['previous_companies'].fillna('').apply(
                lambda s: 0 if s in ['', 'Not Applicable'] else 1
            )
        else:
            df_out['has_previous_company'] = 0
            
        return df_out

    def fit(self, X, y=None):
        X_df = X.copy()
        if isinstance(X_df, np.ndarray):
            raise ValueError("Input must be a pandas DataFrame with named columns")

        # Drop PII columns if present
        cols_to_drop = [c for c in PII_COLUMNS if c in X_df.columns]
        X_clean = X_df.drop(columns=cols_to_drop, errors='ignore')

        # 1. Numerical features
        present_num_cols = [c for c in NUMERICAL_COLS if c in X_clean.columns]
        X_num = X_clean[present_num_cols].copy()
        
        # Add engineered features
        eng_features = self._extract_engineered_features(X_clean)
        for col in eng_features.columns:
            X_num[col] = eng_features[col]
            
        self.active_num_cols = list(X_num.columns)
        self.num_imputer.fit(X_num)
        X_num_imp = self.num_imputer.transform(X_num)
        self.scaler.fit(X_num_imp)
        self.minmax_scaler.fit(X_num_imp)

        # 2. Education Ordinal Feature
        if 'highest_education' in X_clean.columns:
            edu_raw = X_clean[['highest_education']].fillna('Bachelor')
            # Replace unseen categories with Bachelor
            edu_clean = edu_raw['highest_education'].apply(
                lambda e: e if e in EDUCATION_ORDER else 'Bachelor'
            ).to_numpy(dtype=object).reshape(-1, 1)
            self.edu_encoder.fit(edu_clean)

        # 3. Nominal Categorical Features
        present_cat_cols = [c for c in CATEGORICAL_NOMINAL_COLS if c in X_clean.columns]
        self.active_cat_cols = present_cat_cols
        X_cat = X_clean[present_cat_cols].astype(str).to_numpy(dtype=object)
        self.nominal_imputer.fit(X_cat)
        X_cat_imp = self.nominal_imputer.transform(X_cat)
        self.nominal_encoder.fit(X_cat_imp)

        # Build feature names
        num_feature_names = self.active_num_cols
        edu_feature_names = ['highest_education_encoded']
        ohe_feature_names = list(self.nominal_encoder.get_feature_names_out(self.active_cat_cols))
        self.feature_names_ = num_feature_names + edu_feature_names + ohe_feature_names

        # Transform to fit PCA
        X_all = self.transform(X, apply_pca_override=False)
        self.pca.fit(X_all)
        self.pca_2d.fit(X_all)
        
        self.fitted_ = True
        return self

    def transform(self, X, apply_pca_override=None):
        X_df = X.copy()
        cols_to_drop = [c for c in PII_COLUMNS if c in X_df.columns]
        X_clean = X_df.drop(columns=cols_to_drop, errors='ignore')

        # 1. Numerical
        present_num_cols = [c for c in NUMERICAL_COLS if c in X_clean.columns]
        X_num = X_clean[present_num_cols].copy()
        
        eng_features = self._extract_engineered_features(X_clean)
        for col in eng_features.columns:
            X_num[col] = eng_features[col]
            
        # Ensure all fitted columns are present
        for col in self.active_num_cols:
            if col not in X_num.columns:
                X_num[col] = np.nan
        X_num = X_num[self.active_num_cols]
        
        X_num_imp = self.num_imputer.transform(X_num)
        X_num_scaled = self.scaler.transform(X_num_imp)

        # 2. Education
        if 'highest_education' in X_clean.columns:
            edu_raw = X_clean[['highest_education']].fillna('Bachelor')
            edu_clean = edu_raw['highest_education'].apply(
                lambda e: e if e in EDUCATION_ORDER else 'Bachelor'
            ).to_numpy(dtype=object).reshape(-1, 1)
            X_edu_enc = self.edu_encoder.transform(edu_clean)
            # Impute unknown (-1) with median (2.0 = Bachelor)
            X_edu_enc[X_edu_enc == -1] = 2.0
            # Scale education
            X_edu_enc = (X_edu_enc - 2.0) / 1.5
        else:
            X_edu_enc = np.zeros((len(X_clean), 1))

        # 3. Categorical Nominal
        for col in self.active_cat_cols:
            if col not in X_clean.columns:
                X_clean[col] = 'Unknown'
        X_cat = X_clean[self.active_cat_cols].astype(str).to_numpy(dtype=object)
        X_cat_imp = self.nominal_imputer.transform(X_cat)
        X_cat_ohe = self.nominal_encoder.transform(X_cat_imp)

        # Combine all features
        X_combined = np.hstack([X_num_scaled, X_edu_enc, X_cat_ohe])

        use_pca = self.apply_pca if apply_pca_override is None else apply_pca_override
        if use_pca:
            return self.pca.transform(X_combined)
        return X_combined

    def transform_non_negative(self, X):
        """Transform features to non-negative range [0, 1] for MultinomialNB."""
        X_dense = self.transform(X, apply_pca_override=False)
        # Shift and scale to [0, 1]
        min_vals = np.min(X_dense, axis=0)
        shifted = X_dense - min_vals
        max_vals = np.max(shifted, axis=0)
        max_vals[max_vals == 0] = 1.0
        return shifted / max_vals

    def transform_2d_pca(self, X):
        """Transform features into 2D PCA for visual representations."""
        X_combined = self.transform(X, apply_pca_override=False)
        return self.pca_2d.transform(X_combined)
