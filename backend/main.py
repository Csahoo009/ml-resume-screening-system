"""
FastAPI Backend Application for Resume Screening & Ranking System
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from backend.schemas import (
    CandidateInput, CandidatePredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthCheckResponse
)
from src.ranking import compute_composite_leaderboard


app = FastAPI(
    title="Resume Screening & Candidate Ranking API",
    description="End-to-end ML API for candidate screening, suitability regression, talent pool clustering, and composite leaderboard ranking.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"

# Global model state
feature_pipeline = None
best_classifier = None
best_regressor = None
talent_clusterer = None
pipeline_metadata = {}


def load_artifacts():
    global feature_pipeline, best_classifier, best_regressor, talent_clusterer, pipeline_metadata
    try:
        if os.path.exists(os.path.join(MODELS_DIR, 'feature_pipeline.joblib')):
            feature_pipeline = joblib.load(os.path.join(MODELS_DIR, 'feature_pipeline.joblib'))
            best_classifier = joblib.load(os.path.join(MODELS_DIR, 'best_classifier.joblib'))
            best_regressor = joblib.load(os.path.join(MODELS_DIR, 'best_regressor.joblib'))
            talent_clusterer = joblib.load(os.path.join(MODELS_DIR, 'talent_clusterer.joblib'))
            
            meta_path = os.path.join(MODELS_DIR, 'pipeline_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    pipeline_metadata = json.load(f)
            print("ML Models & Preprocessors loaded successfully!")
            return True
    except Exception as e:
        print(f"Error loading models: {e}")
    return False


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Resume Screening & Candidate Ranking API",
        "status": "operational",
        "endpoints": [
            "/health",
            "/predict",
            "/batch-predict",
            "/metrics",
            "/clusters",
            "/leaderboard"
        ]
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health():
    models_ready = (
        feature_pipeline is not None and 
        best_classifier is not None and 
        best_regressor is not None
    )
    return HealthCheckResponse(
        status="healthy" if models_ready else "degraded (models loading or not trained yet)",
        version="1.0.0",
        models_loaded=models_ready,
        best_classifier=pipeline_metadata.get('best_classifier_name', 'Random Forest'),
        best_regressor=pipeline_metadata.get('best_regressor_name', 'Ridge Regression'),
        features_count=pipeline_metadata.get('feature_count', 63)
    )


@app.post("/predict", response_model=CandidatePredictionResponse, tags=["Inference"])
def predict_candidate(candidate: CandidateInput):
    global feature_pipeline, best_classifier, best_regressor, talent_clusterer
    if feature_pipeline is None:
        if not load_artifacts():
            raise HTTPException(status_code=503, detail="Models not loaded. Train pipeline first.")

    try:
        candidate_dict = candidate.dict()
        df_input = pd.DataFrame([candidate_dict])
        
        # Preprocessing & Transformation
        X_trans = feature_pipeline.transform(df_input)
        
        # 1. Supervised Classification (Selected Yes/No)
        if hasattr(best_classifier, "predict_proba"):
            prob_selected = float(best_classifier.predict_proba(X_trans)[0, 1])
        else:
            prob_selected = float(best_classifier.predict(X_trans)[0])
            
        is_selected = bool(prob_selected >= 0.5)
        selected_label = "Selected" if is_selected else "Rejected"
        
        # 2. Continuous Suitability Regression Score
        suitability_score = float(best_regressor.predict(X_trans)[0])
        suitability_score = round(max(0.0, min(100.0, suitability_score)), 2)
        
        # 3. Talent Pool Clustering
        cluster_id = int(talent_clusterer.predict(X_trans)[0])
        cluster_name = talent_clusterer.cluster_names_.get(
            cluster_id, f"Talent Pool {cluster_id}"
        )
        
        # 4. Composite Leaderboard Rank Score
        # (35% selection prob, 25% tech score, 25% interview score, 15% ats score)
        tech = float(candidate.technical_test_score)
        interview = float(candidate.interview_score)
        ats = float(candidate.ats_score)
        
        composite_score = round(
            (0.35 * prob_selected * 100.0) +
            (0.25 * tech) +
            (0.25 * interview) +
            (0.15 * ats), 2
        )
        
        return CandidatePredictionResponse(
            candidate_id=candidate.candidate_id or "CAND-001",
            full_name=candidate.full_name or "Anonymous Candidate",
            selected=selected_label,
            is_selected=is_selected,
            selection_probability=round(prob_selected, 4),
            predicted_suitability_score=suitability_score,
            talent_pool_cluster=cluster_id,
            talent_pool_name=cluster_name,
            composite_rank_score=composite_score,
            domain_attributes={
                "technical_test_score": tech,
                "interview_score": interview,
                "ats_score": ats,
                "experience_years": candidate.experience_years,
                "cgpa": candidate.cgpa,
                "job_role": candidate.job_role
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_predict(batch: BatchPredictionRequest):
    global feature_pipeline, best_classifier, best_regressor, talent_clusterer
    if feature_pipeline is None:
        if not load_artifacts():
            raise HTTPException(status_code=503, detail="Models not loaded. Train pipeline first.")

    try:
        records = [c.dict() for c in batch.candidates]
        df_batch = pd.DataFrame(records)
        
        X_trans = feature_pipeline.transform(df_batch)
        
        # Classification
        if hasattr(best_classifier, "predict_proba"):
            probs = best_classifier.predict_proba(X_trans)[:, 1]
        else:
            probs = best_classifier.predict(X_trans).astype(float)
            
        suitabilities = best_regressor.predict(X_trans)
        clusters = talent_clusterer.predict(X_trans)
        
        leaderboard_df = compute_composite_leaderboard(
            df_batch, probs, suitability_scores=suitabilities
        )
        leaderboard_df['talent_pool_cluster'] = clusters
        leaderboard_df['talent_pool_name'] = [
            talent_clusterer.cluster_names_.get(c, f"Pool {c}") for c in clusters
        ]
        leaderboard_df['selected'] = [
            "Selected" if p >= 0.5 else "Rejected" for p in probs
        ]
        
        records_out = leaderboard_df.to_dict(orient="records")
        selected_count = sum(1 for p in probs if p >= 0.5)
        
        return BatchPredictionResponse(
            total_candidates=len(records),
            selected_count=selected_count,
            rejected_count=len(records) - selected_count,
            leaderboard=records_out
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/metrics", tags=["Analytics"])
def get_metrics():
    if not pipeline_metadata:
        load_artifacts()
    return pipeline_metadata


@app.get("/clusters", tags=["Analytics"])
def get_clusters():
    if not pipeline_metadata:
        load_artifacts()
    return pipeline_metadata.get('cluster_profiles', {})


@app.get("/leaderboard", tags=["Analytics"])
def get_sample_leaderboard(limit: int = 50):
    sample_csv = os.path.join(ARTIFACTS_DIR, 'leaderboard_sample.csv')
    if os.path.exists(sample_csv):
        df_sample = pd.read_csv(sample_csv)
        return df_sample.head(limit).to_dict(orient='records')
    return []
