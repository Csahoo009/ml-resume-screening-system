"""
Regression & Candidate Ranking Pipeline
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_ground_truth_suitability(df):
    """
    Computes a continuous ground truth suitability score (0-100) for candidates
    based on weighted domain evaluation criteria:
    - Technical Test Score: 30%
    - Interview Score: 30%
    - ATS Resume Match Score: 20%
    - Problem Solving Score: 10%
    - Communication Score: 10%
    """
    tech = df['technical_test_score'].fillna(df['technical_test_score'].median())
    interview = df['interview_score'].fillna(df['interview_score'].median())
    ats = df['ats_score'].fillna(df['ats_score'].median())
    ps = df['problem_solving_score'].fillna(df['problem_solving_score'].median())
    comm = df['communication_score'].fillna(df['communication_score'].median())
    
    suitability = (
        0.30 * tech +
        0.30 * interview +
        0.20 * ats +
        0.10 * ps +
        0.10 * comm
    )
    return np.round(suitability, 2)


def get_regression_models():
    """
    Returns candidate ranking regression models.
    """
    return {
        'Multiple Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        )
    }


def compute_composite_leaderboard(
    df, 
    selection_probs, 
    suitability_scores=None,
    weights=None
):
    """
    Combines model prediction probabilities and weighted domain attributes
    (ats_score, technical_test_score, interview_score) to produce a definitive
    ranked candidate leaderboard.
    
    Composite Score Formula:
    Composite = (w_prob * P(selected) * 100) + 
                (w_tech * technical_test_score) + 
                (w_inter * interview_score) + 
                (w_ats * ats_score)
    """
    if weights is None:
        weights = {
            'prob': 0.35,
            'tech': 0.25,
            'interview': 0.25,
            'ats': 0.15
        }
        
    df_board = df.copy()
    df_board['selection_probability'] = np.round(selection_probs, 4)
    
    if suitability_scores is not None:
        df_board['predicted_suitability'] = np.round(suitability_scores, 2)
        
    prob_scaled = df_board['selection_probability'] * 100.0
    tech = df_board['technical_test_score'].fillna(50.0)
    interview = df_board['interview_score'].fillna(50.0)
    ats = df_board['ats_score'].fillna(50.0)
    
    composite = (
        weights['prob'] * prob_scaled +
        weights['tech'] * tech +
        weights['interview'] * interview +
        weights['ats'] * ats
    )
    
    df_board['composite_rank_score'] = np.round(composite, 2)
    df_board = df_board.sort_values(by='composite_rank_score', ascending=False).reset_index(drop=True)
    df_board['candidate_rank'] = df_board.index + 1
    
    return df_board
