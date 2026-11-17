import sys
sys.path.insert(0, '.')
import pandas as pd
from frontend.app import engine

print("Testing batch scoring simulation...")
# Load 5 raw rows from ai_resume_screening_dataset.csv
df_raw = pd.read_csv('ai_resume_screening_dataset.csv', nrows=5)
print("Raw columns:", list(df_raw.columns)[:10])
assert 'composite_rank_score' not in df_raw.columns

# Run through the pipeline as Tab 2 does
X_b = engine['pipeline'].transform(df_raw)
probs = engine['classifier'].predict_proba(X_b)[:, 1]
suits = engine['regressor'].predict(X_b)
c_ids = engine['clusterer'].predict(X_b)

tech = df_raw['technical_test_score']
inter = df_raw['interview_score']
ats = df_raw['ats_score']

comp = (
    0.35 * (probs * 100.0) +
    0.25 * tech +
    0.25 * inter +
    0.15 * ats
)

df_raw['selection_probability'] = probs
df_raw['composite_rank_score'] = comp
df_raw['candidate_rank'] = df_raw['composite_rank_score'].rank(ascending=False).astype(int)

print("Batch scored successfully!")
print("Max composite score:", df_raw['composite_rank_score'].max())
print("Top candidate:", df_raw[['candidate_id', 'full_name', 'composite_rank_score', 'candidate_rank']])
print("\nTEST PASSED WITHOUT ANY KEYERROR!")
