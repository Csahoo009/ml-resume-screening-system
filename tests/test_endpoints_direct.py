import sys
sys.path.insert(0, '.')
from backend.main import (
    health, predict_candidate, batch_predict, 
    get_metrics, get_clusters, get_sample_leaderboard, load_artifacts
)
from backend.schemas import CandidateInput, BatchPredictionRequest

# Ensure artifacts are loaded
load_artifacts()

def run_direct_tests():
    print("Testing /health...")
    h = health()
    print("Health response:", h)
    assert h.status == "healthy"
    assert h.models_loaded is True

    print("\nTesting /predict...")
    cand = CandidateInput(
        candidate_id="CAND-TEST-001",
        full_name="Dr. Sarah Connor",
        age=32,
        gender="Female",
        highest_education="PhD",
        field_of_study="Computer Science",
        cgpa=3.92,
        graduation_year=2018,
        experience_years=8,
        current_job_title="AI Engineer",
        internship_experience="Yes",
        leadership_experience="Yes",
        technical_skills="Python, PyTorch, Kubernetes, SQL, Docker",
        projects_completed=12,
        publications=3,
        communication_score=90,
        problem_solving_score=95,
        technical_test_score=94,
        interview_score=92,
        aptitude_score=91,
        job_role="AI Engineer",
        expected_salary=185000,
        employment_type="Full-Time",
        remote_preference="Remote",
        availability="Immediate",
        resume_quality_score=95,
        resume_length=2,
        keyword_match_percentage=89.5,
        ats_score=92
    )
    pred = predict_candidate(cand)
    print("Single Prediction Result:")
    print(f"  Candidate: {pred.full_name} ({pred.candidate_id})")
    print(f"  Selected: {pred.selected} (Probability: {pred.selection_probability})")
    print(f"  Predicted Continuous Suitability: {pred.predicted_suitability_score}")
    print(f"  Talent Pool: {pred.talent_pool_name} (Cluster {pred.talent_pool_cluster})")
    print(f"  Composite Rank Score: {pred.composite_rank_score}")
    assert pred.is_selected is True
    assert pred.composite_rank_score > 80

    print("\nTesting /batch-predict...")
    batch_req = BatchPredictionRequest(candidates=[
        cand,
        CandidateInput(
            candidate_id="CAND-TEST-002",
            full_name="Junior Dev",
            age=22,
            gender="Male",
            highest_education="Bachelor",
            experience_years=1,
            technical_test_score=55,
            interview_score=50,
            ats_score=45,
            job_role="Software Engineer"
        )
    ])
    b_res = batch_predict(batch_req)
    print(f"Batch evaluated: {b_res.total_candidates} candidates")
    print(f"Selected: {b_res.selected_count}, Rejected: {b_res.rejected_count}")
    print(f"Rank #1: {b_res.leaderboard[0]['full_name']} (Score: {b_res.leaderboard[0]['composite_rank_score']})")
    print(f"Rank #2: {b_res.leaderboard[1]['full_name']} (Score: {b_res.leaderboard[1]['composite_rank_score']})")
    assert b_res.leaderboard[0]["candidate_rank"] == 1

    print("\nTesting /metrics...")
    m = get_metrics()
    assert "classification_metrics" in m
    print(f"Loaded metrics for {len(m['classification_metrics'])} classification models.")

    print("\nTesting /clusters...")
    c = get_clusters()
    assert len(c) == 4
    print("Clusters verified:", list(c.keys()))

    print("\nTesting /leaderboard...")
    lb = get_sample_leaderboard(limit=5)
    assert len(lb) > 0
    print(f"Retrieved {len(lb)} pre-evaluated candidates from leaderboard.")

    print("\n" + "=" * 60)
    print("ALL DIRECT ENDPOINT HANDLER TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    run_direct_tests()
