"""
Unit & Integration Tests for Resume Screening API
"""
import sys
sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    print("[PASS] Health endpoint test passed.")

def test_single_predict():
    payload = {
        "candidate_id": "CAND-TEST-001",
        "full_name": "Dr. Sarah Connor",
        "age": 32,
        "gender": "Female",
        "highest_education": "PhD",
        "field_of_study": "Computer Science",
        "cgpa": 3.92,
        "graduation_year": 2018,
        "experience_years": 8,
        "current_job_title": "AI Engineer",
        "internship_experience": "Yes",
        "leadership_experience": "Yes",
        "technical_skills": "Python, PyTorch, Kubernetes, SQL, Docker",
        "projects_completed": 12,
        "publications": 3,
        "communication_score": 90,
        "problem_solving_score": 95,
        "technical_test_score": 94,
        "interview_score": 92,
        "aptitude_score": 91,
        "job_role": "AI Engineer",
        "expected_salary": 185000,
        "employment_type": "Full-Time",
        "remote_preference": "Remote",
        "availability": "Immediate",
        "resume_quality_score": 95,
        "resume_length": 2,
        "keyword_match_percentage": 89.5,
        "ats_score": 92
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "selected" in data
    assert "selection_probability" in data
    assert "predicted_suitability_score" in data
    assert "talent_pool_cluster" in data
    assert "composite_rank_score" in data
    print(f"[PASS] Predict endpoint test passed. Selected: {data['selected']}, Suitability: {data['predicted_suitability_score']}, Rank Score: {data['composite_rank_score']}")

def test_batch_predict():
    payload = {
        "candidates": [
            {
                "candidate_id": "CAND-001",
                "full_name": "Applicant One",
                "age": 24,
                "gender": "Male",
                "highest_education": "Bachelor",
                "field_of_study": "Software Engineering",
                "cgpa": 3.2,
                "graduation_year": 2022,
                "experience_years": 2,
                "current_job_title": "Software Engineer",
                "internship_experience": "Yes",
                "leadership_experience": "No",
                "technical_skills": "Java, Spring Boot, MySQL",
                "projects_completed": 4,
                "publications": 0,
                "communication_score": 70,
                "problem_solving_score": 75,
                "technical_test_score": 72,
                "interview_score": 68,
                "aptitude_score": 74,
                "job_role": "Software Engineer",
                "expected_salary": 95000,
                "employment_type": "Full-Time",
                "remote_preference": "Hybrid",
                "availability": "30 Days",
                "resume_quality_score": 75,
                "resume_length": 1,
                "keyword_match_percentage": 60.0,
                "ats_score": 68
            },
            {
                "candidate_id": "CAND-002",
                "full_name": "Applicant Two",
                "age": 42,
                "gender": "Female",
                "highest_education": "Master",
                "field_of_study": "Computer Science",
                "cgpa": 3.9,
                "graduation_year": 2006,
                "experience_years": 18,
                "current_job_title": "Cloud Architect",
                "internship_experience": "Yes",
                "leadership_experience": "Yes",
                "technical_skills": "AWS, GCP, Terraform, Kubernetes, Go",
                "projects_completed": 15,
                "publications": 2,
                "communication_score": 92,
                "problem_solving_score": 95,
                "technical_test_score": 96,
                "interview_score": 94,
                "aptitude_score": 90,
                "job_role": "Cloud Engineer",
                "expected_salary": 220000,
                "employment_type": "Full-Time",
                "remote_preference": "Remote",
                "availability": "Immediate",
                "resume_quality_score": 96,
                "resume_length": 3,
                "keyword_match_percentage": 92.0,
                "ats_score": 94
            }
        ]
    }
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_candidates"] == 2
    assert len(data["leaderboard"]) == 2
    assert data["leaderboard"][0]["candidate_rank"] == 1
    assert data["leaderboard"][1]["candidate_rank"] == 2
    print(f"[PASS] Batch predict test passed. Top candidate: {data['leaderboard'][0]['full_name']} (Rank score: {data['leaderboard'][0]['composite_rank_score']})")

def test_analytics_endpoints():
    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "classification_metrics" in r_metrics.json()

    r_clusters = client.get("/clusters")
    assert r_clusters.status_code == 200
    assert len(r_clusters.json()) == 4

    r_leaderboard = client.get("/leaderboard?limit=5")
    assert r_leaderboard.status_code == 200
    assert len(r_leaderboard.json()) <= 5
    print("[PASS] Analytics endpoints (/metrics, /clusters, /leaderboard) passed.")

if __name__ == "__main__":
    test_health()
    test_single_predict()
    test_batch_predict()
    test_analytics_endpoints()
    print("\nALL API TESTS PASSED SUCCESSFULLY!")
