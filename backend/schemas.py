"""
Pydantic Schemas for FastAPI Resume Screening Backend
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CandidateInput(BaseModel):
    candidate_id: Optional[str] = "CAND-001"
    full_name: Optional[str] = "Jane Doe"
    email: Optional[str] = "jane.doe@example.com"
    phone: Optional[str] = "+1-555-0192"
    age: int = Field(28, ge=18, le=75, description="Candidate age")
    gender: str = Field("Female", description="Gender (Male, Female, Other)")
    highest_education: str = Field("Master", description="Highest education level (High School, Diploma, Bachelor, Master, PhD)")
    university: Optional[str] = "Stanford University"
    field_of_study: str = Field("Computer Science", description="Degree major")
    cgpa: float = Field(3.8, ge=0.0, le=4.0, description="Cumulative GPA")
    graduation_year: int = Field(2020, ge=1970, le=2030, description="Year of graduation")
    experience_years: int = Field(5, ge=0, le=45, description="Years of professional experience")
    current_job_title: Optional[str] = "Software Engineer"
    internship_experience: str = Field("Yes", description="Has internship experience (Yes/No)")
    leadership_experience: str = Field("Yes", description="Has leadership experience (Yes/No)")
    technical_skills: Optional[str] = "Python, PyTorch, Docker, React, SQL, Git"
    programming_languages: Optional[str] = "Python, JavaScript"
    frameworks: Optional[str] = "FastAPI, React, PyTorch"
    databases: Optional[str] = "PostgreSQL, Redis"
    cloud_platform: Optional[str] = "AWS, GCP"
    projects_completed: int = Field(8, ge=0, le=50, description="Completed projects count")
    certifications: Optional[str] = "AWS Certified Solutions Architect"
    publications: int = Field(1, ge=0, le=20, description="Publications count")
    communication_score: int = Field(85, ge=0, le=100, description="Communication test score")
    problem_solving_score: int = Field(88, ge=0, le=100, description="Problem solving test score")
    technical_test_score: int = Field(90, ge=0, le=100, description="Technical assessment score")
    interview_score: int = Field(86, ge=0, le=100, description="Behavioral/technical interview score")
    aptitude_score: int = Field(84, ge=0, le=100, description="General aptitude score")
    job_role: str = Field("Software Engineer", description="Target job role")
    expected_salary: int = Field(145000, ge=30000, le=500000, description="Annual expected salary ($)")
    employment_type: str = Field("Full-Time", description="Employment type (Full-Time, Part-Time, Contract, Internship)")
    remote_preference: str = Field("Hybrid", description="Remote preference (Remote, Hybrid, On-site)")
    availability: str = Field("30 Days", description="Availability (Immediate, 15 Days, 30 Days, 60 Days, 90 Days)")
    resume_quality_score: int = Field(88, ge=0, le=100, description="Resume formatting/structure quality")
    resume_length: int = Field(2, ge=1, le=5, description="Resume length in pages")
    keyword_match_percentage: float = Field(78.5, ge=0.0, le=100.0, description="ATS keyword match percentage")
    ats_score: int = Field(82, ge=0, le=100, description="Overall ATS screening score")


class CandidatePredictionResponse(BaseModel):
    candidate_id: str
    full_name: str
    selected: str
    is_selected: bool
    selection_probability: float
    predicted_suitability_score: float
    talent_pool_cluster: int
    talent_pool_name: str
    composite_rank_score: float
    domain_attributes: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    candidates: List[CandidateInput]


class BatchPredictionResponse(BaseModel):
    total_candidates: int
    selected_count: int
    rejected_count: int
    leaderboard: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    best_classifier: Optional[str]
    best_regressor: Optional[str]
    features_count: int
