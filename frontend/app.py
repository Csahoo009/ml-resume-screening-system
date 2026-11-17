"""
Interactive Recruiter Dashboard - Streamlit Frontend
AI Resume Screening & Candidate Ranking System
"""
import os
import sys
import json
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure Project root directory is in sys.path regardless of launch cwd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Configure page
st.set_page_config(
    page_title="AI Resume Screening & Ranking System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants & Paths (Using absolute paths to avoid cwd mismatches)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Load local models & metadata as fallback
@st.cache_resource
def load_local_engine():
    try:
        pipeline_path = os.path.join(MODELS_DIR, 'feature_pipeline.joblib')
        if not os.path.exists(pipeline_path):
            return {
                'ready': False,
                'error': f"Models not found at {MODELS_DIR}. Please run 'python run_pipeline.py' first."
            }
            
        pipeline = joblib.load(pipeline_path)
        classifier = joblib.load(os.path.join(MODELS_DIR, 'best_classifier.joblib'))
        regressor = joblib.load(os.path.join(MODELS_DIR, 'best_regressor.joblib'))
        clusterer = joblib.load(os.path.join(MODELS_DIR, 'talent_clusterer.joblib'))
        all_classifiers = joblib.load(os.path.join(MODELS_DIR, 'all_classifiers.joblib'))
        
        meta_path = os.path.join(MODELS_DIR, 'pipeline_metadata.json')
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        return {
            'ready': True,
            'pipeline': pipeline,
            'classifier': classifier,
            'regressor': regressor,
            'clusterer': clusterer,
            'all_classifiers': all_classifiers,
            'metadata': meta
        }
    except Exception as e:
        return {'ready': False, 'error': str(e)}

engine = load_local_engine()

# Helper: check backend health
def check_api_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None

api_online, api_info = check_api_health()

# ==========================================
# SIDEBAR NAVIGATION & SETTINGS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=70)
    st.title("Recruitment Portal")
    st.caption("AI-Powered Resume Screening & Ranking")
    
    st.markdown("---")
    st.subheader("System Status")
    if api_online:
        st.success("🟢 **FastAPI Backend Online**")
        st.caption(f"Best Model: `{api_info.get('best_classifier', 'N/A')}`")
    elif engine.get('ready'):
        st.info("🟡 **Local Direct Engine Active**")
        best_name = engine.get('metadata', {}).get('best_classifier_name', 'Neural Network (MLP)')
        st.caption(f"Loaded: `{best_name}`")
    else:
        st.error("🔴 **Local Engine Not Loaded**")
        st.caption(f"Error: {engine.get('error')}")
        
    st.markdown("---")
    st.subheader("Leaderboard Weights")
    w_prob = st.slider("Selection Probability Weight", 0.1, 0.6, 0.35, 0.05)
    w_tech = st.slider("Technical Test Weight", 0.1, 0.5, 0.25, 0.05)
    w_inter = st.slider("Interview Score Weight", 0.1, 0.5, 0.25, 0.05)
    w_ats = st.slider("ATS Score Weight", 0.05, 0.4, 0.15, 0.05)
    
    total_w = w_prob + w_tech + w_inter + w_ats
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"Weights sum: {total_w:.2f} (normalized dynamically)")
        
    st.markdown("---")
    st.caption("Project: 5th Sem ML/DA Autonomous Pipeline")

# ==========================================
# HEADER
# ==========================================
st.title("🎯 AI Resume Screening & Candidate Ranking System")
st.markdown(
    """
    An enterprise end-to-end recruitment intelligence system featuring **supervised classification** (Yes/No screening),
    **continuous suitability regression**, **unsupervised talent pool clustering**, and **multi-attribute leaderboard ranking**.
    """
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Single Candidate Screener",
    "📊 Batch Screening & Leaderboard",
    "🧬 Talent Pool Clusters",
    "📈 Model Performance & Analytics"
])


# ==========================================
# TAB 1: SINGLE CANDIDATE SCREENER
# ==========================================
with tab1:
    st.header("Candidate Profile Evaluation")
    st.markdown("Enter applicant attributes to evaluate their screening recommendation, continuous suitability score, and talent cluster.")
    
    with st.form("candidate_eval_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal & Academic")
            c_name = st.text_input("Full Name", "Alex Mercer")
            c_id = st.text_input("Candidate ID", "CAND-09281")
            c_age = st.number_input("Age", min_value=18, max_value=65, value=29)
            c_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            c_edu = st.selectbox("Highest Education", ["Bachelor", "Master", "PhD", "Diploma", "High School"], index=1)
            c_study = st.selectbox("Field of Study", [
                "Computer Science", "Software Engineering", "Data Science", 
                "Electrical Engineering", "Information Systems", "Mathematics & Statistics", "Other"
            ])
            c_cgpa = st.slider("CGPA (Scale 0-4.0)", 2.0, 4.0, 3.65, 0.01)
            c_grad = st.number_input("Graduation Year", 1990, 2026, 2019)
            
        with col2:
            st.subheader("Experience & Technical")
            c_role = st.selectbox("Target Job Role", [
                "Software Engineer", "Machine Learning Engineer", "AI Engineer", 
                "Full Stack Developer", "Cloud Engineer", "Mobile App Developer",
                "Data Scientist", "Cyber Security Analyst", "DevOps Engineer",
                "Backend Developer", "Frontend Developer", "UI UX Designer", "QA Engineer", "Business Analyst"
            ])
            c_exp = st.slider("Experience (Years)", 0, 35, 6)
            c_salary = st.number_input("Expected Salary ($/yr)", 30000, 400000, 150000, 5000)
            c_projects = st.slider("Projects Completed", 0, 20, 8)
            c_publications = st.slider("Publications", 0, 10, 1)
            c_intern = st.radio("Internship Experience", ["Yes", "No"], horizontal=True)
            c_lead = st.radio("Leadership Experience", ["Yes", "No"], horizontal=True)
            c_skills = st.text_input("Technical Skills", "Python, Docker, FastAPI, Scikit-learn, React, SQL")
            
        with col3:
            st.subheader("Assessment Scores")
            c_tech = st.slider("Technical Test Score (0-100)", 0, 100, 84)
            c_interview = st.slider("Interview Score (0-100)", 0, 100, 82)
            c_ats = st.slider("ATS Resume Score (0-100)", 0, 100, 78)
            c_ps = st.slider("Problem Solving Score (0-100)", 0, 100, 80)
            c_comm = st.slider("Communication Score (0-100)", 0, 100, 85)
            c_apt = st.slider("Aptitude Score (0-100)", 0, 100, 79)
            c_resume_q = st.slider("Resume Quality Score", 0, 100, 88)
            c_keyword = st.slider("Keyword Match %", 0.0, 100.0, 68.5, 0.5)
            
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            c_emp_type = st.selectbox("Employment Type", ["Full-Time", "Contract", "Part-Time", "Internship"])
        with col_sub2:
            c_avail = st.selectbox("Availability", ["Immediate", "15 Days", "30 Days", "60 Days", "90 Days"], index=2)
            
        submitted = st.form_submit_button("🔍 Screen & Rank Candidate", use_container_width=True)
        
    if submitted:
        candidate_data = {
            "candidate_id": c_id,
            "full_name": c_name,
            "age": int(c_age),
            "gender": c_gender,
            "highest_education": c_edu,
            "field_of_study": c_study,
            "cgpa": float(c_cgpa),
            "graduation_year": int(c_grad),
            "experience_years": int(c_exp),
            "current_job_title": c_role,
            "internship_experience": c_intern,
            "leadership_experience": c_lead,
            "technical_skills": c_skills,
            "projects_completed": int(c_projects),
            "publications": int(c_publications),
            "communication_score": int(c_comm),
            "problem_solving_score": int(c_ps),
            "technical_test_score": int(c_tech),
            "interview_score": int(c_interview),
            "aptitude_score": int(c_apt),
            "job_role": c_role,
            "expected_salary": int(c_salary),
            "employment_type": c_emp_type,
            "remote_preference": "Hybrid",
            "availability": c_avail,
            "resume_quality_score": int(c_resume_q),
            "resume_length": 2,
            "keyword_match_percentage": float(c_keyword),
            "ats_score": int(c_ats)
        }
        
        # Inference (API or Local Engine)
        res_data = None
        if api_online:
            try:
                resp = requests.post(f"{BACKEND_URL}/predict", json=candidate_data, timeout=3.0)
                if resp.status_code == 200:
                    res_data = resp.json()
            except Exception:
                pass
                
        if res_data is None and engine.get('ready'):
            df_in = pd.DataFrame([candidate_data])
            X_t = engine['pipeline'].transform(df_in)
            if hasattr(engine['classifier'], "predict_proba"):
                p = float(engine['classifier'].predict_proba(X_t)[0, 1])
            else:
                p = float(engine['classifier'].predict(X_t)[0])
            suit = float(engine['regressor'].predict(X_t)[0])
            cid = int(engine['clusterer'].predict(X_t)[0])
            cname = engine['clusterer'].cluster_names_.get(cid, f"Talent Pool {cid}")
            
            norm_w = (w_prob + w_tech + w_inter + w_ats) or 1.0
            comp = (
                (w_prob / norm_w) * (p * 100.0) +
                (w_tech / norm_w) * c_tech +
                (w_inter / norm_w) * c_interview +
                (w_ats / norm_w) * c_ats
            )
            res_data = {
                "selected": "Selected" if p >= 0.5 else "Rejected",
                "is_selected": p >= 0.5,
                "selection_probability": round(p, 4),
                "predicted_suitability_score": round(max(0.0, min(100.0, suit)), 2),
                "talent_pool_cluster": cid,
                "talent_pool_name": cname,
                "composite_rank_score": round(comp, 2)
            }
            
        if res_data:
            st.success("Evaluation Complete!")
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)
            
            is_sel = res_data['is_selected']
            badge_color = "🟢" if is_sel else "🔴"
            
            with rcol1:
                st.metric("Screening Verdict", f"{badge_color} {res_data['selected']}")
                st.caption(f"Confidence: {res_data['selection_probability']*100:.1f}%")
                
            with rcol2:
                st.metric("Continuous Suitability", f"{res_data['predicted_suitability_score']}/100")
                st.caption("Predicted by Regressor Model")
                
            with rcol3:
                st.metric("Composite Rank Score", f"{res_data['composite_rank_score']}/100")
                st.caption("Weighted Multi-Attribute Score")
                
            with rcol4:
                st.metric("Talent Pool Group", f"Group {res_data['talent_pool_cluster']}")
                st.caption(f"*{res_data['talent_pool_name']}*")
                
            st.markdown("---")
            st.subheader("Assessment Radar & Profile Metrics")
            scores = {
                'Technical Test': c_tech,
                'Interview Score': c_interview,
                'ATS Score': c_ats,
                'Problem Solving': c_ps,
                'Communication': c_comm,
                'Aptitude': c_apt
            }
            df_scores = pd.DataFrame(list(scores.items()), columns=['Evaluation Metric', 'Candidate Score'])
            
            sc1, sc2 = st.columns([2, 1])
            with sc1:
                st.bar_chart(df_scores.set_index('Evaluation Metric'))
            with sc2:
                st.markdown("##### Candidate Summary Card")
                st.markdown(f"**Name:** {c_name}")
                st.markdown(f"**Target Role:** {c_role}")
                st.markdown(f"**Education:** {c_edu} in {c_study} (GPA {c_cgpa})")
                st.markdown(f"**Experience:** {c_exp} Years")
                st.markdown(f"**Salary Target:** ${c_salary:,.0f}")
                st.markdown(f"**Talent Pool Persona:** `{res_data['talent_pool_name']}`")
        else:
            st.error(f"Engine unavailable: {engine.get('error', 'Models not found')}. Please run training pipeline first.")


# ==========================================
# TAB 2: BATCH SCREENING & LEADERBOARD
# ==========================================
with tab2:
    st.header("Batch Resume Screening & Ranked Leaderboard")
    st.markdown("Upload a candidate batch CSV or evaluate the validation dataset to generate an automated candidate leaderboard.")
    
    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        uploaded_file = st.file_uploader("Upload Candidates CSV (44 attributes)", type=['csv'])
    with b_col2:
        st.write("")
        st.write("")
        load_sample_btn = st.button("📂 Load Pre-evaluated Sample Batch", use_container_width=True)
        
    df_batch = None
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.info(f"Loaded **{len(df_batch):,}** candidates from uploaded file.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif load_sample_btn or os.path.exists(os.path.join(ARTIFACTS_DIR, 'leaderboard_sample.csv')):
        sample_path = os.path.join(ARTIFACTS_DIR, 'leaderboard_sample.csv')
        if os.path.exists(sample_path):
            df_batch = pd.read_csv(sample_path)
            
    if df_batch is not None:
        # Check if already processed
        if 'composite_rank_score' not in df_batch.columns:
            if not engine.get('ready'):
                st.error(f"⚠️ Cannot score uploaded candidates because the ML engine is not loaded: {engine.get('error')}. Please run 'python run_pipeline.py' to generate model artifacts.")
            else:
                n_rows = len(df_batch)
                if n_rows > 2500:
                    st.warning(f"Large dataset detected ({n_rows:,} rows). Scoring the first 2,000 candidates for interactive responsiveness.")
                    df_to_score = df_batch.iloc[:2000].copy()
                else:
                    df_to_score = df_batch.copy()
                    
                with st.spinner(f"Scoring {len(df_to_score):,} candidates through feature pipeline & ML models..."):
                    try:
                        X_b = engine['pipeline'].transform(df_to_score)
                        if hasattr(engine['classifier'], "predict_proba"):
                            probs = engine['classifier'].predict_proba(X_b)[:, 1]
                        else:
                            probs = engine['classifier'].predict(X_b).astype(float)
                        suits = engine['regressor'].predict(X_b)
                        c_ids = engine['clusterer'].predict(X_b)
                        
                        norm_w = (w_prob + w_tech + w_inter + w_ats) or 1.0
                        tech = df_to_score['technical_test_score'].fillna(50) if 'technical_test_score' in df_to_score.columns else pd.Series(50, index=df_to_score.index)
                        inter = df_to_score['interview_score'].fillna(50) if 'interview_score' in df_to_score.columns else pd.Series(50, index=df_to_score.index)
                        ats = df_to_score['ats_score'].fillna(50) if 'ats_score' in df_to_score.columns else pd.Series(50, index=df_to_score.index)
                        
                        comp = (
                            (w_prob / norm_w) * (probs * 100.0) +
                            (w_tech / norm_w) * tech +
                            (w_inter / norm_w) * inter +
                            (w_ats / norm_w) * ats
                        )
                        df_to_score['selection_probability'] = np.round(probs, 4)
                        df_to_score['selected'] = ["Selected" if p >= 0.5 else "Rejected" for p in probs]
                        df_to_score['predicted_suitability'] = np.round(suits, 2)
                        df_to_score['talent_pool_name'] = [engine['clusterer'].cluster_names_.get(c, f"Pool {c}") for c in c_ids]
                        df_to_score['composite_rank_score'] = np.round(comp, 2)
                        df_to_score = df_to_score.sort_values(by='composite_rank_score', ascending=False).reset_index(drop=True)
                        df_to_score['candidate_rank'] = df_to_score.index + 1
                        df_batch = df_to_score
                    except Exception as ex:
                        st.error(f"Error during batch scoring: {ex}")

        # Guard: Only render leaderboard and KPIs if composite_rank_score exists
        if 'composite_rank_score' in df_batch.columns:
            sel_count = (df_batch['selected'] == 'Selected').sum() if 'selected' in df_batch.columns else 0
            total_count = len(df_batch)
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Candidates Evaluated", f"{total_count:,}")
            kpi2.metric("Screened In (Selected)", f"{sel_count:,} ({sel_count/total_count*100:.1f}%)" if total_count else "0")
            kpi3.metric("Highest Composite Score", f"{df_batch['composite_rank_score'].max():.2f}")
            avg_tech = df_batch['technical_test_score'].mean() if 'technical_test_score' in df_batch.columns else 0.0
            kpi4.metric("Avg Technical Test Score", f"{avg_tech:.1f}")
            
            st.markdown("---")
            # Interactive Filters
            st.subheader("Filter & Sort Leaderboard")
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                available_statuses = list(df_batch['selected'].unique()) if 'selected' in df_batch.columns else []
                default_sel = ["Selected"] if "Selected" in available_statuses else available_statuses
                status_filter = st.multiselect("Selection Status", available_statuses, default=default_sel)
            with fcol2:
                roles = list(df_batch['job_role'].dropna().unique()) if 'job_role' in df_batch.columns else []
                role_filter = st.multiselect("Target Job Role", roles, default=roles[:4] if len(roles)>=4 else roles)
            with fcol3:
                min_score = st.slider("Minimum Composite Score", 0.0, 100.0, 50.0, 5.0)
                
            filtered_df = df_batch.copy()
            if status_filter and 'selected' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['selected'].isin(status_filter)]
            if role_filter and 'job_role' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['job_role'].isin(role_filter)]
            if 'composite_rank_score' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['composite_rank_score'] >= min_score]
            
            st.markdown(f"Displaying **{len(filtered_df):,}** matching candidates:")
            
            display_columns = [
                'candidate_rank', 'candidate_id', 'full_name', 'job_role', 
                'highest_education', 'experience_years', 'cgpa',
                'technical_test_score', 'interview_score', 'ats_score', 
                'selection_probability', 'composite_rank_score', 'talent_pool_name'
            ]
            valid_cols = [c for c in display_columns if c in filtered_df.columns]
            
            st.dataframe(
                filtered_df[valid_cols].style.background_gradient(subset=['composite_rank_score'], cmap='Blues'),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Leaderboard (CSV)",
                data=csv_data,
                file_name="candidate_leaderboard.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("Upload a candidate batch dataset to score applicants and view the composite leaderboard.")


# ==========================================
# TAB 3: TALENT POOL CLUSTERING
# ==========================================
with tab3:
    st.header("Unsupervised Talent Pool Clustering")
    st.markdown(
        """
        Unsupervised segmentation (K-Means & Hierarchical Clustering) automatically organizes applicants
        into distinct talent pools based on multidimensional feature patterns.
        """
    )
    
    meta = engine.get('metadata', {})
    profiles = meta.get('cluster_profiles', {})
    
    if profiles:
        st.subheader("Talent Pool Personas & Group Profiles")
        p_cols = st.columns(len(profiles))
        for i, (cid, p_data) in enumerate(profiles.items()):
            with p_cols[i % len(p_cols)]:
                st.markdown(f"#### 🏷️ Group {cid}")
                st.markdown(f"**{p_data['persona_name']}**")
                st.caption(f"{p_data['size']:,} applicants ({p_data['percentage']}%)")
                st.write(f"- **Avg Experience:** {p_data['avg_experience_years']} yrs")
                st.write(f"- **Avg Technical Score:** {p_data['avg_technical_score']}")
                st.write(f"- **Avg Interview Score:** {p_data['avg_interview_score']}")
                st.write(f"- **Avg ATS Score:** {p_data['avg_ats_score']}")
                st.write(f"- **Avg Target Salary:** ${p_data['avg_salary']:,.0f}")
                
        st.markdown("---")
        
    c_img1, c_img2 = st.columns(2)
    with c_img1:
        st.subheader("2D PCA Talent Pool Visualization")
        pca_img_path = os.path.join(ARTIFACTS_DIR, 'pca_talent_clusters.png')
        if os.path.exists(pca_img_path):
            st.image(pca_img_path, caption="Talent Pool Clusters Projected onto 2D PCA Space", use_container_width=True)
        else:
            st.info("PCA Cluster visualization generating...")
            
    with c_img2:
        st.subheader("Elbow Method & Silhouette Scores")
        elbow_img_path = os.path.join(ARTIFACTS_DIR, 'clustering_elbow_silhouette.png')
        if os.path.exists(elbow_img_path):
            st.image(elbow_img_path, caption="Elbow Inertia and Silhouette Curves for k=2..7", use_container_width=True)
        else:
            st.info("Clustering evaluation curve generating...")


# ==========================================
# TAB 4: MODEL PERFORMANCE & ANALYTICS
# ==========================================
with tab4:
    st.header("Model Validation & Performance Benchmark")
    st.markdown("Comprehensive performance metrics across all 11 supervised classification models, cross-validation, bootstrapping, and regression ranking.")
    
    meta = engine.get('metadata', {})
    cls_metrics = meta.get('classification_metrics', [])
    cv_summary = meta.get('cv_summary', {})
    reg_metrics = meta.get('regression_metrics', [])
    boot_ci = meta.get('bootstrap_confidence_intervals', {})
    
    if cls_metrics:
        df_bench = pd.DataFrame(cls_metrics).drop(columns=['confusion_matrix'], errors='ignore')
        st.subheader("1. Supervised Classification Benchmark (Test Set)")
        st.dataframe(
            df_bench.style.highlight_max(subset=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'], color='#d4edda'),
            use_container_width=True
        )
        
        # Display artifacts images
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.subheader("Model Comparison Bar Chart")
            comp_img = os.path.join(ARTIFACTS_DIR, 'model_comparison.png')
            if os.path.exists(comp_img):
                st.image(comp_img, use_container_width=True)
        with m_col2:
            st.subheader("Best Model Confusion Matrix")
            cm_img = os.path.join(ARTIFACTS_DIR, 'confusion_matrix_best.png')
            if os.path.exists(cm_img):
                st.image(cm_img, use_container_width=True)
                
    if cv_summary:
        st.markdown("---")
        st.subheader("2. 5-Fold Stratified Cross-Validation & 95% Confidence Intervals")
        
        cv_rows = []
        for m_name, vals in cv_summary.items():
            ci_val = boot_ci.get(m_name, {})
            cv_rows.append({
                'Model Architecture': m_name,
                'CV Accuracy (Mean +/- Std)': f"{vals['cv_accuracy_mean']:.4f} +/- {vals['cv_accuracy_std']:.4f}",
                'CV F1-Score (Mean +/- Std)': f"{vals['cv_f1_mean']:.4f} +/- {vals['cv_f1_std']:.4f}",
                'CV ROC-AUC': f"{vals['cv_auc_mean']:.4f}",
                'Bootstrapped 95% Accuracy CI': str(ci_val.get('accuracy_ci_95', 'N/A')),
                'Bootstrapped 95% F1 CI': str(ci_val.get('f1_ci_95', 'N/A'))
            })
        st.table(pd.DataFrame(cv_rows))
        
        kfold_img = os.path.join(ARTIFACTS_DIR, 'kfold_cv_comparison.png')
        if os.path.exists(kfold_img):
            st.image(kfold_img, use_container_width=True)
            
    if reg_metrics:
        st.markdown("---")
        st.subheader("3. Suitability Regression Model Evaluation")
        st.markdown("Regression models trained on selected candidates (`selected == 1`) to predict continuous suitability.")
        st.dataframe(pd.DataFrame(reg_metrics), use_container_width=True)
        
        reg_img = os.path.join(ARTIFACTS_DIR, 'regression_residuals.png')
        if os.path.exists(reg_img):
            st.image(reg_img, use_container_width=True)
