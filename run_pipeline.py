import sys
sys.path.insert(0, '.')
from src.pipeline import run_complete_pipeline

if __name__ == '__main__':
    run_complete_pipeline(
        data_path='ai_resume_screening_dataset.csv',
        models_dir='models',
        artifacts_dir='artifacts',
        sample_size=20000,
        random_state=42
    )
