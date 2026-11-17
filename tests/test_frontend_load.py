import sys
sys.path.insert(0, '.')
import frontend.app as app

print("Checking Streamlit frontend imports and cached engine...")
engine = app.load_local_engine()
print("Engine ready:", engine.get('ready'))
if not engine.get('ready'):
    print("Engine error:", engine.get('error'))
    sys.exit(1)

meta = engine['metadata']
print("Loaded metadata keys:", list(meta.keys()))
print("Best classifier in frontend:", meta.get('best_classifier_name'))
print("Best regressor in frontend:", meta.get('best_regressor_name'))
print("Clusters in frontend:", list(meta.get('cluster_profiles', {}).keys()))

print("\nFRONTEND APP LOGIC VALIDATED SUCCESSFULLY!")
