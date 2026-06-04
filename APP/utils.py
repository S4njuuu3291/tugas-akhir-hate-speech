import pickle
from pathlib import Path

def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)

project_root = Path(__file__).parent.parent

artifact = load_artifact(str(project_root / "MODELS" / "baseline" / "model_artifact_2.pkl"))
vectorizer = artifact["vectorizer"]
labels = artifact["label_columns"]
features_list = vectorizer.get_feature_names_out()

print(features_list)

# save feature list ke txt
with open(str(project_root / "MODELS" / "baseline" / "features_list.txt"), "w") as f:
    for feature in features_list:
        f.write(feature + "\n")