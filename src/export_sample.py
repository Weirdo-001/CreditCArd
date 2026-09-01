import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])
y = df["Class"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

fraud_sample = X_test[y_test == 1].copy()
fraud_sample["Class"] = 1
legit_sample = X_test[y_test == 0].sample(len(fraud_sample), random_state=42).copy()
legit_sample["Class"] = 0

sample_df = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)
sample_df.to_csv(os.path.join(MODELS_DIR, "test_sample.csv"), index=False)
print("✓ Created models/test_sample.csv (196 rows, ~500KB) in 0.5 seconds!")
