import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load the data
df = pd.read_csv("/mnt/data/ml-6 final (1).ipynb")  # Update this with the correct file path

# Preprocess the data
df.columns = [col.lower() for col in df.columns]
X = df.drop(columns=['target'])  # Assuming 'target' is the column with labels
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and training
pipeline = Pipeline([
    ('preprocessor', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'model.pkl')
