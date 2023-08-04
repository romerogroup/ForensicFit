import joblib

# Load the model from the file
dt = joblib.load("combinationDecisionTree.joblib")
X = <REPLACE THIS VECTOR WITH THE DATA OUTPUT OF THE TWO CNN>
y_predicted = dt.predict(X)