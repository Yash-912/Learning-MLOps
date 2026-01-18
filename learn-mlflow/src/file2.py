import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 

import dagshub
dagshub.init(repo_owner='temp99358', repo_name='Learning-MLOps', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/temp99358/Learning-MLOps.mlflow")
  
wine=load_wine()
X=wine.data
y=wine.target

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)
 
max_depth=6
n_estimators=10
mlflow.set_experiment('learn-mlflow')
with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    
    mlflow.log_metric('accuracy',acc)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    
    cf=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cf, annot=True, fmt='d',cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confu Matrix")
    
    plt.savefig('conf_matrix.png')
    
    # mlflow.log_artifact('conf_matrix.png')
    mlflow.set_tags({'Author':'Yash','Version':'v1-learn'})
    mlflow.sklearn.log_model(rf,'Random Forest model')
    print(acc)
    