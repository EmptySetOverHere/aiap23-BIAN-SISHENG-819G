import yaml
from xgboost import XGBClassifier
from sklearn.svm import SVC

def load_model(name, config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_dump(file)
    
    
    
    if name == "XGBClassifier":
        model = XGBClassifier(**config["model"]["parameters"])
    if name == "SVC":
        model = SVC(**config["model"]["parameters"])
    else:
        raise ValueError(f"Model {name} is not supported.")    
    
    
    
    
    
def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
    

    


