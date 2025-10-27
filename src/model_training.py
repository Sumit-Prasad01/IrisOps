import joblib 
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.helpers import Loader

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, model_path : str):
        
        self.model_path = model_path
        self.params_dist = DECISION_TREE_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        self.model = None
        self.best_dt_model = None
        self.X_train =  None
        self.X_test = None
        self.y_train =  None
        self.y_test = None

        os.makedirs(self.model_path, exist_ok = True)

        logger.info("Model initialized successfully.")
    
    def load_data(self):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = Loader.load_processed_data(X_TRAIN_LOAD_PATH, X_TEST_LOAD_PATH, y_TRAIN_LOAD_PATH, y_TEST_LOAD_PATH)
        
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise Exception(f"Error while loading processed data.",e)


    def train_model(self):
        try:
            logger.info("Initializing Decision Tree Classifier.")
            self.model = DecisionTreeClassifier(criterion = "gini", max_depth = 30, random_state = 42)

            logger.info("Starting Hyperparameter Tuning...")

            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )

            random_search.fit(self.X_train, self.y_train)

            logger.info("Hyperparameter tuning completed successfully.")

            self.best_params = random_search.best_params_
            self.best_dt_model = random_search.best_estimator_

            logger.info(f"Best Parameters Found: {self.best_params}")

            return self.best_dt_model

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise Exception(f"Error during Decision Tree training:", e)
    
    def evaluate_model(self):
        try:
            logger.info("Evaluating Our Model")

            y_pred = self.best_dt_model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average = "weighted")
            recall = recall_score(self.y_test, y_pred, average = "weighted")
            f1score = f1_score(self.y_test, y_pred, average = "weighted")

            logger.info(f"Accuracy Score :{accuracy}")
            logger.info(f"Precision Score :{precision}")
            logger.info(f"Recall Score :{recall}")
            logger.info(f"F1 Score Score :{f1score}")


            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot = True, cmap = "Blues", xticklabels = np.unique(self.y_test), yticklabels = np.unique(self.y_test))
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted labels")
            plt.ylabel("Actual labels labels")
            plt.savefig(f"{VISUALS_PATH}/Confusion_Matrix.png")
            plt.close()

            logger.info("Confusion Matrix saved successfully.")

        
        except Exception as e:
            logger.error(f"Error During Evaluating model {e}")
            raise CustomException("Failed to Evaluate model",e)
         

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            logger.info("Saving the model.")

            joblib.dump(self.best_dt_model, SAVE_MODEL_PATH)
            logger.info(f"Model Saved to {self.model_path}")
        
        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to Save model",e)
        
    
    def run(self):
        try:
            logger.info("Starting model training pipeline")

            self.load_data()
            self.train_model()
            self.evaluate_model()
            self.save_model()

            logger.info("Model training completed successfully.")

        except Exception as e:
            logger.error(f"Error while running model training pipeline. {e}")
            raise CustomException("Failed to run model training pipeline: ",e)


if __name__ == "__main__":

    trainer = ModelTraining(MODEL_PATH)
    trainer.run()