import os
import numpy as np
import pandas as pd
import joblib
from config.paths_config import *
from src.logger import get_logger
from utils.helpers import Loader
from src.custom_exception import CustomException 
from sklearn.model_selection import train_test_split


logger = get_logger(__name__)

class DataProcessing:

    def __init__(self, file_path : str, processed_data_path : str):

        self.file_path = file_path
        self.df = None
        self.processed_data_path = processed_data_path

        os.makedirs(self.processed_data_path, exist_ok = True)

    def clean_data(self):
        try:
            logger.info("Strating data cleaning process.")
            self.df = Loader.load_data(self.file_path)
            self.df.drop_duplicates(inplace = True)
        
            logger.info("Data cleaned sucessfully.")

            return self.df
        
        except Exception as e:
            logger.error("Failed to clean data.")
            raise CustomException(f"Error while cleaning data.", e)


    def handel_outliers(self, column : str):
        try:
            logger.info("Starting handel outliers operation.")

            self.df = self.clean_data()

            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)

            IQR = Q3 - Q1

            Lower_value = Q1 - 1.5*IQR
            Upper_value = Q3 + 1.5*IQR

            sepal_median = np.median(self.df[column])

            logger.info(f"Lower Value : {Lower_value}")
            logger.info(f"Upper Value : {Upper_value}")
            logger.info(f"Sepal Median : {sepal_median}")

            for i in self.df[column]:
                if i > Upper_value or i < Lower_value:
                    self.df[column] = self.df[column].replace(i, sepal_median)

            
            logger.info("Handled outliers successfully.")

        except Exception as e:
                logger.error("Failed to handel outliers.")
                raise CustomException(f"Error while handling outliers.", e)

    def split_data(self):
        try:
            logger.info("Starting data splitting.")

            X = self.df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
            y = self.df["species"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

            logger.info("Data splitted successfully")

            return X_train, X_test, y_train, y_test

        except Exception as e:
                logger.error("Failed to split data.")
                raise CustomException(f"Error while splitting data.", e)


    def save_data(self):
        try:
            logger.info("Saving our data.")

            X_train, X_test, y_train, y_test = self.split_data()

            joblib.dump(X_train, X_TRAIN_PATH)
            joblib.dump(X_test, X_TEST_PATH)
            joblib.dump(y_train, y_TRAIN_PATH)
            joblib.dump(y_test, y_TEST_PATH)

            logger.info("Data saved successfully.")
        
        except Exception as e:
                logger.error("Failed to save data.")
                raise CustomException(f"Error while saving data.", e)


    def run(self):
        try:
            logger.info("Starting data processing pipeline.")

            self.handel_outliers("sepal_width")
            self.save_data()
        
        except Exception as e:
                logger.error("Failed to run data processing pipeline.")
                raise CustomException(f"Error while runing data processing pipeline.", e)



if __name__ == "__main__":

    processor = DataProcessing(DATA_DIR, PROCESSED_DIR)
    processor.run()