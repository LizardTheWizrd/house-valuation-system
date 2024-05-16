import pandas as pd

import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

import io
from io import BytesIO

class GBRClass:

    def __init__(self, file_data, feature_names, output_feature, model_chosen):
        self.file_data = file_data
        self.feature_names = feature_names
        self.output_feature = output_feature
        self.model_chosen = model_chosen

    def train_model(self):

        # convert the CSV to dataframe
        dataframe = pd.read_csv(BytesIO(self.file_data))

        X = dataframe[self.feature_names]
        y = dataframe[self.output_feature]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Splitting and Training~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Create the GBR model
        gbr_model = GradientBoostingRegressor()

        # Fit the model to the training data
        gbr_model.fit(X_train, y_train)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Saving~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save the trained model
        # saved_model_path = "../trainedModels/" + self.model_chosen + " New.joblib"

        model_bytes = io.BytesIO()
        joblib.dump(gbr_model, model_bytes)
        model_bytes.seek(0)
        
        return model_bytes.getvalue()

        # print("Done")
