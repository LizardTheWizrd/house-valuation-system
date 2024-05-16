import pandas as pd
import numpy as np

import joblib
# ~~~~~~~~~~~~~~~~~~~~~~~~~END OF IMPORTS~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~Work On THIS NOW~~~~~~~~~~~~~~~~~~~~~~~~~~~
def user_input_to_testing_features(feature_template, user_input):
    property_type_mapping = {
        "Flat": 3,
        "House": 4,
        "Lower Portion": 5,
        "Penthouse": 6,
        "Room": 7,
        "Upper Portion": 8,
    }
    
    print("First Half")
    #First Half
    # Extract the features from the user input
    features = user_input[0]
    values = user_input[1]

    # Initialize the template with zeros and empty strings
    user_input_template = [
        ["baths", "bedrooms", "area", "property_type", "location", "city"],
        [values[5], values[4], values[0], values[1], values[3], values[2]]
    ]   
    print(user_input_template)
    user_input_template[1][-2] = values[3] + " - " + values[2]
    del user_input_template[0][-1]
    del user_input_template[1][-1]

    print("Second Half")
    print()
    print(user_input_template)
    # Second Half
    data_path = "..\scripts\Location_Prices.csv"

    # Convert CSV to DataFrame
    location_prices_df = pd.read_csv(data_path)

    for feature, value in zip(user_input_template[0], user_input_template[1]):
        if feature == "property_type":
            # Update the property type index based on the mapping
            print("if 1")
            feature_template[1][feature_template[0].index(f"property_type_{value}")] = 1
        elif feature == "location":
            # Update the location with the encoded value
            print("if 2")
            location_target_encoded = location_prices_df[location_prices_df['location_identifier'] == value]['location_target_encoded'].values[0]

            feature_template[1][
                feature_template[0].index("location_target_encoded")
            ] = location_target_encoded
        else:
            # For other features, update the second row directly
            print("if 3")
            feature_template[1][feature_template[0].index(feature)] = int(value)


    return feature_template


def make_prediction(testing_features, model_path):

    feature_values = np.array(testing_features[1])
    feature_values = feature_values.reshape(1, -1)

    print("Model Path:", model_path)
    loaded_model = joblib.load(model_path)

    predicted_price = loaded_model.predict(feature_values)
    return predicted_price


class DefaultDataClass:

    def __init__(self, training_model, user_values):
        self.training_model = training_model
        self.user_values = user_values
        
    def run_default(self):

        user_input_template = [
            ["baths", "bedrooms", "area", "property_type", "location", "city"],
            [0, 0, 0, "", "", ""],
        ]

        feature_names = [
            [
                "baths",
                "bedrooms",
                "area",
                "property_type_Flat",
                "property_type_House",
                "property_type_Lower Portion",
                "property_type_Penthouse",
                "property_type_Room",
                "property_type_Upper Portion",
                "location_target_encoded",
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]


        if self.training_model == "GBR":
            model_path = r"..\trainedModels\gradient_boosting_regression_trained.joblib"
        elif self.training_model == "RandomForest":
            model_path = r"..\trainedModels\random_forest_trained.joblib"

        
        testing_features = user_input_to_testing_features(feature_names, self.user_values)
        predicted_price = make_prediction(testing_features, model_path)
        predicted_price = predicted_price.item()

        print("Predicted Price:", predicted_price)

        return predicted_price