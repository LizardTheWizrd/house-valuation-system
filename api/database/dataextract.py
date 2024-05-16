import base64
import pandas as pd
from io import BytesIO


def decode_data(file_data):
    return base64.b64decode(file_data)


def feature_columns(file_data, prediction_column):

    df = pd.read_csv(BytesIO(file_data))

    columns = df.select_dtypes(include=["int", "float"]).columns.tolist()

    columns.remove(prediction_column)

    print(columns)

    return columns
