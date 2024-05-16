import sys

sys.path.append("..")

from flask import Flask, jsonify, request

from flask_jwt_extended import (
    create_access_token,
    get_jwt_identity,
    jwt_required,
    JWTManager,
    get_jwt,
    set_access_cookies,
    unset_jwt_cookies,
)

from datetime import datetime, timedelta, timezone

import re
from validate_email import validate_email

from database.crud import *
from database.dataextract import *

import pandas as pd
import numpy as np
import joblib
import bcrypt
from io import BytesIO

from trainingModel.Linear_Regression import LinearRegressionClass
from trainingModel.Random_Forest import RandomForestClass
from trainingModel.GBR import GBRClass
from trainingModel.XGBoost import XGBoostClass

from scripts.Main import DefaultDataClass

from utility.error_handlers import *

app = Flask(__name__)

app.config["SECRET_KEY"] = "SuperSecretOMG"
app.config["JWT_COOKIE_SECURE"] = False
app.config["JWT_COOKIE_CSRF_PROTECT"] = False
app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=3)

jwt = JWTManager(app)


@app.after_request
def refresh_expiring_jwts(response):
    try:
        exp_timestamp = get_jwt()["exp"]
        now = datetime.now(timezone.utc)
        target_timestamp = datetime.timestamp(now + timedelta(minutes=60))
        if target_timestamp > exp_timestamp:
            access_token = create_access_token(identity=get_jwt_identity())
            set_access_cookies(response, access_token)
        return response
    except (RuntimeError, KeyError):
        # Case where there is not a valid JWT. Just return the original response
        return response


@app.route("/api/v1/info-for-predict/<user_id>/<file_name>", methods=["GET"])
def info_for_predict(user_id, file_name):

    result = predict_screen_info(user_id, file_name)
    return jsonify(result), 200


@app.route("/api/v1/user-uploaded-files/<user_id>", methods=["GET"])
def user_uploaded_files(user_id):

    result = user_files(user_id)
    return jsonify(result), 200


@app.route("/api/v1/all-info-predict/<user_id>", methods=["GET"])
def all_info_predict(user_id):

    result = all_predict_info(user_id)
    return jsonify(result), 200


@app.route("/api/v1/upload-dataset", methods=["POST"])
def upload_datset():

    upload_id = request.form.get("upload_id", None)
    prediction_column = request.form.get("prediction_column", None)
    training_model = request.form.get("training_model", None)
    user_id = request.form.get("user_id", None)

    print(training_model)

    file_key = next(iter(request.files))
    file_data = request.files[file_key].read()
    file_name = (request.files[file_key]).filename
    file_size = request.content_length

    try:
        columns = feature_columns(file_data, prediction_column)
    except ValueError as e:
        return log_and_return_error(
            "Prediction Column not found",
            404,
            f"Error while looking for prediction column in list. Exception: {e}",
        )

    try:
        if training_model == "Linear Regression":
            model = LinearRegressionClass(
                file_data, columns, prediction_column, training_model
            )

        elif training_model == "Random Forest":
            if file_size <= 3100000:
                model = RandomForestClass(
                    file_data, columns, prediction_column, training_model
                )
            else:
                return log_and_return_error(
                    "File too big for requested model",
                    413,
                    f"Error file too big for the model selected. Exception:",
                )
        elif training_model == "GBR":
            model = GBRClass(file_data, columns, prediction_column, training_model)

        elif training_model == "XGBoost":
            model = XGBoostClass(file_data, columns, prediction_column, training_model)
    except TypeError as e:
        return log_and_return_error(
            "Training Model does not exist",
            404,
            f"Error while chosing training model. Exception: {e}",
        )

    trained_model = model.train_model()

    result = add_upload(
        upload_id,
        file_name,
        columns,
        prediction_column,
        training_model,
        trained_model,
        user_id,
    )

    return jsonify(result), 200


@app.route("/api/v1/predict-value", methods=["POST"])
def predict_value():

    file_name = request.json.get("file_name")
    training_model = request.json.get("training_model")

    column_names = list(request.json.get("column_values"))
    column_values = list(request.json.get("column_values").values())

    user_input = [column_names, column_values]

    if file_name == "default":

        defaultClass = DefaultDataClass(training_model, user_input)
        predicted_value = defaultClass.run_default()

        print(predicted_value)

        return jsonify(predicted_value), 200
    else:
        user_id = request.json.get("user_id")
        # Get from database
        trained_model = trained_model_for_prediction(user_id, file_name, training_model)

        trained_model = trained_model[0]["trained_model"]

        file_obj = BytesIO(trained_model)
        loaded_trained_model = joblib.load(file_obj)

        predicted_value = loaded_trained_model.predict(column_values)

        print(predicted_value)

        return jsonify(predicted_value), 200


@app.route("/api/v1/sign-up", methods=["POST"])
def sign_up():

    username = request.json.get("username")
    email = request.json.get("email")
    password = request.json.get("password")

    username_available = username_availability(username)
    email_available = email_availability(email)

    if len(password) < 8:
        return log_and_return_error(
            "Password must be atleast 8 charcters",
            422,
            f"Error while creating user. Exception:",
        )

    if username_available:
        if email_available:
            hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
            print(hashed_password)
            create_user(username, email, hashed_password)

            return "User Created", 201
        else:
            print("Email already exists")
            return log_and_return_error(
                "Email Already Exists", 409, f"Error while creating user. Exception:"
            )
    else:
        if email_available:
            print("Username already exists")
            return log_and_return_error(
                "Username already exists", 409, f"Error while creating user. Exception:"
            )
        else:
            print("Username and Email already exist")
            return log_and_return_error(
                "Username and Email already exists",
                409,
                f"Error while creating user. Exception:",
            )


@app.route("/api/v1/log-in", methods=["POST"])
def log_in():

    username = request.json.get("username")
    email = request.json.get("email")
    password = request.json.get("password", None)

    if len(username) == 0:
        print("Login with email")
        hashed_password = password_with_email(email)

    elif len(email) == 0:
        print("Login with username")
        hashed_password = password_with_username(username)

        print(hashed_password)

    if bcrypt.checkpw(password.encode("utf-8"), hashed_password):
        return "Passwords match", 200
    else:
        return "Password does not match", 401


# Version 2 of the routes
@app.route("/api/v2/sign-up", methods=["POST"])
def sign_up_v2():
    try:
        username = request.json.get("username")
        email = request.json.get("email")
        password = request.json.get("password")

        if not username:
            return jsonify({"message": "Please enter username"}), 400
        if not email:
            return jsonify({"message": "Please enter email"}), 400

        if not re.match(r"^[a-zA-Z0-9_.][a-zA-Z0-9_.-]{2,}$", username):
            return jsonify({"message": "Enter a valid username"}), 400
        if not validate_email(email):
            return jsonify({"message": "Enter a valid email"})

        username_available = username_availability(username)
        email_available = email_availability(email)

        if len(password) < 8:
            return jsonify({"message": "Password must be atleast 8 charcters"}), 422

        if username_available:
            if email_available:
                hashed_password = bcrypt.hashpw(
                    password.encode("utf-8"), bcrypt.gensalt()
                )
                create_user(username, email, hashed_password)

                user_id = userID_with_username(username)
                response = jsonify({"msg": "User Created"})
                access_token = create_access_token(identity=user_id)
                set_access_cookies(response, access_token)
                return response, 201

            else:
                print("Email already exists")
                return jsonify({"message": "Email Already Exists"}), 409
        else:
            if email_available:
                print("Username already exists")
                return jsonify({"message": "Username already exists"}), 409
            else:
                print("Username and Email already exist")
                return jsonify({"message": "Username and Email already exists"}), 409
    except Exception as e:
        return log_and_return_error(
            "An unexpected error occurred while creating user",
            500,
            f"Error while creating user. Exception: {e}",
        )


@app.route("/api/v2/log-in", methods=["POST"])
def log_in_v2():
    try:
        username = request.json.get("username")
        email = request.json.get("email")
        password = request.json.get("password", None)

        if len(username) == 0:
            print("login with email")
            if email_availability(email):
                return jsonify({"message": "Email or Password doesnt exist"}), 404

            hashed_password = password_with_email(email)
            if bcrypt.checkpw(password.encode("utf-8"), hashed_password):
                user_id = userID_with_email(email)
                response = jsonify({"msg": "login successful"})
                access_token = create_access_token(identity=user_id)
                set_access_cookies(response, access_token)
                return response, 200

            else:
                return jsonify({"message": "Email or Password doesnt exist"}), 404

        elif len(email) == 0:
            print("login with username")
            if username_availability(username):
                return jsonify({"message": "Username or Password doesnt exist"}), 404

            hashed_password = password_with_username(username)
            if bcrypt.checkpw(password.encode("utf-8"), hashed_password):
                user_id = userID_with_username(username)
                response = jsonify({"msg": "login successful"})
                access_token = create_access_token(identity=user_id)
                set_access_cookies(response, access_token)
                return response, 200

            else:
                return jsonify({"message": "Username or Password doesnt exist"}), 404
    except Exception as e:
        return log_and_return_error(
            "An unexpected error occurred while logging in",
            500,
            f"Error while logging in. Exception: {e}",
        )


@app.route("/api/v2/upload-dataset", methods=["POST"])
@jwt_required()
def upload_dataset_v2():

    try:
        prediction_column = request.form.get("prediction_column", None)
        training_model = request.form.get("training_model", None)
        user_id = get_jwt_identity()["user_id"]

        print(training_model)

        file_key = next(iter(request.files))
        file_data = request.files[file_key].read()
        file_name = (request.files[file_key]).filename
        file_size = request.content_length

        try:
            columns = feature_columns(file_data, prediction_column)
        except ValueError as e:
            return log_and_return_error(
                "Prediction Column not found",
                404,
                f"Error while looking for prediction column in list. Exception: {e}",
            )

        try:
            if training_model == "Linear Regression":
                model = LinearRegressionClass(
                    file_data, columns, prediction_column, training_model
                )

            elif training_model == "Random Forest":
                if file_size <= 3100000:
                    model = RandomForestClass(
                        file_data, columns, prediction_column, training_model
                    )
                else:
                    return log_and_return_error(
                        "File too big for requested model",
                        413,
                        f"Error file too big for the model selected. Exception:",
                    )
            elif training_model == "GBR":
                model = GBRClass(file_data, columns, prediction_column, training_model)

            elif training_model == "XGBoost":
                model = XGBoostClass(
                    file_data, columns, prediction_column, training_model
                )
        except TypeError as e:
            return log_and_return_error(
                "Training Model does not exist",
                404,
                f"Error while chosing training model. Exception: {e}",
            )

        trained_model = model.train_model()

        result = add_upload(
            file_name,
            columns,
            prediction_column,
            training_model,
            trained_model,
            user_id,
        )

        return jsonify(result), 200
    except Exception as e:
        return log_and_return_error(
            "An unexpected error occurred while uploading dataset",
            500,
            f"Error while uploading dataset. Exception: {e}",
        )


@app.route("/api/v2/predict-value", methods=["POST"])
@jwt_required(optional=True)
def predict_value_v2():
    try:
        file_name = request.json.get("file_name")
        training_model = request.json.get("training_model")

        column_names = list(request.json.get("column_values"))
        column_values = list(request.json.get("column_values").values())

        user_input = [column_names, column_values]

        if file_name == "default":

            defaultClass = DefaultDataClass(training_model, user_input)
            predicted_value = defaultClass.run_default()

            return jsonify(predicted_value), 200
        else:
            identity = get_jwt_identity()

            if identity == None:
                return jsonify({"message": "Log in is required"}), 401

            user_id = identity["user_id"]

            # Get from database
            trained_model = trained_model_for_prediction(
                user_id, file_name, training_model
            )

            trained_model = trained_model[0]["trained_model"]

            file_obj = BytesIO(trained_model)
            loaded_trained_model = joblib.load(file_obj)

            column_values_2d = np.reshape(column_values, (1, -1))

            predicted_value = loaded_trained_model.predict(column_values_2d)

            predicted_value = predicted_value.tolist()
            print(predicted_value)

            return jsonify(predicted_value), 200
    except Exception as e:
        return log_and_return_error(
            "An unexpected error occurred while predicting value",
            500,
            f"Error while predicting value. Exception: {e}",
        )


@app.route("/api/v2/all-info-predict", methods=["GET"])
@jwt_required(optional=True)
def all_info_predict_v2():

    try:
        jwt = get_jwt()
        if len(jwt) == 0:
            return jsonify({"message": "JWT missing, nothing to retrieve"}), 204
        else:

            user_id = get_jwt_identity()["user_id"]
            result = all_predict_info(user_id)

            for item in result:
                # Convert the "columns" value from a string to a list
                item["columns"] = json.loads(item["columns"])

            return jsonify(result), 200
    except Exception as e:
        return log_and_return_error(
            "An unexpected error occurred while getting info to predict",
            500,
            f"Error while getting info to predict. Exception: {e}",
        )


@app.route("/api/v1/log-out", methods=["POST"])
@jwt_required()
def log_out():
    response = jsonify({"msg": "logout successful"})
    unset_jwt_cookies(response)
    return response


if __name__ == "__main__":
    app.run(debug=True)
