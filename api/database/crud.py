import json
from .dbhelper import DBHelper


def user_files(user_id):
    db = DBHelper()
    sql = "SELECT file_name FROM Upload where user_id = %s" % user_id
    result = db.fetch(sql)
    return result


def predict_screen_info(user_id, file_name):
    db = DBHelper()
    sql = (
        'SELECT columns, training_model, prediction_column FROM Upload WHERE file_name = "%s" AND user_id = %s'
        % (file_name, user_id)
    )
    result = db.fetch(sql)
    return result


def all_predict_info(user_id):
    db = DBHelper()
    sql = (
        "SELECT file_name, columns, prediction_column, training_model FROM Upload WHERE user_id = %s"
        % user_id
    )
    result = db.fetch(sql)
    return result


def add_upload(
    file_name,
    columns,
    prediction_column,
    training_model,
    trained_model,
    user_id,
):
    db = DBHelper()
    insert_sql = """
        INSERT INTO Upload (file_name, columns, prediction_column, training_model, trained_model, user_id)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (
        f"{prediction_column}-{file_name}",
        json.dumps(columns),
        prediction_column,
        training_model,
        trained_model,
        user_id,
    )
    db.execute(insert_sql, values)
    return "Completed"


def trained_model_for_prediction(user_id, file_name, training_model):
    db = DBHelper()
    sql = (
        'SELECT trained_model FROM Upload WHERE user_id = %s AND file_name = "%s" AND training_model = "%s"'
        % (user_id, file_name, training_model)
    )
    result = db.fetch(sql)
    return result


def username_availability(username):
    db = DBHelper()
    sql = 'SELECT COUNT(*) FROM User WHERE username = "%s"' % (username)
    result = db.fetch(sql)

    count = result[0]["COUNT(*)"]

    if count == 0:
        return True
    else:
        return False


def email_availability(email):
    db = DBHelper()
    sql = 'SELECT COUNT(*) FROM User WHERE email = "%s"' % (email)
    result = db.fetch(sql)

    count = result[0]["COUNT(*)"]

    if count == 0:
        return True
    else:
        return False


def create_user(username, email, password):

    db = DBHelper()

    sql = "INSERT INTO User (username, email, password) VALUES (%s, %s, %s)"
    db.execute(sql, (username, email, password))

    return "User Created"


def password_with_username(username):

    db = DBHelper()
    sql = 'SELECT password FROM User WHERE username = "%s"' % (username)

    result = db.fetchone(sql)
    snipped_result = result["password"].rstrip(b"\x00")

    return snipped_result


def password_with_email(email):

    db = DBHelper()
    sql = 'SELECT password FROM User WHERE email = "%s"' % (email)

    result = db.fetchone(sql)
    snipped_result = result["password"].rstrip(b"\x00")

    return snipped_result


def userID_with_email(email):
    db = DBHelper()
    sql = 'SELECT user_id from User WHERE email = "%s"' % (email)

    result = db.fetchone(sql)

    return result


def userID_with_username(username):
    db = DBHelper()
    sql = 'SELECT user_id from User WHERE username = "%s"' % (username)

    result = db.fetchone(sql)

    return result
