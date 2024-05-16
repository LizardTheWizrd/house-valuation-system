DROP TABLE Upload;
DROP TABLE User;


CREATE TABLE `User` (
		`user_id` INT AUTO_INCREMENT,
        `username` VARCHAR(50) NOT NULL,
        `email` VARCHAR(255) NOT NULL,
        `password` BINARY(255),
		PRIMARY KEY (`user_id`)
);

CREATE TABLE `Upload` (
		`upload_id` INT AUTO_INCREMENT,
        `file_name` VARCHAR(255) NOT NULL,
        `columns` JSON,
        `prediction_column` VARCHAR(255) NOT NULL,
        `training_model` VARCHAR(255) NOT NULL,
        `trained_model` LONGBLOB,
        `user_id` INT,
        PRIMARY KEY (`upload_id`),
        FOREIGN KEY (`user_id`) REFERENCES User(user_id)
);

INSERT INTO User VALUES (554422, "3antar" , "Abu_3antar@gmail.com", NULL);
INSERT INTO User VALUES (334466, "Naruto", "I_am_hokage@gmail.com", NULL);

INSERT INTO Upload Values (000001, "Stock Price-Price", JSON_ARRAY("numberOfShares","column2","column3"), "price", "RandomForest", NULL, 554422);
INSERT INTO Upload Values (000004, "Stock Price-Price", JSON_ARRAY("numberOfShares","column2","column3"), "price", "GBR", NULL, 554422);
INSERT INTO Upload Values (000002, "Stock Price-numberOfShares", JSON_ARRAY("price","column2","column3"), "numberOfShares", "GBR", NULL, 554422);
INSERT INTO Upload Values (000003, "Diseases-ChanceOfDisease", JSON_ARRAY("age", "height", "weight"), "chanceOFDisease", "RandomForest", NULL, 334466);
INSERT INTO Upload Values (000005, "Diseases-ChanceOfDisease", JSON_ARRAY("age", "height", "weight"), "chanceOFDisease", "RandomForest", NULL, 554422);

SELECT * FROM User;
SELECT * FROM Upload;

SELECT columns, training_model FROM Upload 
WHERE file_name = "Stock Price-Price"
AND user_id = 554422;

SELECT file_name FROM Upload
where user_id = 554422;

SELECT file_name, columns, prediction_column, training_model FROM Upload
Where user_id = 554422;

SELECT trained_model, prediction_column FROM Upload
WHERE user_id = 554422
AND file_name = "price-99k_Final.csv"
AND training_model = "GBR";

SELECT COUNT(*) FROM User WHERE email = "%s";

SELECT COUNT(*) FROM User WHERE username = 'Naruto';

SELECT file_name, columns, prediction_column, training_model FROM Upload WHERE user_id = 1

SELECT user_id from User WHERE email = 'test1@gmail.com';

