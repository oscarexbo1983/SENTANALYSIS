import csv


def submission_csv(y_pred, filename):
    """
    DESCRIPTION:
            Creates the final submission file to be uploaded on Kaggle platform
    INPUT:
            y_pred: List of sentiment predictions. Contains 1 and -1 values
    """

    with open(filename + ".csv", "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()

        r1 = 1

        for r2 in y_pred:
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
            r1 += 1
