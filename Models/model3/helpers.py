import csv


def read_file(filename):
    """
    DESCRIPTION: 
            Reads a file and returns it as a list
    INPUT: 
            filename: Name of the file to be read
    """
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(line)
    return data


def remove_tweet_id(tweet):
    """
    DESCRIPTION: 
                removes the id from a string that contains an id and a tweet
                e.g "<id>,<tweet>" returns "<tweet>"
    INPUT: 
            tweet: a python string which contains an id concatinated with a tweet of the following format:
           "<id>,<tweet>"
    OUTPUT: 
            only the tweet is returned as a python string
    """
    return tweet.split(',', 1)[-1]


def submission_csv(y_pred, submission_name):
    """
    DESCRIPTION: 
            Creates the final submission file to be uploaded on Kaggle platform
    INPUT: 
            y_pred: List of sentiment predictions. Contains 1 and -1 values
    """

    with open(submission_name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        r1 = 1
        for r2 in y_pred:
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
            r1 += 1
