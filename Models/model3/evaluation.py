from helpers import *
import os
import numpy as np
import tensorflow as tf
import yaml
import data_helpers
from tensorflow.contrib import learn

##Code modified from: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

def evaluation(timestamp):
    """
    DESCRIPTION: Evaluates the test data set on a pretrained model, gets predicted labels and creates submission file
    INPUT:
            timestamp: Name of the folder that contains the pretrained model
    """

    new_test = open("preprocessed/pre_test_v2.txt", 'rb')
    print("\nLoading Test Tweets")
    test_dataset = []
    for line in new_test:
        line = line.decode('utf8')
        test_dataset.append(line)
    print("Number of Test Tweets:", len(test_dataset))

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        if x.ndim == 1:
            x = x.reshape((1, -1))
        max_x = np.max(x, axis=1).reshape((-1, 1))
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Parameters
    tf.flags.DEFINE_string("checkpoint_dir", os.getcwd() + "/runs/" + timestamp + "/checkpoints/",
                           "Checkpoint directory from training run")
    tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Load Data
    x_raw = test_dataset
    y_test = [1, 0]

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    # Evaluation
    print("\nEvaluating the model with Test Dataset...\n")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_probabilities = None

            for x_test_batch in batches:
                batch_predictions_scores = sess.run([predictions, scores],
                                                    {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
                probabilities = softmax(batch_predictions_scores[1])
                if all_probabilities is not None:
                    all_probabilities = np.concatenate([all_probabilities, probabilities])
                else:
                    all_probabilities = probabilities

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), [int(prediction) for prediction in all_predictions],
                                                  ["{}".format(probability) for probability in all_probabilities]))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)

    CNN_predictions = all_predictions
    CNN_predictions[CNN_predictions == 0] = -1

    # Write predictions to file
    print('Creating final csv submission file')
    submission_csv(CNN_predictions, 'submission_model3.csv')
    print("File created")
