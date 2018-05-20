from keras.preprocessing.text import Tokenizer
import helpers
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def get_featured_data(train_tweets, test_tweets, ngram=False):
    """
    DESCRIPTION: Return data ready to be an input of Keras models
    INPUT:
            train_tweets: list of train tweets
            test_tweets: list of test tweets
            ngram: Boolean to use ngram extension
    OUTPUT:
            x_train: Numerical Representation of train features
            y_train: Numerical Representation of train labels
            x_test: Numerical Representation of test features
            max_features: Max number that can have a word representation, related to vocabulary and ngram extension
            tokenizer: Keras Tokenizer object that includes the vocabulary
            ngram_rep: Vocabulary for ngram extension
    """

    np.random.seed(0)

    train_len = len(train_tweets)
    tokenizer = Tokenizer(num_words=20000, filters='')
    tokenizer.fit_on_texts(train_tweets)

    print("Creating Train Sequences")

    train_seq = tokenizer.texts_to_sequences(train_tweets)

    print("Creating Test Sequences")

    test_seq = tokenizer.texts_to_sequences(test_tweets)

    # Get Data depending if ngram extension is required
    if ngram:
        train_seq, test_seq, max_features, ngram_rep = get_ngram_featured_data(train_seq, test_seq, tokenizer)
    else:
        max_features = len(tokenizer.word_index) + 1

    # Get a tweet numerical representation vector of the same length

    print("Padding Train Sequences")

    x_train = pad_sequences(train_seq, maxlen=60)

    print("Padding Test Sequences")

    x_test = pad_sequences(test_seq, maxlen=60)

    y_train = np.array(int(train_len / 2) * [0] + int(train_len / 2) * [1])

    # Shuffling the dataset
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]


    return x_train, y_train, x_test, max_features, tokenizer, ngram_rep


def get_ngram_featured_data(train_seq, test_seq, tokenizer):
    """
    DESCRIPTION: Return data ready to be an input of Keras models
    INPUT:
            train_seq: list of train sequences
            test_seq: list of test sequences
            tokenizer: Keras Tokenizer object that contains the vocabulary
    OUTPUT:
            ngram_train_seq: Numerical Representation of train features, including ngram extension
            ngram_test_seq: Numerical Representation of test features, including ngram extension
            max_features: Max number that can have a word representation, related to vocabulary and ngram extension
            ngram_rep: Vocabulary for ngram extension
    """

    s = set()

    for seq in train_seq:
        s.update(helpers.get_bigram(seq))

    max_features = len(tokenizer.word_index) + 1

    ngram_rep = { value: max_features + index for index, value in enumerate(s) }

    return helpers.get_ngram_seq(train_seq, ngram_rep),\
           helpers.get_ngram_seq(test_seq, ngram_rep),\
           np.max(list(ngram_rep.values())) + 1,\
           ngram_rep

def get_pretrained_data(vocab, tokenizer, max_features=None):
    """
    DESCRIPTION: Return pretrained embedding matrix with ngram extesions
    INPUT:
            vocab: Vocabulary
            tokenizer: Keras Tokenizer object
            max_features: Max number that can have a word representation, related to vocabulary and ngram extension, if it is not given it is None by default.
    OUTPUT:
            max_features: Updated Max number that can have a word representation, related to vocabulary and ngram extension
            embedding_matrix: Pretrained embedding matrix filtered using vocabulary words and ngram extesion
    """

    if max_features is None:
        max_features = len(vocab)

    print('Extracting word vectors from Glove Pretrained Data')
    embeddings_index = {}
    with open('embeddings/glove200.txt', "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Creating embedding')
    embedding_matrix = np.zeros((max_features + 1, 200))

    index_word = { v: k for k, v in tokenizer.word_index.items() }

    for word, i in vocab.items():
        if i > max_features:
            continue
        if type(word) is tuple:
            ngram_list = []

            for word_index in list(word):
                word_str = index_word[word_index]
                embedding_vector = embeddings_index.get(word_str)

                # Look for each word representation of the ngram
                if embedding_vector is not None:
                    ngram_list.append(embedding_vector)
                else:
                    ngram_list.append(np.zeros((1, 200)))

            # Take the mean of both word vectors to add to embedding matrix
            embedding_matrix[i] = np.mean(np.array(ngram_list), axis=0)
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    print('Embedding matrix created')
    return max_features, embedding_matrix

def get_pretrained_data_without_ngram(vocab, max_features=None):
    """
    DESCRIPTION: Return pretrained embedding matrix without ngram extension
    INPUT:
            vocab: Vocabulary
            max_features: Max number that can have a word representation, related to vocabulary, if it is not given it is None by default.
    OUTPUT:
            max_features: Updated Max number that can have a word representation, related to vocabulary
            embedding_matrix: Pretrained embedding matrix filtered using vocabulary words
    """


    if max_features is None:
        max_features = len(vocab)

    print('Extracting word vectors from Glove Pretrained Data')
    embeddings_index = {}
    with open('embeddings/glove200.txt', "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Creating embedding')
    embedding_matrix = np.zeros((max_features + 1, 200))
    for word, i in vocab.items():
        if i > max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print('Embedding matrix created')
    return max_features, embedding_matrix