import numpy as np
import util
import math


def get_words(message):
    """Get the normalized list of words from a message string.

    This function splits a message into words, normalizes them, and returns
    the resulting list. For splitting, I've split on spaces. For normalization,
    I've converted everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
        The list of normalized words from the message
    """
    return message.lower().split(" ")
    

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function processes a list of SMS messages to build a vocabulary
    of frequently occurring words. Each message is first tokenized and
    normalized using get_words. A word is included in the dictionary only
    if it appears in at least five distinct messages, which helps remove
    rare words that typically do not improve classification.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A Python dict mapping integer indices to words that appear
        frequently enough to be useful for modeling
    """
    processed_messages = []

    for message in messages:
        processed_messages.append(get_words(message))

    all_words = [item for sublist in processed_messages for item in sublist]
    all_unique_words = set(all_words)

    dict_word = {}
    dict_idx = 0

    for word in all_unique_words:
        message_appearances = 0
        for message_list in processed_messages:
            if word in message_list:
                message_appearances += 1
        if message_appearances >= 5:
            dict_word[dict_idx] = word
            dict_idx += 1
    
    return dict_word

# For testing create_dictionary
# test_list = ["i liKe cookIes",
#              "cookies are great i love them",
#              "i like coffee and cookies",
#              "do you like cookies i like cookies",
#              "cookies are my favorite I loev them",
#              "I really Like Chocolate"]
# print(create_dictionary(test_dic))


def transform_text(messages, word_dictionary):
    """Convert messages into a matrix of vocabulary word counts.

    This function transforms each message into a numeric vector that records
    how many times each vocabulary word appears in the message. Using the
    word dictionary, each word maps to a specific column index in the output
    matrix. Words not present in the dictionary are ignored. Each message is
    tokenized using get_words.

    Args:
        messages: A list of SMS message strings to be transformed
        word_dictionary: A mapping from integer indices to vocabulary words

    Returns:
        A numpy array of shape (num_messages, vocab_size), where entry (i, j)
        represents the number of times the j-th vocabulary word appears in the
        i-th message
    """
    num_rows = len(messages)
    num_cols = len(word_dictionary)

    words_present_matrix = np.zeros((num_rows, num_cols))
    print(f"Shape: {words_present_matrix.shape}")

    vocabulary = list(word_dictionary.values())

    for i in range(num_rows):
        for j in range(num_cols):
            word_to_search = vocabulary[j]
            for word in get_words(messages[i]):
                if word == word_to_search:
                    words_present_matrix[i, j] += 1

    return words_present_matrix

# For testing transform_text
# test_list = ["i liKe cookIes",
#              "cookies are great i love them",
#              "i like coffee and cookies",
#              "do you like cookies i like cookies",
#              "cookies are my favorite I loev them",
#              "I really Like Chocolate"]
# words_dic = create_dictionary(test_list)
# print(transform_text(test_list, words_dic))


def fit_naive_bayes_model(matrix, labels):
    """Fit a Naive Bayes classifier for spam detection.

    This function computes the parameters of a multinomial Naive Bayes model
    using the provided training matrix and labels. It estimates the prior
    probabilities of spam and non-spam messages as well as the conditional
    likelihoods of each vocabulary word given each class. Laplace smoothing is
    applied to avoid zero probabilities for unseen words.

    Args:
        matrix: A numpy array where each row corresponds to a message and each
            column contains the count of a vocabulary word
        labels: A numpy array of binary labels (1 = spam, 0 = not spam)

    Returns:
        A tuple representing the trained model:
            (P(Y=1), P(Y=0), P(X|Y=1) list, P(X|Y=0) list)
    """
    # using Laplace smoothing
    p_spam = (np.sum(labels == 1) + 1) / (labels.size + 2)
    p_not_spam = (np.sum(labels == 0) + 1) / (labels.size + 2)

    num_words = matrix.shape[1]

    p_xis_given_spam = []
    mask_spam = labels == 1
    total_words_in_spam = np.sum(matrix[mask_spam], axis=0)
    total_sum_of_words_spam = np.sum(total_words_in_spam)

    for i in range(num_words):
        sum_spam_column_i = np.sum(matrix[mask_spam, i])
        # using laplace smoothing
        p_xis_given_spam.append((sum_spam_column_i + 1) / (total_sum_of_words_spam + num_words))
    
    p_xis_given_not_spam = []
    mask_not_spam = labels == 0
    total_words_in_not_spam = np.sum(matrix[mask_not_spam], axis=0)
    total_sum_of_words_not_spam = np.sum(total_words_in_not_spam)

    for i in range(num_words):
        sum_not_spam_column_i = np.sum(matrix[mask_not_spam, i])
        # using laplace smoothing
        p_xis_given_not_spam.append((sum_not_spam_column_i + 1) / (total_sum_of_words_not_spam + num_words))

    # Item 1: P(Y=1)
    # Item 2: P(Y=0)
    # Item 3: List containing P(Xi|Y=1) for every vocabulary word
    # Item 4: List containing P(Xi|Y=0) for every vocabulary word
    return (p_spam, p_not_spam, p_xis_given_spam, p_xis_given_not_spam)


def predict_from_naive_bayes_model(model, matrix):
    """Predict whether messages are spam using the trained Naive Bayes model.

    This function applies the previously fitted Naive Bayes model to new message
    data. For each message, it computes the log-probabilities of belonging to
    the spam and non-spam classes based on the observed word counts and the
    model parameters. The predicted label is whichever class has the higher
    log-probability.

    Args:
        model: A trained Naive Bayes model returned by fit_naive_bayes_model
        matrix: A numpy array of word counts for the messages to classify

    Returns:
        A numpy array of predicted binary labels (1 = spam, 0 = not spam)
    """
    num_messages = matrix.shape[0]
    num_words = matrix.shape[1]

    predictions = np.zeros(num_messages)

    p_spam = model[0]
    p_not_spam = model[1]
    p_xis_given_spam = model[2]
    p_xis_given_not_spam = model[3]

    for message_num in range(num_messages):
        message_vect = matrix[message_num]
        p_xis_given_spam_message = []
        p_xis_given_not_spam_message = []

        for word_idx in range(num_words):
            for i in range(int(message_vect[word_idx])):
                p_xis_given_spam_message.append(p_xis_given_spam[word_idx])
                p_xis_given_not_spam_message.append(p_xis_given_not_spam[word_idx])

        log_p_xis_given_spam_message = [math.log(x) for x in p_xis_given_spam_message]
        log_p_xis_given_not_spam_message = [math.log(x) for x in p_xis_given_not_spam_message]
        
        log_p_message_spam = math.log(p_spam) + sum(log_p_xis_given_spam_message)
        log_p_message_not_spam = math.log(p_not_spam) + sum(log_p_xis_given_not_spam_message)
        
        if log_p_message_spam >= log_p_message_not_spam:
            predictions[message_num] = 1

    return predictions


def calculate_precision_recall(predictions, labels):
    """Compute precision and recall for binary classification.

    This function compares predicted labels with true labels to evaluate how
    well a classifier identifies spam messages. Precision measures the fraction
    of predicted spam messages that were actually spam, while recall measures
    the fraction of actual spam messages that the model successfully detected.

    Args:
        predictions: A numpy array of predicted labels (0 or 1)
        labels: A numpy array of true labels (0 or 1)

    Returns:
        A tuple (precision, recall), where each value is a float
    """
    true_positives = np.sum((predictions == 1) & (labels == 1))
    
    false_positives = np.sum((predictions == 1) & (labels == 0))
    
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall


def get_top_five_naive_bayes_words(model, dictionary):
    """Identify the most indicative words for the spam class.

    This function ranks vocabulary words by how strongly they indicate that a
    message is spam. It uses the log ratio of P(word|spam) to
    P(word|not spam), which reflects how much more likely the word is to
    appear in spam messages. The five words with the highest ratios are
    returned in descending order of importance.

    Args:
        model: A Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of integer indices to vocabulary words

    Returns:
        A list of the five most spam-indicative words, ordered from most to
        least informative
    """
    p_xis_given_spam = np.array(model[2])
    p_xis_given_not_spam = np.array(model[3])

    informative_words_array = np.log(p_xis_given_spam / p_xis_given_not_spam)

    sorted_indices = np.argsort(informative_words_array)
    top_five_indices = sorted_indices[-5:]

    return [dictionary[idx] for idx in top_five_indices]


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    # Combine validation and test sets for testing
    test_messages = val_messages + test_messages
    test_labels = np.concatenate([val_labels, test_labels])

    # Print training/test data size
    print(f"Training set: {len(train_messages)} samples")
    print(f"Test set: {len(test_messages)} samples")

    # Create dictionary from the training set
    dictionary = create_dictionary(train_messages)

    # Print dictionary size
    print('Size of dictionary:', len(dictionary))

    # Create word-count matrix from the training set
    train_matrix = transform_text(train_messages, dictionary)

    # Create word-count matrix from the test set
    test_matrix = transform_text(test_messages, dictionary)

    # Fit the model
    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    # Make predictions on the test set
    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    # Calculate accuracy
    accuracy = np.mean(naive_bayes_predictions == test_labels)
    
    # Calculate precision and recall
    precision, recall = calculate_precision_recall(naive_bayes_predictions, test_labels)
    
    # Calculate F1 score (harmonic mean of precision and recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print('Naive Bayes Performance Metrics:')
    print('  Accuracy: {:.4f}'.format(accuracy))
    print('  Precision: {:.4f}'.format(precision))
    print('  Recall: {:.4f}'.format(recall))
    print('  F1 Score: {:.4f}'.format(f1_score))

    # Get the top 5 most indicative words for the spam class
    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)


if __name__ == "__main__":
    main()