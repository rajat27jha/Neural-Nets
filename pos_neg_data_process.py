# in this we are going to use pos and neg data to train and process
# but it has much problem ie they are strings not vectors, so we hv to somehow convert them
# another thing is length is not of equal length for that we hv to make something called lexicon ie dictionary
# to know more about lexicon and we work, see tut 5
import nltk
from nltk.tokenize import word_tokenize
# this will make list of the sentence separating each word by a comma

from nltk.stem import WordNetLemmatizer
# for run ran running etc those three words hv the same meaning, so this helps those three words identify as run
# but tense matters in determining its positive or negative but it will gonna take care of that but i donno how

import numpy as np
import random  # to shuffle our dataset
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000
# althought in our dataset 5000 is present but out NN its each neuron dataset every thing will be present
# in our RAM


def create_lexicon(pos, neg):  # make word tensor
    lexicon = []  # initially it will be empty
    for fi in [pos, neg]:  # in first iteration it will take post doc then neg doc
        with open(fi, 'r') as f:  # with as is used for file streams to clear resources after the work is done
            contents = f.readlines()  # contents will be a list in which element contains every line
            for l in contents[:hm_lines]:  # hm_lines for precautionary purposes
                all_words = word_tokenize(l)  # l will be a line and tokenize will separate every word from it
                lexicon += list(all_words)  # list will add comma after each word so as it could be converted to a list

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]  # i will be a word
    w_counts = Counter(lexicon)
    # for ex w_counts will look like this
    # w_counts = {'the': 7657, 'and': 928} it will show how many no. of times that word occured

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            # here we dont want our lexicon to be large instead short
            # so we r removing all super common words from aur lexicon, so as three hidden layers would be enough
            l2.append(w)
    print(len(l2))
    return l2


def sample_handling(sample, lexicon, classification):  # converting that word in hot format
    featureset = []
    # ex
    # [
    # [[0 1 0 1 1], [0 1]], 0 1 will be negative and 1 0 positive
    # []
    # []
    # ]

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))  # make a list equal to length of lexicon and fill zeroes at each place
            for word in current_words:
                if word.lower() in lexicon:  # earlier we hv removed very common words
                    index_value = lexicon.index(word.lower())  # return the index where word.lower() appears in lexicon
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
            # for each line this whole will be done

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)  # to know why shuffled see tut 6 15:00

    features = np.array(features)

    testing_size = int(test_size*len(features))
    # ex our features looks like like this
    # [[[0 1 0 1 1],[1, 0]],
    # [features, labels],
    # []]
    train_x = list(features[:, 0][:-testing_size])  # this can only be done in np, it will only take features,
    #  not labels from features maybe it can also be done on lists
    # after taking whole features, slice it to -testing_size
    train_y = list(features[:, 1][:-testing_size])  # this is for labels

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 0][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('pos_neg_data_process.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)


