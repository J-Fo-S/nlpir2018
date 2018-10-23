import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def add_arguments(parser):
    parser.add_argument("--personify", type=str, default=False, help="match sentiment to speaker")
    parser.add_argument("--dataclean", type=str, default=True, help="remove extra classes")


#usage
#if args.personify == False:

#elif args.personify == True:

def load_friends_json(data_path):
    friends_path = [os.path.join(data_path,file) for file in os.listdir(data_path) if file.endswith(".json")]
    print(len(friends_path), type(friends_path), friends_path[0])
  
    # JSON is stored as nested lists of 80 dict conversations
    # note this method reduces nested lists to just a list of dicts
    for i in range(len(friends_path)):
        with open(friends_path[i], 'r') as f:
            #print(f)
            if "train" in f.name:
                friends_train = json.load(f)
                print("1st train utterance: %s" % (friends_train[0][0]['utterance']))

    for i in range(len(friends_path)):
        with open(friends_path[i], 'r') as f:
            if "test" in f.name:
                friends_test = json.load(f)
                print("1st test utterance: %s" % (friends_test[0][0]['utterance']))

    for i in range(len(friends_path)):
        with open(friends_path[i], 'r') as f:
            if "dev" in f.name:
                friends_dev = json.load(f)
                print("1st dev utterance: %s" % (friends_dev[0][0]['utterance']))

    if args.dataclean == True:
        friends_train = remove_data_classes(friends_train)
        friends_test = remove_data_classes(friends_test)
        friends_dev = remove_data_classes(friends_dev)

    return friends_train, friends_test, friends_dev

#remove extra classes
def remove_data_classes(data):
    #data_remove = [print(data.pop()) for i in data if data[:][:][:] == 'surprise' or 'non-neutral' or 'fear']
    #data_remove = filter(lambda x:x[:] == "fear" or "non-neutral" or "surprise", data)
    #data_remove = [[subl for subl in nested if subl != "fear" or "non-neutral" or "surprise"] for nested in data]
    #data_remove = [data.remove(subl) for subl in data if subl == "fear" or "non-neutral" or "surprise"]
    #data_remove = [[item for item in seq if item != "fear" or "non-neutral" or "surprise"] for seq in data]
    count = 0
    data_remove = []
    for r in range(len(data)):
        for c in range(len(data[r])):
            if ('surprise' not in data[r][c]['emotion'] and 'fear' not in data[r][c]['emotion'] 
                and 'disgust' not in data[r][c]['emotion'] and 'non-neutral' not in data[r][c]['emotion']):
                count += 1
                data_remove.append(data[r][c])
    print("entries %d " % (count))

    return data_remove

def text_label_vector(data):
    tlv = get_annotations(data)
    tlv = [list(s) for s in tlv]
    #convert to int
    tlv = [[int(y) for y in x] for x in tlv]
    tlv = np.asarray(tlv)
    #reduce 3 removed data columns
    tlv = np.delete(tlv, np.s_[:-4], axis=1)
    labels = np.asarray(get_emotions(data))
    labels = labels[:,np.newaxis]
    tlv_one = np.expand_dims(np.max(tlv,axis=1),axis=1)
    labels = np.concatenate((tlv_one,labels),axis=1)
    return tlv, tlv_one, labels

# the following "get" funcs return a vector of each attribute
def get_speakers(data):
    speakers = []
    for r in range(len(data)):
        #for c in range(len(data[r])):
        #print(data[r]['utterance'])
        speakers.append(data[r]['speaker'])
    return speakers

def get_utterances(data):
    utterances = []
    for r in range(len(data)):
        #for c in range(len(data[r])):
        #print(data[r]['utterance'])
        utterances.append(data[r]['utterance'])
    return utterances

def get_emotions(data):
    emotions = []
    for r in range(len(data)):
        #for c in range(len(data[r])):
        #print(data[r]['utterance'])
        emotions.append(data[r]['emotion'])
    return emotions

def get_annotations(data):
    annotations = []
    for r in range(len(data)):
        #for c in range(len(data[r])):
        #print(data[r]['utterance'])
        annotations.append(data[r]['annotation'])
    return annotations

def get_num_words_per_sample(data):
    """Returns the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    word_lines = get_utterances(data)
    words_median = np.asarray([len(s) for s in word_lines])
    words_median = np.median(words_median)
    plot_sample_length_distribution(word_lines, words_median)
    plot_frequency_distribution_of_ngrams(word_lines, (1,1))
    plot_frequency_distribution_of_ngrams(word_lines, (2,2))
    plot_frequency_distribution_of_ngrams(word_lines, (3,3))
    plot_frequency_distribution_of_ngrams(word_lines, (4,4))

    print("median words per utterance %.1f" % (words_median))
    #return np.median([len(s.split()) for s in num_words])

def get_class_distribution(data):
    classes = get_emotions(data)
    count_map = Counter(classes)
    #print(list(count_map.keys()))
    plot_class_distribution(count_map)
    print("samples by emotion class %s " % (list(count_map.most_common())))

def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=50):
    """Plots the frequency distribution of n-grams.
    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams where n is %d' % (ngram_range[0]))
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()

def plot_sample_length_distribution(word_lines, words_median):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in word_lines], 50)
    plt.xlabel('Word length of sample utterances')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution - median utterance %.1f words' % (words_median))
    plt.show()


def plot_class_distribution(count_map):
    """Plots the class distribution.
    # Arguments
    """
    emotions = list(count_map.keys())
    y_pos = np.arange(len(emotions))
    plt.bar(y_pos, list(count_map.values()), align='center', alpha=0.5)
    plt.xticks(y_pos, emotions)
    plt.ylabel('Number of samples')
    plt.title('Sample class distribution of %d total samples ' % (np.sum(list(count_map.values()))))
    plt.show()


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
