import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


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

def data_split(data):
    #to do
    return


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
    emotions = []
    for r in range(len(data)):
        #for c in range(len(data[r])):
        #print(data[r]['utterance'])
        annotations.append(data[r]['annotation'])
    return utterances

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

    print("median words per utterance %.1f" % (words_median))
    #return np.median([len(s.split()) for s in num_words])

def get_class_distribution(data):
    classes = get_emotions(data)
    count_map = Counter(classes)
    #print(list(count_map.keys()))
    plot_class_distribution(count_map)
    print("samples by emotion class %s " % (list(count_map.most_common())))


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