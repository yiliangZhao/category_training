"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""
import numpy as np
import itertools
from collections import Counter
from gensim.models import Word2Vec 


def pad_sentences(sentences, vocabulary, max_len, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    :param sentences: list of list of word tokens
    :type: list[list[str]]
    :param vocabulary: mapping from word token to index
    :type: dict[word:index]
    :param max_len: maximum number of tokens to consider
    :type: int
    :param padding_word: the character for padding
    :type: str
    :return: padded sentences
    :rtype: list[list[str]]
    """

    # counter to keep track how many items are truncated
    hh = 0

    # sequence_length is the length of the longest document in terms of number of tokens or max-len,
    # whichever is shorter
    sequence_length = min(max(len(x) for x in sentences), max_len)
    padded_sentences = []

    for i in range(len(sentences)):
        sentence = [word for word in sentences[i] if word in vocabulary]
        num_padding = sequence_length - len(sentence)
        if num_padding > 0:
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
            hh += 1
        padded_sentences.append(new_sentence)
    print ('%d in %d documents are truncated' % (hh, len(sentences)))
    return padded_sentences


def build_vocab(sentences, vocab_size=10000):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    :param sentences: list of list of tokens
    :type list[list[str]]
    :param vocab_size: size of vocabulary to keep
    :return: a list where the first item is the mapping from word to index and second item is the list of words
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))  # Counter that counts the number of occurrences of each token

    # Mapping from index to word
    print ('%d words has been reduced to %d' % (len(word_counts), vocab_size))

    # vocabulary_list is a list of tokens that are kept
    vocabulary_list = [x[0] for x in word_counts.most_common(vocab_size)]

    # Mapping from word to index
    vocabulary_mapping = {x: i for i, x in enumerate(vocabulary_list)}
    return [vocabulary_mapping, vocabulary_list]


def build_input_data(sentences, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    :param sentences: list of list of word tokens after padding, each inner list contains max_len tokens
    :type: list[list[str]]
    :param vocabulary: dictionary that maps from word to index
    :type: dict[str:int]
    :return: matrix of dimension number_of_titles * max_len, where each element in the matrix is the index of the token
    """
    x = []
    for sentence in sentences:
        t = []
        for word in sentence:
            if word in vocabulary:
                t.append(vocabulary[word])
        x.append(t)
    x = np.array(x)
    return x


def load_data(all_corpus, max_dict, max_len):
    """
    Loads and preprocessed data for the MR dataset.
    :param all_corpus: list of title + descriptions
    :type: list[str]
    :param max_dict: the number of unique word tokens to be considered (vocabulary size)
    :type: int
    :param max_len: maximum number of tokens to consider
    :type: int
    :return: input matrix, vocabulary mapping from word to index and vocabulary list
    """
    # Load and preprocess data

    # sentences is a list of list of tokens
    sentences = [s.split(" ") for s in all_corpus]

    # vocabulary_mapping is a mapping from word token to index
    # vocabulary_list is a list of word tokens
    vocabulary_mapping, _ = build_vocab(sentences, max_dict)

    # sentences_padded is of the same type as sentences
    sentences_padded = pad_sentences(sentences, vocabulary_mapping, max_len)

    vocabulary_mapping, vocabulary_list = build_vocab(sentences_padded, max_dict + 1)
    x = build_input_data(sentences_padded, vocabulary_mapping)
    return [x, vocabulary_mapping, vocabulary_list]


def infer_data(all_corpus, vocab, max_len, padding_word="<PAD/>"):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    padded_sentences = []
    sentences = [s.split(" ") for s in all_corpus]
    for i in range(len(sentences)):
        sentence = [word for word in sentences[i] if word in vocab]
        sequence_length = len(sentence)
        num_padding = max_len - sequence_length
        if num_padding>0:
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_len]
        print (len(new_sentence))
        padded_sentences.append(new_sentence)

    #print padded_sentences
    x = build_input_data(padded_sentences, vocab)
    return x


def load_data_fusion(all_title, all_descr,  max_dict=10000, max_titlelen=50, max_descrlen=300):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    all_corpus = all_title + all_descr
    sentences_all = [s.split(" ") for s in all_corpus]
    vocabulary, vocabulary_inv = build_vocab(sentences_all, max_dict)
    sentences_title = sentences_all[:len(all_title)]
    sentences_descr = sentences_all[len(all_title):]

    title_padded = pad_sentences(sentences_title, vocabulary, max_titlelen)
    descr_padded = pad_sentences(sentences_descr, vocabulary, max_descrlen)
    vocabulary_inv.append("<PAD/>")
    vocabulary["<PAD/>"] = len(vocabulary_inv) - 1
    x_title = build_input_data(title_padded, vocabulary)
    x_descr = build_input_data(descr_padded, vocabulary)
    return [[x_title, x_descr],  vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_W(word_vecs, vocab, k=50):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, k), dtype='float32')            
    for word in word_vecs:
        W[vocab[word]] = word_vecs[word]
    return W


def load_gensim_w2v(fname, vocab):
    """
    Construct dictionary that maps word to its word embedding
    param fname: path to word embedding file
    :type str
    :param vocab: mapping from word token to index
    :type dict
    :return: dictionary that maps word to its word embedding
    :rtype: dict
    """
    word_vecs = {}
    w2v_model = Word2Vec.load(fname)
    for word in vocab:
        if word in w2v_model.wv.vocab:
            word_vecs[word] = w2v_model.wv[word]
    return word_vecs


def add_unknown_words(word_vecs, vocab, k):
    """
    For words that do not exist in the previous word embedding, initialize the word embedding randomly.
    For words that occur in at least min_df documents, create a separate word vector. 0.25 is chosen so the unknown
    vectors have (approximately) same variance as pre-trained ones
    :param word_vecs: existing word to word embedding mapping
    :param vocab:
    :param k: dimension of word embedding
    :return:
    """
    word_vecs["<PAD/>"] = np.zeros(k, dtype='float32')
    print ('automatic added')
    for word in vocab:
        if word not in word_vecs and word is not "<PAD/>":
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
