from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
                
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {"SOS": 0, "EOS": 1, "UNK": 2}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_chars = 3  # Count SOS and EOS
        self.biggest_word = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            if self.biggest_word < len(word):
                self.biggest_word = len(word)
            for char in word:
                self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    return s

def readLangs(dataset, lang1, lang2, reverse=False, reverse_source=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(dataset+'%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs
    pairs = [[s for s in l.split('\t')] for l in lines]

    if reverse_source:
        for i in range(len(pairs)):
            pairs[i][0] = pairs[i][0][::-1]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH 

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(dataset, lang1, lang2, reverse=False, reverse_source=False):
    input_lang, output_lang, pairs = readLangs(dataset, lang1, lang2, reverse_source=reverse_source)
    print("Read %s word pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s word pairs" % len(pairs))
    print("Counting characters...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted characters:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs