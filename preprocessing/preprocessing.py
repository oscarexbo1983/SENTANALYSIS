# This python file contains functions for preprocessing raw datasets

import re
import nltk
import enchant
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import words


contractions_dict = {
    """
        DESCRIPTION:
        Dictionary used for expanding contractions
    """
    
    "didn\'t": "did not", "don\'t": "do not",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they shall have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

## Initialize word dictionary
word_dictionary = list(set(words.words()))
## Initialize Stemmer
lancaster_stemmer = LancasterStemmer()
## Initialize Lemmatizer
lmt = nltk.stem.WordNetLemmatizer()
## Initialize tokenizer
punc_tokenizer = RegexpTokenizer(r'\w+')
tweets = []
## Define Stopwords
stopWords = stopwords.words("english")
for w in ['no', 'not', 'nor', 'only', 'against', 'up', 'down', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
          'isn', 'ain', 'aren', 'mightn', 'mustn', 'needn', 'shouldn', 'wasn', 'weren', 'wouldn']:
    stopWords.remove(w)


def filter_digits(tweet):
    """
    DESCRIPTION: Replaces digits with tag <number>
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            a tweet where a digit is replaced by the tag <number>
            (e.g. "today is friday 2017" outputs "today is friday <number>")
    """

    t = []
    for w in tweet.split():
        try:
            num = re.sub('[,\.:%_\-\+\*\/\%\_]', '', w)
            float(num)
            t.append("<number>")
        except:
            t.append(w)
    return (" ".join(t)).strip()


def remove_stopwords(tweet):
    """
    DESCRIPTION: filters stopwords from a tweet
    INPUT:
            tweet: a tweet as a python string
    OUTPUT: 
            a stopword-filtered tweet as a python string
            (e.g. "today is the first friday" outputs "today first friday")
    """

    removal_list = stopWords
    word_list = tweet.split()
    tweet = ' '.join([i for i in word_list if i not in removal_list])
    return tweet


def remove_words(tweet):
    """
    DESCRIPTION: filters repeated tags (user, url and number) from a tweet
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            a tag-filtered tweet as a python string
            (e.g. "user believes today is friday number" outputs "believes today is friday")
    """

    removal_list = ["user", "url", "number"]
    word_list = tweet.split()
    tweet = ' '.join([i for i in word_list if i not in removal_list])
    return tweet


def replace_moreletters(tweet):
    """
    DESCRIPTION: Replaces by 2 repeated letters when there are more than two repeated letters
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            A tweet without letters repeating more than two times
            (e.g. "today is fridayyyyyy" outputs "today is fridayy")
    """

    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", tweet)


def remove_punctuation(tweet):
    """
    DESCRIPTION: Filters punctuation from a tweet
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            punctuation-filtered tweet as a python string without hash, exclamation and apostrophe marks.
            (e.g. "hi, today is friday...." outputs "hi today is friday")
    """

    completar = []
    name1 = tweet.split()
    for i in name1:
        probando = re.split('[-:_*^<>/{}()<>",?&.$%@~!]', i)
        tweet = " ".join(probando)
        completar.append(tweet)
    out = " ".join(completar)
    return out



def lemmatize_single(w):
    """
    DESCRIPTION: Lemmatize a single word
    INPUT:  
            w: a word as a python string
    OUTPUT: 
            lemmatized word as a python string. In case the word cannot be lemmatized
            it will be returned in its first form.
            (e.g. "hildren" outputs "child")
    """

    try:
        a = lmt.lemmatize(w).lower()
        return a
    except Exception as e:
        return w


def lemmatize(tweet):
    """
    DESCRIPTION: Lemmatize all words from a tweet one by one
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            Lemmatized tweet as a python string.
            (e.g. "hey children are you playing?" outputs "hey child are you play?")
    """

    x = [lemmatize_single(t) for t in tweet.split()]
    return " ".join(x)


def stemming_single(word):
    """
    DESCRIPTION: Apply stemming to a single word
    INPUT: 
            w: a word as a python string
    OUTPUT: 
            stemmed word as a python string. In case the word cannot be lemmatized
            it will be returned in its first form.
            (e.g. "played" outputs "play")
    """

    return lancaster_stemmer.stem(word)


def stemming(tweet):
    """
    DESCRIPTION: Apply stemming to all the words from a tweet one by one
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            stemmed tweet as a python string.
            e.g. "they played before" outputs "they play before")
    """

    x = [stemming_single(t) for t in tweet.split()]
    return " ".join(x)


def remove_duplicates(data):
    """
    DESCRIPTION: Remove Duplicate Tweets
    INPUT:
            data: Panda Dataframe
    OUTPUT:
            Panda Dataframe without repetitions
    """
    print('Removing Duplicate tweets')
    print('Number of tweets before duplicates removal:\t', data.shape[0])
    tweets = data.drop_duplicates(subset='tweet')
    print('Number of tweets after duplicates removal:\t', tweets.shape[0])
    print('Duplicates removal DONE')
    return tweets


def hasNumbers(inputString):
    """
    DESCRIPTION: Returns true or false if it has numbers
    INPUT:
            inputString: Python String
    OUTPUT:
            Boolean that says if it has a number
    """
    return bool(re.search(r'\d', inputString))


def split_number_text(tweet):
    """
    DESCRIPTION: Splits numbers and characters that are together
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            tweet with text and numbers split
            (e.g. "89hello77" outputs "89 hello 77")
    """
    link = []
    prueba = tweet.split()
    for r in prueba:
        if hasNumbers(r):
            prueba1 = re.split('(\d+)', r)
            for j in prueba1:
                if re.search('[a-zA-Z]', j) and len(j) > 1:
                    link.append(j)
        elif len(r) > 1:
            link.append(r)
        elif r == '#' or r == '!' or r == "+":
            link.append(r)
    tweet = " ".join(link)
    return tweet


def separate_hash(tweet):
    """
    DESCRIPTION: Separates hash symbol from words
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            tweet with hash symbols and words separated
            (e.g. "#hello" outputs "# hello")
    """
    tweet = " ".join(re.split('(\W#)', tweet))
    return tweet


def interpret_emoji(tweet):
    """
    DESCRIPTION: 
                transforms emoticons to sentiment tags e.g :) --> <smile>
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            transformed tweet as a python string
            (e.g. "today is friday :-) <3" outputs "today is friday <smile> <heart>")
    """
    # Construct emojis
    hearts = ["<3", "♥"]
    eyes = ["8", ":", "=", ";"]
    nose = ["'", "`", "-", r"\\"]
    smilefaces = []
    lolfaces = []
    sadfaces = []
    neutralfaces = []

    for e in eyes:
        for n in nose:
            for s in ["\)", "d", "]", "}", ")"]:
                smilefaces.append(e + n + s)
                smilefaces.append(e + s)
            for s in ["\(", "\[", "{", "(", "["]:
                sadfaces.append(e + n + s)
                sadfaces.append(e + s)
            for s in ["\|", "\/", r"\\", "|"]:
                neutralfaces.append(e + n + s)
                neutralfaces.append(e + s)
            # reversed
            for s in ["\(", "\[", "{", "[", "("]:
                smilefaces.append(s + n + e)
                smilefaces.append(s + e)
            for s in ["\)", "\]", "}", ")", "]"]:
                sadfaces.append(s + n + e)
                sadfaces.append(s + e)
            for s in ["\|", "\/", r"\\", "|"]:
                neutralfaces.append(s + n + e)
                neutralfaces.append(s + e)
            lolfaces.append(e + n + "p")
            lolfaces.append(e + "p")

    smilefaces = set(smilefaces)
    lolfaces = set(lolfaces)
    sadfaces = set(sadfaces)
    neutralfaces = set(neutralfaces)
    t = []
    for w in tweet.split():
        if (w in hearts):
            t.append("<heart>")
        elif (w in smilefaces):
            t.append("<smile>")
        elif (w in lolfaces):
            t.append("<lol>")
        elif (w in neutralfaces):
            t.append("<neutral>")
        elif (w in sadfaces):
            t.append("<sad>")
        else:
            t.append(w)
    return (" ".join(t)).strip()


def correct_misspell(tweet):
    """
    DESCRIPTION: Corrects misspelled words according to words contained in a large big.txt
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            tweet with corrected misspelled words, if word is not in big.txt, the word is not corrected.
            (e.g. "tody is frday" outputs "today is friday")
    """
    
    palabras = tweet.split()
    oracion = []
    for j in palabras:
        new = correction(j)
        oracion.append(new)
    tweet = ' '.join(oracion)
    return tweet

def words(text):
    """
    DESCRIPTION: Splits the tweet into words
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            a list of words contained in the input tweet
            (e,g "today is friday" outputs [today,is,friday])
    """
    return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())):
    """
    DESCRIPTION: Gets a word probability to appear in big.txt
    INPUT:
            word: a python string
            N: total number of words in big.txt
    OUTPUT:
            probability of input word
    """
    
    return WORDS[word] / N


def correction(word):
    """
    DESCRIPTION: Outputs the most probable spelling correction of a word
    INPUT:
            word: a python string
    OUTPUT:
            most probable spelling correction of input word
    """

    return max(candidates(word), key=P)


def candidates(word):
    """
    DESCRIPTION: Generate possible spelling corrections for word
    INPUT:
            word: a python string
    OUTPUT:
            a list of possible spelling corrections for word.
    """

    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    """
    DESCRIPTION: Outputs the subset of words that appear in the dictionary of WORDS
    INPUT:
            word: a python string
    OUTPUT:
            a list of the subset of `words` that appear in the dictionary of WORDS.
    """
    
    return set(w for w in words if w in WORDS)


def edits1(word):
    """
    DESCRIPTION: Outputs all edits that are one edit away from word
    INPUT:
            word: a python string
    OUTPUT:
            set of possible edits that are one edit away from word
    """
    
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """
    DESCRIPTION: Outputs all edits that are two edits away from word
    INPUT:
            word: a python string
    OUTPUT:
            set of possible edits that are two edits away from word
    """
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def replace_hashtag(tweet):
    """
    DESCRIPTION: Replaces hashtags (e.g. #iloveyou) by the tag "<hashtag>"
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            modified tweet, replacing hashtags with the tag <hashtag>
            (e.g. "today is friday #hapoy" outputs "today is friday <hashtag>")
    """
    sentence = []
    line = tweet.split()
    for w in line:
        if "#" in w and len(w) > 1:
            sentence.append("<hashtag>")
        else:
            sentence.append(w)
    tweet = ' '.join(sentence)
    return tweet


def one_space(tweet):
    """
    DESCRIPTION: Removes extra spaces between words, ensures only one space is left
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            modified tweet, containing only one space between words
            (e.g. "today is     friday" outputs "today is friday")
    """
    tweet = re.sub("\s\s+", " ", tweet)
    return tweet


for alphabet in "bcdefghjklmnopqrstuvwxyz":
    word_dictionary.remove(alphabet)


##Code modified from https://github.com/matchado/HashTagSplitter/blob/master/split_hashtags.py
def split_hashtag_to_words(hashtag):
    """
    DESCRIPTION: Generates all possible options for a hashtag
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            set of possible possible options for a given hashtag in tweet
    """
    
    all_possibilities = []
    split_posibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag) + 1))]
    possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]

    for split_pos in possible_split_positions:
        split_words = []
        word_1, word_2 = hashtag[:len(hashtag) - split_pos], hashtag[len(hashtag) - split_pos:]

        if word_2 in word_dictionary:
            split_words.append(word_1)
            split_words.append(word_2)
            all_possibilities.append(split_words)

            another_round = split_hashtag_to_words(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
        else:
            another_round = split_hashtag_to_words(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
    return all_possibilities


def hashtag_remove(tweet):
    """
    DESCRIPTION: Splits hashtags into correct sequence of words (if possible)
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            modified tweet, replacing hashtags with a correct sentence.
            (e.g. "#todayisfriday" outputs "today is friday")
    """
    
    d = enchant.Dict("en_US")
    sentence = []
    words = tweet.split()
    for j in words:
        if "#" in j:
            j = j.replace("#", "")
            if len(j) > 1 and len(j) < 20:
                if d.check(j) == False:
                    split = split_hashtag_to_words(j)
                    if split:
                        hashtag = ' '.join(split[0])
                        sentence.append(hashtag)
                    else:
                        sentence.append(j)
                else:
                    sentence.append(j)
            else:
                sentence.append(j)
        else:
            sentence.append(j)
    tweet = ' '.join(sentence)
    return tweet


##Code modified from https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
    """
    DESCRIPTION: Expands contractions of all words (contracted) in the tweet
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            modified tweet, replacing contracted words by their expanded version
            (e.g. "don't do that, today isn't friday" outputs "do not do that, today is not friday")
    """
    
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)






