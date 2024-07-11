import re
import string
PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation))
def tokenize(text):
    splitted = text.split()
    tokens = list(filter(lambda x: x !="",splitted))
    tokens.extend([splitted[i] +" "+ splitted[i+1] for i in range(len(splitted)-1)])
    tokens.append(text)
    return tokens

def lowercase_filter(tokens):
    return [token.lower() for token in tokens]

def punctuation_filter(tokens):
    return [PUNCTUATION.sub('', token) for token in tokens]

def analyze(text):
    tokens = tokenize(text)
    tokens = lowercase_filter(tokens)
    tokens = punctuation_filter(tokens)

    return [token for token in tokens if token]
