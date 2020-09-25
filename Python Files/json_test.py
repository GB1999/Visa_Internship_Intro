import os
import json
import nltk
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk

nltk.download('vader_lexicon')
#nltk.download('words')
#nltk.download('maxent_ne_chunker')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')

messages = []

# identify digits/special characters for later removal
punctuation = re.compile(r'[-.?,:;()|0-9]')
emojis = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
threshold = .3

with open("D:\Documents\Side Projects (CS)\Python\Visa Intro\Source Data\Direct Messages\messages.json", encoding='utf-8') as json_file:
    data = json.load(json_file)
    for direct_messages in data:
        conversation = direct_messages['conversation']
        for message in conversation:
            if(message['sender'] == "gage_e_benham"):
                try:
                    messages.append(message['text'])
                except:
                    print(message)




for message in messages:

    score = SentimentIntensityAnalyzer().polarity_scores(message)
    if score['pos'] > threshold:
        print(message)




