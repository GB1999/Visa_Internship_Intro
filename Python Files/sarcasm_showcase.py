from nltk.sentiment.vader import SentimentIntensityAnalyzer

score_text = "I just love when people don't comment in their code."
score = SentimentIntensityAnalyzer().polarity_scores(score_text)

#represents the "positivity" score of the text
print(score['pos'])





