import GettingCompanyNews
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Convert to lower case
    words = [word.lower() for word in words]
    
    # Remove punctuation
    words = [word for word in words if word.isalnum()]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    print(text)

    # Get the sentiment scores
    sentiment_scores = sid.polarity_scores(cleaned_text)
    print(sentiment_scores['compound'])

    # Determine if the sentiment is positive, negative, or neutral
    return sentiment_scores['compound']

def rate_all_news(company):
    AllNews = GettingCompanyNews.getCompanyNews(company)    
    newsRatePercent = 0

    for rawArticle in AllNews:
        article = preprocess_text(rawArticle)
        score = analyze_sentiment(article)

        newsRatePercent = newsRatePercent + score

    print(newsRatePercent)

    if newsRatePercent > 2.5:
        print('Promising news about ' + company + '!')
        print('Stock values should be rising or already high, consider purchasing.')
    elif newsRatePercent < .7:
        print('Not the best news about ' + company + '...')
        print('Consider where the low point of the stock will rest and consider purchasing there.')
    else:
        print('Results from ' + company +' news are ordinary.')
        print('Unsure if purchasing stock is the best choice or not with news.')