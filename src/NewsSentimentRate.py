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
    if sentiment_scores['compound'] >= 0.3:
        return 'Positive'
    elif sentiment_scores['compound'] == 0:
        return 'Neutral'
    elif sentiment_scores['compound'] < 0.3:
        return 'Negative'

def rate_all_news(company):
    AllNews = GettingCompanyNews.getCompanyNews(company)    
    goodNews = 0
    badNews = 0
    for rawArticle in AllNews:
        article = preprocess_text(rawArticle)
        newsRate = analyze_sentiment(article)

        if newsRate == 'Positive':
            goodNews = goodNews + 1
        elif newsRate == 'Negative':
            badNews = badNews + 1

    print(goodNews)
    print(badNews)

    if goodNews > badNews:
        print('Promising news about ' + company + '!')
    elif badNews > goodNews:
        print('Not the best news about ' + company + '...')
    else:
        print('Results inconclusive... unchanged prediction from stock history')
    
if __name__ == '__main__':
    Company = input('Enter the company you would like to research: ')
    rate_all_news(Company)