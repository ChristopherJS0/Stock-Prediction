from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import re
import requests
#html5lib will also be necessary

def createLink(search):
    link = "https://www.google.com/search?sca_esv=8e72db86bea876f8&sxsrf=ADLYWIJp1ZpBq0B_KOnTdcszX8B2BzgAag:1716701456597&q=PLACEHOLDER&tbm=nws&source=lnms&prmd=nvismbt&sa=X&ved=2ahUKEwik2IWky6qGAxVJlIkEHUUgAagQ0pQJegQIDhAB&biw=1536&bih=730&dpr=1.25"
    fixedSearch = search.replace(' ','+')
    link = link.replace('PLACEHOLDER',fixedSearch+'+news')
    return link

def getCompanyNews(company):
    '''
    param: String of company to research
    rtype: List of given company news.
    '''
    link = createLink(company)
    AllNews = []
    req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()

    with requests.Session() as c:
        soup = BeautifulSoup(webpage,'html5lib')
        linkPattern = re.compile('BNeawe')
        for item in soup.find_all('div', attrs={'class': 'kCrYT'}):
            rawNewsArticle = item.find('div',class_=linkPattern)
            newsArticle = rawNewsArticle.get_text()
            AllNews.append(newsArticle)

        if len(AllNews) == 0:
            print("An error has occurred and no news on this company has been found!")
            return AllNews
        else:
            return AllNews

if __name__ == '__main__':
    nvidiaNews = getCompanyNews('apple')
    for article in nvidiaNews:
        print('\n')
        print(article)