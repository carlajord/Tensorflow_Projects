import pandas as pd
import numpy as np

import datetime as dt
from numpy import newaxis

import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

FILE_NAME = 'input_df.csv'

class PreProcessing(object):

    def __init__(self, forceLoad=False):
        
        if forceLoad:
            self.final_data = self.MakeFinalInputData()
        else:
            try:
                self.final_data = pd.read_csv(FILE_NAME)
            except:
                self.final_data = self.MakeFinalInputData()

    def ImportNewsData(self):
        news_1 = pd.read_parquet('../dataset/news_2015_2022_1.parquet', engine='fastparquet')
        news_2 = pd.read_parquet('../dataset/news_2015_2022_2.parquet', engine='fastparquet')
        news_3 = pd.read_parquet('../dataset/news_2015_2022_3.parquet', engine='fastparquet')
        news_4 = pd.read_parquet('../dataset/news_2015_2022_4.parquet', engine='fastparquet')
        news_5 = pd.read_parquet('../dataset/news_2015_2022_5.parquet', engine='fastparquet')
        news_6 = pd.read_parquet('../dataset/news_2015_2022_6.parquet', engine='fastparquet')
        news_7 = pd.read_parquet('../dataset/news_2015_2022_7.parquet', engine='fastparquet')
        news_8 = pd.read_parquet('../dataset/news_2015_2022_8.parquet', engine='fastparquet')
        
        #Week 1 - May 2nd to May 6th prediction
        news_9 = pd.read_parquet('../dataset/week_1/news_2022_04_20.parquet', engine='fastparquet')
        news_10 = pd.read_parquet('../dataset/week_1/news_2022_04_22.parquet', engine='fastparquet')
        news_11 = pd.read_parquet('../dataset/week_1/news_2022_04_24.parquet', engine='fastparquet')

        #Week 2 - May 9th to May 13th prediction (news from Apr 26th to apr 2+th)
        news_12 = pd.read_parquet('../dataset/week_2/news_2022_04_25.parquet', engine='fastparquet')
        news_13 = pd.read_parquet('../dataset/week_2/news_2022_04_26.parquet', engine='fastparquet')
        news_14 = pd.read_parquet('../dataset/week_2/news_2022_04_27.parquet', engine='fastparquet')
        news_15 = pd.read_parquet('../dataset/week_2/news_2022_04_28.parquet', engine='fastparquet')
        news_16 = pd.read_parquet('../dataset/week_2/news_2022_04_29.parquet', engine='fastparquet')
        news_17 = pd.read_parquet('../dataset/week_2/news_2022_04_30.parquet', engine='fastparquet')

        #Week 3 - May 16th to May 20th prediction (news from May 2nd to May 6th)
        news_18 = pd.read_parquet('../dataset/week_3/2022_05_07_23_57_21.parquet', engine='fastparquet')
        news_19 = pd.read_parquet('../dataset/week_3/2022_05_06_22_55_14.parquet', engine='fastparquet')
        news_20 = pd.read_parquet('../dataset/week_3/2022_05_05_21_54_33.parquet', engine='fastparquet')
        news_21 = pd.read_parquet('../dataset/week_3/2022_05_04_20_54_07.parquet', engine='fastparquet')
        news_22 = pd.read_parquet('../dataset/week_3/2022_05_03_19_53_28.parquet', engine='fastparquet')
        news_23 = pd.read_parquet('../dataset/week_3/2022_05_02_18_52_33.parquet', engine='fastparquet')

        #Week 4 - May 23rd to May 27th prediction (news from May 9th to May 13th)
        news_24 = pd.read_parquet('../dataset/week_4/news_2022_05_09.parquet', engine='fastparquet')
        news_25 = pd.read_parquet('../dataset/week_4/news_2022_05_10.parquet', engine='fastparquet')
        news_26 = pd.read_parquet('../dataset/week_4/news_2022_05_11.parquet', engine='fastparquet')
        news_27 = pd.read_parquet('../dataset/week_4/news_2022_05_12.parquet', engine='fastparquet')
        news_28 = pd.read_parquet('../dataset/week_4/news_2022_05_13.parquet', engine='fastparquet')
        news_29 = pd.read_parquet('../dataset/week_4/news_2022_05_14.parquet', engine='fastparquet')
        news_30 = pd.read_parquet('../dataset/week_4/news_2022_05_15.parquet', engine='fastparquet')

        #Week 5 - May 30th to June 3rd prediction (news from May 16th to May 20th)
        news_31 = pd.read_parquet('../dataset/week_5/news_2022_05_16.parquet', engine='fastparquet')
        news_32 = pd.read_parquet('../dataset/week_5/news_2022_05_17.parquet', engine='fastparquet')
        news_33 = pd.read_parquet('../dataset/week_5/news_2022_05_18.parquet', engine='fastparquet')
        news_34 = pd.read_parquet('../dataset/week_5/news_2022_05_19.parquet', engine='fastparquet')
        news_35 = pd.read_parquet('../dataset/week_5/news_2022_05_20.parquet', engine='fastparquet')

        #Week 6 - June 6th to June 10th prediction (news from May 23rd to May 27th)
        news_36 = pd.read_parquet('../dataset/week_6/news_2022_05_22.parquet', engine='fastparquet')
        news_37 = pd.read_parquet('../dataset/week_6/news_2022_05_23.parquet', engine='fastparquet')
        news_38 = pd.read_parquet('../dataset/week_6/news_2022_05_24.parquet', engine='fastparquet')
        news_39 = pd.read_parquet('../dataset/week_6/news_2022_05_25.parquet', engine='fastparquet')
        news_40 = pd.read_parquet('../dataset/week_6/news_2022_05_26.parquet', engine='fastparquet')
        news_41 = pd.read_parquet('../dataset/week_6/news_2022_05_27.parquet', engine='fastparquet')
        news_42 = pd.read_parquet('../dataset/week_6/news_2022_05_28.parquet', engine='fastparquet')
        news_43 = pd.read_parquet('../dataset/week_6/news_2022_05_29.parquet', engine='fastparquet')
        
        #Week 7 - June 13th to June 17th prediction (news from May 30th to June 5th)
        news_44 = pd.read_parquet('../dataset/week_7/news_2022_05_30.parquet', engine='fastparquet')
        news_45 = pd.read_parquet('../dataset/week_7/news_2022_05_31.parquet', engine='fastparquet')
        news_46 = pd.read_parquet('../dataset/week_7/news_2022_06_01.parquet', engine='fastparquet')
        news_47 = pd.read_parquet('../dataset/week_7/news_2022_06_02.parquet', engine='fastparquet')
        news_48 = pd.read_parquet('../dataset/week_7/news_2022_06_03.parquet', engine='fastparquet')
        news_49 = pd.read_parquet('../dataset/week_7/news_2022_06_04.parquet', engine='fastparquet')
        news_50 = pd.read_parquet('../dataset/week_7/news_2022_06_05.parquet', engine='fastparquet')

        news = pd.concat((news_1, news_2, news_3,news_4,news_5,news_6,news_7,news_8,
                        news_9,news_10,news_11,
                        news_12,news_13,news_14,news_15,news_16,news_17,
                        news_18,news_19,news_20,news_21,news_22,news_23,
                        news_24,news_25,news_26,news_27,news_28,news_29, news_30,
                        news_31, news_32, news_33, news_34, news_35,
                        news_36, news_37, news_38, news_39, news_40, news_41, news_42, news_43,
                        news_44, news_45, news_46, news_47, news_48, news_49, news_50))

        SLBNews = news[news["related_company"]=='SLB']
        SLBNews = SLBNews.sort_values(by=['date'], ascending=True)
        SLBNews=SLBNews.reset_index(inplace=False)
        del SLBNews['index']

        SLBNews['date'] = pd.to_datetime(SLBNews['date']).dt.date
        
        return SLBNews

    def ImportMarketData(self):
        market0 = pd.read_parquet('../dataset/market_2015_2022.parquet', engine='fastparquet')
        market1 = pd.read_parquet('../dataset/week_1/market_2022-04-24.parquet', engine='fastparquet')
        market2 = pd.read_parquet('../dataset/week_2/market_2022_05_01.parquet', engine='fastparquet')
        market3 = pd.read_parquet('../dataset/week_3/market_2022_05_08.parquet', engine='fastparquet')
        market4 = pd.read_parquet('../dataset/week_4/market_2022-05-09.parquet', engine='fastparquet')
        market5 = pd.read_parquet('../dataset/week_5/market_2022-05-21.parquet', engine='fastparquet')
        market6 = pd.read_parquet('../dataset/week_6/market_2022-05-23.parquet', engine='fastparquet')
        market7 = pd.read_parquet('../dataset/week_7/market_2022-05-29.parquet', engine='fastparquet')
        market = pd.concat((market0, market1, market2,market3,market4, market5, market6))

        # Reorganizing columns
        columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        market = market[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

        # Getting only SLB
        sp = market[market["Ticker"]=='SLB']
        sp['Date'] = pd.to_datetime(sp['Date']).dt.date

        return sp


    def RunSentimentAnalysis(self):

        SLBNews = self.ImportNewsData()
        title_score = [sid.polarity_scores(sent) for sent in SLBNews.summary]
        
        compound=[]
        neg=[]
        neu=[]
        pos=[]

        for i in range(len(title_score)):
            compound.append(title_score[i]['compound'])
            neg.append(title_score[i]['neg'])
            neu.append(title_score[i]['neu'])
            pos.append(title_score[i]['pos'])
        
        SLBNews['compound'] = compound
        SLBNews['neg'] = neg
        SLBNews['neu'] = neu
        SLBNews['pos'] = pos

        # Group news that belong to the same date
        SLBNews=SLBNews.groupby(['date']).agg(['mean'],as_index=False)
        SLBNews=SLBNews.reset_index()

        SLBNews.columns = SLBNews.columns.get_level_values(0)
        SLBNews.rename(columns = {'date':'Date'}, inplace = True)

        SLBNews=SLBNews.groupby('Date').agg({'Date': 'first', 'compound': 'first', 'neg': 'first', 'neu': 'first', 'pos': 'first'}).reset_index(drop=True)

        return SLBNews

    def MakeFinalInputData(self):
        
        # merge news and market
        news = self.RunSentimentAnalysis()
        market = self.ImportMarketData()
        data = pd.merge(market,news, how='outer')
        
        # Eliminate rows that do not have slb close value
        data = data[data['Close'].notnull()]

        # Assign a value of 0 to the compounds that have null values
        data['compound'] = data['compound'].fillna(0)

        # Consolidate final input data
        DataModel = pd.DataFrame()
        DataModel['date'] = pd.to_datetime(data.Date).dt.tz_localize(None)
        
        # compute elapsed time
        base = DataModel['date'][0]
        for i in range(len(DataModel)):
            DataModel['date'][i] = (DataModel['date'][i]-base).days
        
        DataModel['wsj']=data['compound']  # no noise
        DataModel['price'] = data['Close']

        self.final_data = DataModel

        DataModel.to_csv(FILE_NAME)

        return DataModel