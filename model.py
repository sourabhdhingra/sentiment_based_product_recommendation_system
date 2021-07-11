import joblib
import pandas as pd
import math


class Model:
    
    def __init__(self):
        self.lr_clf_pipeline = joblib.load('logistic_user_sentiment_classifer_pipeline.sav')
        self.user_based_recommender = joblib.load('user_based_recommender.sav')
        self.username_bidict = joblib.load('username_bidict.sav')
        self.master_data = joblib.load('master_data.sav')
    
    def get_user_id(self, username):
        '''
        Will return user_id for the given username
        '''
        return self.username_bidict.inverse[username]

    def get_top_20_recommendations(self, user_id):
        '''
        Will return top 20 product ids based on review ratings
        '''
        top_20_recommendations = self.user_based_recommender.recommend_top_20_products(user_id)
        return top_20_recommendations

    def filter_top_5_recommendations(self, product_ids):
        '''
        Will consume top 20 recommended products
        perform sentiment analysis
        and return top 5 products based upon percentage/mean sentiment score
        '''

        # filtering dataframe for our product ids
        top_20 = self.master_data.query('product_id in {}'.format(list(product_ids.index)))

        # making sentiment predictions for top_20
        top_20_pred = self.lr_clf_pipeline.predict(top_20['reviews_text'])

        # assigning predictions to a column 'pred_user_sentiment'
        top_20['pred_user_sentiment'] = pd.Series(top_20_pred, dtype=int)

        # imputing NaN values as per review ratings
        top_20['pred_user_sentiment'] = top_20.apply(lambda x: self.imputer(x), axis=1 )
        
        # preparing final_result
        final_result = top_20[['product_id', 'pred_user_sentiment']].groupby('product_id').mean()
        final_result = final_result.sort_values(by='pred_user_sentiment', ascending=False)
        final_result.reset_index(inplace=True)
        
        # mapping product names with product ids on final_result dataset
        get_name_brand = lambda x: pd.unique(top_20[top_20['product_id']==x]['name_brand'])[0]
        final_result['name_brand'] = final_result.apply(lambda x: get_name_brand(x['product_id']), axis=1)
        
        # picking up top 5
        top_5 = final_result.head(5)
        
        return top_5


    def imputer(self, row):
        if math.isnan(row['pred_user_sentiment']):
            if row['reviews_rating'] < 4:
                return 0
            else:
                return 1
        else:
            return row['pred_user_sentiment']

