import re
from flask import Flask, request, render_template
from model import Model
import os

app = Flask(__name__)

class UserBasedRecommender:
    '''
    A class to encapsulate User Based Recommendation System for persistence
    '''
    def __init__(self, user_rating_matrix):
        self.user_rating_matrix = user_rating_matrix
    
    def recommend_top_20_products(self, user_input_id):
        return self.user_rating_matrix.loc[user_input_id].sort_values(ascending=False)[0:20]


class LRModel:
    '''
    A model wrapper which would have all the transformations
    '''
    def __init__(self, vectr, svd, scaler, clf):
        self.vectr = vectr
        self.svd = svd
        self.scaler = scaler
        self.clf = clf
  
    def predict(self, data):
        data_1 = self.vectr.transform(data)
        data_2 = self.svd.transform(data_1)
        data_3 = self.scaler.transform(data_2)
        result = self.clf.predict(data_3)
        return result


model = Model()


@app.route('/')
def home():
    print("***** path over heroku - {} ******".format(os.getcwd()) )
    print("below directories exist **** {}".format(os.listdir()) )
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Get values from browser
    username = request.args['username']
    try:
        user_id = model.get_user_id(username)
        product_ids = model.get_top_20_recommendations(user_id)
        top_5 = model.filter_top_5_recommendations(product_ids)
        display_result = top_5[['product_id', 'name_brand']]
        display_result[['product_name', 'brand']] = display_result['name_brand'].str.split('|',expand=True)
        display_result.drop(['name_brand','product_id'], inplace=True, axis=1)
        display_result_dict = display_result.to_dict('records')
        col_names = display_result.columns.values
        return render_template('index.html', records=display_result_dict, colnames=col_names)
    except KeyError:
        return render_template('index.html',input_error="Enter a valid username!")
    

if __name__ == '__main__':
    app.run(debug=True)