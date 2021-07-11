from flask import Flask, request, render_template
from model import Model, LRModel, UserBasedRecommender
import json

app = Flask(__name__)

model = Model()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Get values from browser
    username = request.args['username']
    user_id = model.get_user_id(username)
    product_ids = model.get_top_20_recommendations(user_id)
    top_5 = model.filter_top_5_recommendations(product_ids)
    display_result = top_5[['product_id', 'name_brand']]
    display_result[['product_name', 'brand']] = display_result['name_brand'].str.split('|',expand=True)
    display_result.drop(['name_brand','product_id'], inplace=True, axis=1)
    display_result_dict = display_result.to_dict('records')
    col_names = display_result.columns.values
    return render_template('index.html', records=display_result_dict, colnames=col_names)

if __name__ == '__main__':
    app.run(debug=True)