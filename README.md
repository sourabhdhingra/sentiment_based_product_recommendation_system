# Sentiment Based Product Recommendation System
<p>
This is a product recommendation system generally used by e-commerce websites implemented as a part of my capstone project - IIITB-upgrad. The whole project is deployed at (here)[https://prod-recommender-app.herokuapp.com/]  🙂
</p>

### Repository Content

1. `model.py` - It contains Model class which is our wrapper over User Based Recommendation logic and Sentiment Analsyser. Using the two subcomponents it will first recommend products bought from other similar users, then would predict user sentiment for the existing reviews of products, further filter based upon percentage sentiment score and then recommend top 5 products to the user
2. `app.py` - This forms the interfacing between our backend (model.py) and frontend. Exposes end point `app/predict` which in turn calls the model, gets the result and shows to the user over front-end
3. `static` - this is a sub-folder containing images used by website
4. `templates` - this is a requirement from Flask library, all the HTML pages are to be stored in it. It holds our index.html

### Important Persisted Objects

<p>
Important objects required for modelling end-to-end system were dumped using python joblib which inturn uses pickle.
</p>

1. `master_data.sav` - final data set obtained after all the required EDA, pre-processing, Text - processing from the jupyter notebook
2. `logistic_user_sentiment_classifer_pip3eline` - Entire pip3eline useful for making predictions given we have the reivew text availabe in processed form which is already ensured by master_data.sav. pip3eline contains count vectorizer, Truncated SVD, MinMax Scaler and the chosen Logistic Regression based classifier.
3. `username_bidict.sav` - It is the bidrectional one to one mapping between custom introduced user identifiers of username and the usernames itself. It proves very userful in querying the user_id and username in either way because of using bidict.

### Heroku Deployment related

1. `requirements.txt` - This is obtained using pip3 freeze requirements containing a mix of dependencies from Google Colab and experimentation on local environment. It basically ensures that project is deployed using the same set of libraries on Heroku on which it is certain to perform well.
2. `Procfile` - To indicate Heroku about the starting point of our application telling its a python based web project and deploy app.py
3. `runtime.txt` - To configure python version on Heroku same as that was used on colab.

### Deployment to Heroku

<p>
For deployment to Heroku make sure your project is pushed into a github repo as Heroku requires it as a dependency.
</p>

1. First of all create a virtual environment:
> python3 -m venv capstone
> source capstone/bin/activate

2. Then install all the required dependencies using:
> pip3 install <package-name>

3. Freeze requirements using:
> pip3 freeze > requirements.txt

4. Create a git repo and push your code to github 
 

> git init 
> echo venv > .gitignore
> echo __pycache__ >> .gitignore
> git add -all
> git commit -m "Initial commit"
> git push origin main

5. Now download heroku cli using below:
> curl https://cli-assets.heroku.com/install.sh | sh

6. Login to Heroku:
> heroku login

7. Build a ProcFile using below command:
> echo "web: gunicorn app:app" > Procfile

8. Install gunicorn using pip3
> pip3 install gunicorn==20.0.4
> pip3 freeze > requirements.txt

9. Now commit the files and push to github

10. Then create app on heroku using:
> heroku create product-recommander-app
   
11. start the build process using:
> git push heroku master

12. open app using:
> heroku open

13. watch application logs using:
> heroku logs --app product-recommender-app --tail
