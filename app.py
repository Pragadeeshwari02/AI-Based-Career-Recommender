#Importing Flask 
from flask import Flask,request,url_for,render_template
import os

# App creation
app=Flask(__name__)

#Creating Home Route
@app.route('/',methods=['GET'])
def homepage():
    try:
        return render_template('index.html')
    except Exception as err:
        return "Error occurred:{}".format(str(err))

#Creating Route for Prediction and Recommendation 
@app.route('/recommend',methods=['GET'])
def recommend():
    try:
        #Getting all Requests
        interests=request.args.get('interests').lower() #Interests
        level=request.args.get('level').lower() #Level
        count=int(request.args.get('count',10)) #Number of Recommendations
        skills=request.args.get('skills').lower() #Skills
        
        #Combining Inputs 
        inputs_for_career="{0} {1} {2}".format(skills,interests,level)
        inputs_for_recommendation="{0} {1} {2}".format(skills,interests,level)

        #Function Calls for Prediction and Recommendation
        career=career_prediction(inputs_for_career,count=5)
        recommendations=course_recommendation(inputs_for_recommendation,count=count)

        #Plotting Graph
        graph(career)

        return render_template('recommendation.html',recommendations=recommendations.to_dict(orient='records'),career=career)
    except Exception as err:
        return "Error occurred:{}".format(str(err))

#All Models Loading

import joblib

tfidf_matrix=joblib.load('Models/tfidf_matrix.pkl')
df=joblib.load('Models/courses_dataframe.pkl')
vectorizer=joblib.load('Models/tfidf_vectorizer.pkl')
career_model=joblib.load('Models/careerpath_model.pkl')
career_vectorizer=joblib.load('Models/careerpath_vectorizer.pkl')


from sklearn.metrics.pairwise import cosine_similarity
#Function for Course recommendation
def course_recommendation(recommendation_input,count=10):
    #Converting into Vector 
    vector=vectorizer.transform([recommendation_input])
    #Calculating similarity
    similarity=cosine_similarity(vector,tfidf_matrix)
    #Top Best Recommendations
    top_best=similarity[0].argsort()[-count:][::-1]
    return df.iloc[top_best][["Course Name","Difficulty Level","Course URL"]]

#Function for Career Prediction
def career_prediction(user_inp_career,count=5):
    #Converting into Vector
    vector=career_vectorizer.transform([user_inp_career])
    #Finding Probabilities
    probs=career_model.predict_proba(vector)[0]
    #Top Career Paths
    top_careers=probs.argsort()[-count:][::-1]
    result=[]
    for career in top_careers:
        name=career_model.classes_[career]
        confidence=round(probs[career]*100,2)
        result.append({"career":name,"confidence":confidence})
    return result

#Importing Librabries for Graph     
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg') 

#Function for Graph 
def graph(career):
    career_df=pd.DataFrame(career)
    career_df.plot(kind='bar', x='career', y='confidence', legend=False,color='darkred')
    plt.title('Top Career Paths')
    plt.xlabel('Career Path')
    plt.ylabel('Confidence (%)')
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig('static/career_graph.png')
    plt.close()

#Run the Flask App
if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
