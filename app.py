from flask import Flask,request,url_for,render_template

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/recommend',methods=['GET'])
def recommend():
    try:
        interests=request.args.get('interests').lower()
        level=request.args.get('level').lower()
        count=int(request.args.get('count',10))
        skills=request.args.get('skills').lower()
        
        user_inp_career=f"{skills} {interests} {level}"
        career=predict_career(user_inp_career,top_n=5)
        user_input=f"{skills} {interests} {level}"
        recommendations=recommend_courses(user_input, top_n=count)

        plot_graph(career)

        return render_template('recommendation.html', recommendations=recommendations.to_dict(orient='records'), career=career)
    except Exception as e:
        return f"An error occurred: {str(e)}"

#MODEL 
import joblib

tfidf_matrix = joblib.load('Models/tfidf_matrix.pkl')
df = joblib.load('Models/courses_dataframe.pkl')
vectorizer = joblib.load('Models/tfidf_vectorizer.pkl')
career_model = joblib.load('Models/careerpath_model.pkl')
career_vectorizer = joblib.load('Models/careerpath_vectorizer.pkl')


from sklearn.metrics.pairwise import cosine_similarity
def recommend_courses(user_input, top_n=10):
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix)
    top_idx = sim_scores[0].argsort()[-top_n:][::-1]
    return df.iloc[top_idx][["Course Name", "Difficulty Level", "Course URL"]]

def predict_career(user_inp_career,top_n=5):
    user_vec_career=career_vectorizer.transform([user_inp_career])
    probs=career_model.predict_proba(user_vec_career)[0] 
    top_idx=probs.argsort()[-top_n:][::-1]
    top_careers = []
    for idx in top_idx:
        career_name = career_model.classes_[idx]
        confidence = round(probs[idx]*100,2)
        top_careers.append({"career": career_name, "confidence": confidence})
    return top_careers
    
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg') 

def plot_graph(career):
    career_df=pd.DataFrame(career)
    career_df.plot(kind='bar', x='career', y='confidence', legend=False,color='darkred')
    plt.title('Top Career Paths')
    plt.xlabel('Career Path')
    plt.ylabel('Confidence (%)')
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig('static/career_graph.png')
    plt.close()

if __name__=="__main__":
    app.run(debug=True)