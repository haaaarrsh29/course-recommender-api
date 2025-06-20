from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load('hybrid_xgb_model.pkl')
mlb_interests = joblib.load('mlb_interests.pkl')
mlb_tags = joblib.load('mlb_tags.pkl')
le_skill = joblib.load('le_skill.pkl')
le_goal = joblib.load('le_goal.pkl')
le_level = joblib.load('le_level.pkl')

# Sample course data
courses_df = pd.DataFrame({
    'course_id': ['AI_Basics_101', 'Web_Dev_Intermediate', 'Advanced_ML'],
    'tags': [['Machine Learning', 'AI'], ['Web Development', 'JavaScript'], ['Machine Learning', 'Deep Learning']],
    'level': ['Beginner', 'Intermediate', 'Advanced'],
    'popularity_score': [89.5, 92.3, 85.7],
    'watched_pct': [0.0, 0.0, 0.0],
    'quiz_score': [0.0, 0.0, 0.0]
})

 
def create_feature_vector(interests_list, skill_level, goal, course_row):
    interests_encoded = mlb_interests.transform([interests_list])
    
    try:
        skill_enc = le_skill.transform([skill_level])[0]
    except ValueError:
        skill_enc = 0

    try:
        goal_enc = le_goal.transform([goal])[0]
    except ValueError:
        goal_enc = 0

    course_tags_encoded = mlb_tags.transform([course_row['tags']])
    
    try:
        level_enc = le_level.transform([course_row['level']])[0]
    except ValueError:
        level_enc = 0

    level_match = int(skill_enc == level_enc)
    interest_tag_match = int(len(set(interests_list) & set(course_row['tags'])) > 0)
    user_avg_rating = 0

    feature_components = [
        interests_encoded[0],
        course_tags_encoded[0],
        [skill_enc],
        [goal_enc],
        [level_enc],
        [course_row['popularity_score']],
        [course_row['watched_pct']],
        [course_row['quiz_score']],
        [level_match],
        [interest_tag_match],
        [user_avg_rating]
    ]

    feature_vector = np.hstack(feature_components).reshape(1, -1)
    return feature_vector


@app.route('/', methods=['GET', 'POST'])
def home():
    recommended_courses = []
    if request.method == 'POST':
        interests_input = request.form.get('interests', '')
        skill_level = request.form.get('skill_level', 'Beginner')
        goal = request.form.get('goal', 'Career prep')
        interests_list = [i.strip() for i in interests_input.split(',') if i.strip()]

        preds = []
        for _, course in courses_df.iterrows():
            try:
                feature_vector = create_feature_vector(interests_list, skill_level, goal, course)
                prob = model.predict_proba(feature_vector)[0][1]
                preds.append((course['course_id'], prob))
            except Exception:
                preds.append((course['course_id'], 0.0))

        recommended_courses = sorted(preds, key=lambda x: x[1], reverse=True)[:3]

    return render_template('index.html', recommendations=recommended_courses)


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json()

    if not data or 'interests' not in data or 'skill_level' not in data or 'goal' not in data:
        return jsonify({'error': 'Missing required fields'}), 400

    interests_list = [i.strip() for i in data['interests']]
    skill_level = data['skill_level']
    goal = data['goal']

    preds = []
    for _, course in courses_df.iterrows():
        try:
            feature_vector = create_feature_vector(interests_list, skill_level, goal, course)
            prob = model.predict_proba(feature_vector)[0][1]
            preds.append((course['course_id'], float(prob)))
        except Exception as e:
            preds.append((course['course_id'], 0.0))

    top_courses = sorted(preds, key=lambda x: x[1], reverse=True)[:3]

    return jsonify({
        'recommendations': [
            {'course_id': cid, 'likelihood': round(prob, 4)}
            for cid, prob in top_courses
        ]
    })


if __name__ == '__main__':
    app.run(debug=True)
