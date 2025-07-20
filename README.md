# Python-project
Al in Education:Personalized Learning Systems
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# ======= Step 1: Load Sample Dataset =======
data = pd.DataFrame({
    'student_id': [1, 1, 1, 2, 2, 2],
    'topic': ['Algebra', 'Geometry', 'Calculus', 'Algebra', 'Geometry', 'Calculus'],
    'time_spent': [30, 45, 20, 50, 30, 15],
    'quiz_score': [60, 70, 50, 85, 60, 40],
    'performance': [1, 1, 0, 1, 1, 0]  # 1=Good, 0=Weak
})

# Encode categorical features
data_encoded = pd.get_dummies(data, columns=['topic'])

# ======= Step 2: Train ML Model =======
X = data_encoded.drop(['student_id', 'performance'], axis=1)
y = data_encoded['performance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model accuracy:", accuracy_score(y_test, model.predict(X_test)))

# ======= Step 3: Predict Weak Topics for a Student =======
def recommend_weak_topics(student_id, df):
    student_data = df[df['student_id'] == student_id]
    weak_topics = student_data[student_data['performance'] == 0]['topic'].tolist()
    return weak_topics

# ======= Step 4: NLG-Based Question Generator =======
def generate_question(topic):
    templates = {
        'Algebra': [
            "Solve for x: 2x + 3 = 7",
            "Simplify the expression: 3(x + 2)"
        ],
        'Geometry': [
            "What is the sum of interior angles in a triangle?",
            "Define a right angle."
        ],
        'Calculus': [
            "What is the derivative of x^2?",
            "Find the integral of 1/x."
        ]
    }
    return random.choice(templates.get(topic, ["No question available for this topic."]))

# ======= Step 5: Use Case Demo =======
student_id = 1
weak_topics = recommend_weak_topics(student_id, data)
print("\nWeak Topics for Student", student_id, ":", weak_topics)

for topic in weak_topics:
    question = generate_question(topic)
    print(f"\nQuiz Question on {topic}:\n{question}")
