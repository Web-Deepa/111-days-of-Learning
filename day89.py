#day 89: Gradio 
import gradio as gr
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from transformers import pipeline

# 1. Train model
iris  = load_iris()
rf    = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(iris.data, iris.target)


# 2.Load sentiment model
sentiment = pipeline("sentiment-analysis")


# 3. Iris Predictor 
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred     = rf.predict(features)[0]
    probs    = rf.predict_proba(features)[0]
    label    = iris.target_names[pred]
  
    return {name: float(p) for name, p in zip(iris.target_names, probs)}, f"🌸 {label.upper()}"

iris_interface = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.4, label="Sepal Length"),
        gr.Slider(2.0, 4.5, value=3.4, label="Sepal Width"),
        gr.Slider(1.0, 7.0, value=1.3, label="Petal Length"),
        gr.Slider(0.1, 2.5, value=0.2, label="Petal Width"),
    ],
    outputs=[
        gr.Label(label="Confidence Scores"),    
        gr.Textbox(label="Prediction"),
    ],
    title="🌸 Iris Flower Classifier",
    description="Adjust sliders to predict Iris species"
)


# 4. Sentiment Analyzer 
def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text"
    result = sentiment(text)[0]
    emoji  = "😊" if result['label'] == 'POSITIVE' else "😢"
    return f"{emoji} {result['label']} (confidence: {result['score']:.3f})"

sentiment_interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type your review here...", label="Input Text"),
    outputs=gr.Textbox(label="Sentiment Result"),
    title="💬 Sentiment Analyzer",
    examples=[
        ["This product is absolutely amazing!"],
        ["Terrible experience, very disappointed"],
        ["It was okay, nothing special"],
    ]
)




# 5. Combine into tabbed interface 
app = gr.TabbedInterface(
    [iris_interface, sentiment_interface],
    ["🌸 Iris", "💬 Sentiment", "🏥 Cancer Check"],
    title="ML Dashboard — Day 89"
)


app.launch(share=False)  