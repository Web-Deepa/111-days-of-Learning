
#day 88:streamlit
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

#1.Page config 
st.set_page_config(page_title="ML Predictor", page_icon="🤖", layout="centered")

# 2 Title and description 
st.title("🤖 ML Model Predictor")
st.markdown("This app predicts **Iris flower species** using a Random Forest model")

# 3.Train model 
@st.cache_resource  
def load_model():
    iris  = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(iris.data, iris.target)
    return model, iris.target_names

model, class_names = load_model()

# 4.Sidebar: user inputs 
st.sidebar.header("🌸 Input Features")
# st.sidebar puts widgets in the left panel
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width  = st.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 0.2)

# 5. Main panel: prediction 
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred     = model.predict(features)[0]
prob     = model.predict_proba(features)[0]

st.subheader("🔮 Prediction")
# st.success/info/warning
st.success(f"Predicted species: **{class_names[pred].upper()}**")

# 6. Confidence scores as a bar chart 
st.subheader("📊 Confidence Scores")
prob_df = pd.DataFrame({'Species': class_names, 'Probability': prob})
st.bar_chart(prob_df.set_index('Species'))   

#7 .Show input data
st.subheader("📋 Your Input")
input_df = pd.DataFrame(features, columns=['Sepal L','Sepal W','Petal L','Petal W'])
st.dataframe(input_df)   

# 8. Dataset explorer 
with st.expander("📂 View Training Dataset"):
    iris = load_iris()
    df   = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[t] for t in iris.target]
    st.dataframe(df.head(20))
    st.write(f"Total samples: {len(df)}")

# 9 Footer 
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | https://www.linkedin.com/in/web-deepa/")