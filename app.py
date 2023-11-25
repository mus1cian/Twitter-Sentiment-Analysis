import streamlit as st
import joblib

st.title("Twitter Sentiment Analysis")
st.markdown("SVM to determine if a tweet is good, bad or neutral")


tweet = st.text_area("Tweet")


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #F63366;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #ffffff;
    color:#000000;
    font:"sans serif";
    }
</style>""", unsafe_allow_html=True)

if st.button("Predict sentiment"):  
    svm = joblib.load("model.pkl")
    result = svm.predict([tweet])
    if result == 1:
        result = "Neutral"
    elif result == 2:
        result = "Positive"
    else:
        result = "Negative"
    st.text(result)

