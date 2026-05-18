import streamlit as st
import pandas as pd

st.title("Automated Resume Screening Tool")

try:
    df = pd.read_csv("outputs/ranked_candidates.csv")
    st.dataframe(df)
    st.bar_chart(df["Score"])
except:
    st.warning("Please run app.py first to generate output.")