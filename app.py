import streamlit as st
from utils import validate_file, clean_data, explore_data
from agent import analyse

st.set_page_config(page_title="Data Analysis with LLM", page_icon="📊", layout="centered")

st.title("Data Analysis with LLM")
st.caption("Upload a CSV file and ask questions about your data.")

file = st.file_uploader("Upload your csv file", type=["csv"])

if file is not None:
    if validate_file(file):
        with st.spinner("Loading data..."):
            df = clean_data(file)
            data = explore_data(df)

        st.subheader("Data Overview")
        st.write(f"Shape: {data['shape']}")
        st.write(f"Columns: {data['columns']}")
        st.subheader("Preview")
        st.dataframe(data['preview'])
        st.subheader("Statistics")
        st.dataframe(data['stats'])

        question = st.text_input("Ask a question about your data")
        if question:
            with st.spinner("Analyzing data..."):
                result = analyse(df, question)
                st.subheader("Result")
                st.write(result)
    else:
        st.error("Invalid file format. Please upload a CSV file.")
