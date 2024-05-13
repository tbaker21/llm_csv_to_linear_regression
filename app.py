import streamlit as st 
from dotenv import load_dotenv
from utils import query_agent

load_dotenv()

st.title("Let's do some analysis on your CSV")
st.header("Please upload your CSV file here:")

# Capture the CSV file 
data = st.file_uploader("Upload CSV file", type="csv")

st.subheader("Enter your query:")
st.write("For example: 'Find the linear regression function to estimate cost based on labor and machines.'")
query = st.text_area("Enter your query")

button = st.button("Generate Response")

if button and data is not None and query:
    # Get response
    answer = query_agent(data, query)
    st.write(answer)
elif button:
    st.write("Please upload a CSV file and enter a query.")
