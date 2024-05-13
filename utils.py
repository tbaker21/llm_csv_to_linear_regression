
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd 
from langchain_openai import OpenAI
import statsmodels.api as sm
import streamlit as st

# Creating a subroutine 
def query_agent(data, query):
    # Parse the CSV file and create a Pandas DataFrame from its contents.
    df = pd.read_csv(data)
    
    # Debugging output
    st.write("CSV Columns:", df.columns.tolist())

    # Initialize the OpenAI LLM
    llm = OpenAI()
    if 'linear regression' in query:
        # Use the LLM to interpret the query and identify the dependent and independent variables
        interpretation_prompt = f"The following is a query to perform linear regression on a dataset:\n\n'{query}'\n\nIdentify the dependent variable and the independent variables in the format: 'dependent variable: <dependent_var> / independent variables: <independent_var1>, <independent_var2>, ...' "
        interpretation_response = llm(interpretation_prompt)

        # Debugging output
        st.write("LLM Interpretation Response:", interpretation_response)

        try:
            parts = interpretation_response.split('/')
            dependent_var = None
            independent_vars = []
            for part in parts:
                part = part.strip()
                if part.lower().startswith('dependent variable:'):
                    dependent_var = part.split(':')[1].strip()
                elif part.lower().startswith('independent variables:'):
                    independent_vars = [var.strip() for var in part.split(':')[1].split(',')]


            if dependent_var is None:
                raise ValueError("Failed to parse the dependent variable from the LLM's response.")

            # Debugging output
            st.write("Parsed Dependent Variable:", dependent_var)
            st.write("Parsed Independent Variables:", independent_vars)

        except Exception as e:
            return f"Failed to interpret the query. Error: {str(e)}"

        # Validate the variables are correctly identified
        if dependent_var not in df.columns:
            return f"The dependent variable '{dependent_var}' is not in the CSV file."

        for var in independent_vars:
            if var not in df.columns:
                return f"The independent variable '{var}' is not in the CSV file."

        # Debugging output
        st.write("Validated Dependent Variable:", dependent_var)
        st.write("Validated Independent Variables:", independent_vars)

            # Prepare the data for regression
        X = df[independent_vars].apply(pd.to_numeric, errors='coerce')
        y = df[dependent_var].apply(pd.to_numeric, errors='coerce')

        # Add a constant term for the intercept
        X = sm.add_constant(X)

        # Fit the linear regression model
        model = sm.OLS(y, X, missing='drop').fit()

        # Generate predictions
        predictions = model.predict(X)

        # Prepare the results
        result_summary = model.summary().as_text()

        # Construct the regression equation
        intercept = model.params['const']
        coefficients = model.params.drop('const')
        equation = f"{dependent_var} = {intercept:.4f}"
        for var, coef in coefficients.items():
            equation += f" + ({coef:.4f} * {var})"

        # Format the output
        summary_output = f"### Linear Regression Model Summary\n\n```\n{result_summary}\n```\n"
        equation_output = f"### Linear Regression Equation\n\n`{equation}`\n"
        predictions_output = f"### Predictions\n\n{predictions.head().to_string(index=False)}\n"

        return summary_output + equation_output + predictions_output
    else:
        agent = create_pandas_dataframe_agent(llm, df, verbose=True)

        #Python REPL: A Python shell used to evaluating and executing Python commands 
        #It takes python code as input and outputs the result. The input python code can be generated from another tool in the LangChain
   
        return agent.run(query)

