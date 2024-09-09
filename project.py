
import streamlit as st
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
# from pycaret.classification import *
# from pycaret.regression import *
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models



# Set page title and layout
st.set_page_config(page_title="Login", layout="centered")

# Custom CSS to change the background color and adjust button styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ADD8E6; /* Application background */
    }
    .button-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 10px;
    }
    .stButton button {
        background-color: white; /* Button background color (white) */
        color: black; /* Text color (black) */
        border: none; /* Remove border */
        border-radius: 8px; /* Rounded corners */
        padding: 10px 20px;
        margin-bottom: 10px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: white; /* No color change on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Define pages as functions

def login_page():
    st.title("Login")

    # Username input
    username = st.text_input("Username or Email Address---->Mohamed Ammar",value="")

    # Password input
    password = st.text_input("Password----->0000", type="password",value="")

    # Remember me checkbox
    remember_me = st.checkbox("Remember Me")

    # Log In button
    if st.button("Log In"):
        if username == "Mohamed Ammar" and password == "0000":
            st.session_state['logged_in'] = True  # Set session state to indicate user is logged in
            st.session_state.page = 'home'  # Navigate to home page
            #st.experimental_rerun()  # Refresh the page to trigger navigation
        else:
            st.error("Invalid username or password")

    # Forgot password link
    st.markdown("[Lost Your Password?](#)")

def home_page():
    # Step 1: Upload data
    st.title("AutoML App using PyCaret")

    # File uploader for multiple file types
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "json", "xlsx", "tsv", "parquet"])

    # If the file is uploaded
    if uploaded_file is not None:
        # Check the file extension to determine how to read it
        file_extension = uploaded_file.name.split('.')[-1]

        try:
            if file_extension == "csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension == "json":
                data = pd.read_json(uploaded_file)
            elif file_extension == "xlsx":
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type")
                data = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            data = None

        if data is not None:
            # Display a preview of the uploaded dataset
            st.write("Dataset Preview:")
            st.dataframe(data.head())

            # Drop columns
            cols_to_drop = st.multiselect("Select columns to drop (if any):", options=data.columns)
            if cols_to_drop:
                data = data.drop(columns=cols_to_drop)

            # Ask for EDA
            if st.checkbox("Perform EDA?"):
                cols_for_eda = st.multiselect("Select columns to analyze:", options=data.columns)

                if cols_for_eda:
                    st.write(f"Displaying EDA for columns: {', '.join(cols_for_eda)}")
                    eda_data = data[cols_for_eda]

                    # Button for Shape
                    if st.button("Show Shape"):
                        st.write(f"Number of rows: {eda_data.shape[0]}")
                        st.write(f"Number of columns: {eda_data.shape[1]}")
                        st.write("---")

                    # Button for Describe
                    if st.button("Describe"):
                        st.dataframe(eda_data.describe(include='all'), use_container_width=True)
                        st.write("---")

                    # Button for Info
                    if st.button("Info"):
                        buffer = io.StringIO()
                        eda_data.info(buf=buffer)
                        s = buffer.getvalue()
                        st.text(s)
                        st.write("---")

                    # Button for Missing Values
                    if st.button("Missing Values"):
                        st.write(eda_data.isnull().sum())
                        st.write("---")

                    # Button for Correlation Heatmap (only for numerical data)
                    if st.button("Correlation Heatmap"):
                        numeric_cols = eda_data.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 1:  # Need at least two numerical columns to plot a heatmap
                            correlation_matrix = eda_data[numeric_cols].corr()
                            plt.figure(figsize=(10, 8))
                            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
                            st.pyplot(plt)
                        else:
                            st.write("Not enough numerical columns to display correlation heatmap.")

                    st.write("---")

            # Handle missing values
            st.write("Handle Missing Values:")

            # Identify columns with missing values
            missing_cols_cat = [col for col in data.select_dtypes(include=['object']).columns if data[col].isnull().any()]
            missing_cols_num = [col for col in data.select_dtypes(exclude=['object']).columns if data[col].isnull().any()]

            # Handle missing values for categorical columns
            if missing_cols_cat:
                st.write("Categorical Columns with Missing Values:")
                for col in missing_cols_cat:
                    fill_method = st.selectbox(f"How to handle missing values in {col}:", ["Mode", "Add class"], key=f"cat_{col}")
                    if fill_method == "Mode":
                        data[col].fillna(data[col].mode()[0], inplace=True)
                    elif fill_method == "Add class":
                        data[col].fillna('Missing', inplace=True)

            # Handle missing values for numerical columns
            if missing_cols_num:
                st.write("Numerical Columns with Missing Values:")
                for col in missing_cols_num:
                    strategy = st.selectbox(f"How to handle missing values in {col}:", ["Mean", "Median", "Mode"], key=f"num_{col}")
                    if strategy == "Mean":
                        data[col].fillna(data[col].mean(), inplace=True)
                    elif strategy == "Median":
                        data[col].fillna(data[col].median(), inplace=True)
                    else:
                        data[col].fillna(data[col].mode()[0], inplace=True)

            st.write("Missing values handled successfully!")

            # Encode categorical data
            cat_columns = data.select_dtypes(include=['object']).columns.tolist()
            encoding_type = st.radio("Choose categorical encoding type:", ["One-Hot Encoding", "Label Encoding"])
            if encoding_type == "One-Hot Encoding":
                data = pd.get_dummies(data, drop_first=True)
            else:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in cat_columns:
                    data[col] = le.fit_transform(data[col])

            # Select X and Y
            target_col = st.selectbox("Select the target column (Y):", options=data.columns)
            feature_cols = st.multiselect("Select feature columns (X):", options=[col for col in data.columns if col != target_col])

            X = data[feature_cols]
            y = data[target_col]

            # Automatically detect task type: classification or regression
            if data[target_col].dtype == 'object' or len(data[target_col].unique()) < 10:
                task_type = "classification"
                st.write("Detected task: Classification")
                clf = classification_setup(data, target=target_col,verbose=False)
                best_model = classification_compare_models()

            else:
                task_type = "regression"
                st.write("Detected task: Regression")
                reg = regression_setup(data, target=target_col,verbose=False)
                best_model = regression_compare_models()


            # Display the results
            st.write(f"Best model for {task_type} is:")
            st.write(best_model)



# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.page = 'login'

# Display the appropriate page based on session state
if st.session_state.logged_in:
    if st.session_state.page == 'home':
        home_page()
else:
    login_page()

    