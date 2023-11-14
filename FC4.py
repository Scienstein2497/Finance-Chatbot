#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import json
import sqlite3
import base64
import openai

# Monkey patching to fix the issue
if st.__version__ == "0.86.0":
    st.button = st.experimental_singleton(st.button)

# Set your OpenAI API key here
openai.api_key = "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"

# Function to upload Parquet file on Streamlit
def upload_parquet_file():
    uploaded_file = st.file_uploader("Choose a Parquet file", type=["parquet"], key="parquet_file_upload")
    return uploaded_file

# Function to load Parquet file
@st.cache(allow_output_mutation=True)
def load_parquet_file(file):
    # Use pyarrow to read Parquet file
    table = pq.read_table(file)
    # Convert to Pandas DataFrame
    df = table.to_pandas()
    return df

# Function to view contents of a Parquet file
def view_parquet_contents(parquet_file):
    # Use pandas to read Parquet file without 'snappy' codec
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    return df

# Function to apply data transformations based on a JSON file
def apply_transformations(df, transformation_json):
    with st.spinner("Applying Transformations..."):
        try:
            # Check if transformation_json is a file object
            if hasattr(transformation_json, 'read'):
                # If it's a file object, read its content
                transformation_json = transformation_json.read().decode("utf-8")

            # Check if the JSON content is empty
            if not transformation_json.strip():
                st.warning("Transformation JSON content is empty.")
                # Drop empty column here
                df = df.dropna(axis=1, how='all')
                return df

            # Print the content of transformation_json for debugging
            st.write("Transformation JSON Content:")
            st.code(transformation_json)

            # Load transformation rules from JSON
            transformations = json.loads(transformation_json)

            # Apply transformations to DataFrame
            for transformation in transformations:
                # Implement your specific transformations here
                if transformation['type'] == 'convert_data_type':
                    column_to_convert = transformation['column']
                    new_data_type = transformation['new_data_type']
                    date_format = transformation.get('date_format')  # Optional date_format for datetime conversion
                    # Convert data type
                    if new_data_type == "datetime" and date_format:
                        df[column_to_convert] = pd.to_datetime(df[column_to_convert], errors='coerce', format=date_format)
                    else:
                        df[column_to_convert] = df[column_to_convert].astype(new_data_type)

                elif transformation['type'] == 'map_values':
                    column_to_map = transformation['column']
                    value_mapping = transformation['value_mapping']
                    # Map values
                    df[column_to_map] = df[column_to_map].map(value_mapping)

                elif transformation['type'] == 'extract_month':
                    column_to_extract = transformation['column']
                    new_column_name = transformation['new_column']
                    # Extract month
                    df[new_column_name] = pd.to_datetime(df[column_to_extract], errors='coerce').dt.month

                elif transformation['type'] == 'drop_columns':
                    columns_to_drop = transformation['columns']
                    # Drop columns
                    df = df.drop(columns=columns_to_drop, errors='ignore')

                elif transformation['type'] == 'filter_rows':
                    filter_condition = transformation['condition']
                    # Filter rows based on condition
                    df = df.query(filter_condition)

                # Additional transformations for Parquet File 2
                elif transformation['type'] == 'rename_column':
                    old_column_name = transformation['old_column']
                    new_column_name = transformation['new_column']
                    # Rename column
                    df = df.rename(columns={old_column_name: new_column_name}, errors='ignore')

                elif transformation['type'] == 'replace_values':
                    column_to_replace = transformation['column']
                    value_mapping = transformation['value_mapping']
                    # Replace values
                    df[column_to_replace] = df[column_to_replace].replace(value_mapping)

                elif transformation['type'] == 'fill_missing_values':
                    column_to_fill = transformation['column']
                    fill_value = transformation['fill_value']
                    # Fill missing values
                    df[column_to_fill] = df[column_to_fill].fillna(fill_value)

                elif transformation['type'] == 'normalize_column':
                    column_to_normalize = transformation['column']
                    # Normalize column
                    df[column_to_normalize] = (df[column_to_normalize] - df[column_to_normalize].mean()) / df[column_to_normalize].std()

                # Additional transformations for Parquet File 3
                elif transformation['type'] == 'round_values':
                    column_to_round = transformation['column']
                    decimal_places = transformation['decimal_places']
                    # Round values
                    df[column_to_round] = df[column_to_round].round(decimal_places)

                elif transformation['type'] == 'calculate_age':
                    column_birth_year = transformation['column']
                    new_column_name = transformation['new_column']
                    # Calculate age
                    df[new_column_name] = pd.to_datetime('today').year - df[column_birth_year]

                elif transformation['type'] == 'encode_categorical':
                    column_to_encode = transformation['column']
                    # Encode categorical column
                    df = pd.get_dummies(df, columns=[column_to_encode], prefix=[column_to_encode])


            # Add more transformations as needed

        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON content: {e}")
            return df

    st.session_state.df_transformed = df.copy()  # Save the transformed DataFrame in session state
    return df

# Function to export data to SQL Database
def export_to_sql(df, connection, table_name, create_table=True):
    with st.spinner("Exporting Data to SQL Database..."):
        if create_table:
            df.to_sql(table_name, connection, index=False, if_exists='replace')
        else:
            df.to_sql(table_name, connection, index=False, if_exists='append')

# Function to export data to CSV
def export_to_csv(df, file_name):
    with st.spinner("Exporting Data to CSV..."):
        df.to_csv(file_name, index=False)
        st.success(f"Data exported to CSV file '{file_name}'.")

# Function to describe the sample dataset using ChatGPT API
def describe_dataset_with_chatgpt(df):
    with st.spinner("Describing Sample Dataset..."):
        # Take a sample of the dataset for analysis (adjust sample size as needed)
        sample_size = min(len(df), 100)
        sample_data = df.sample(sample_size)

        # Create a text prompt for ChatGPT with a shorter sample data
        prompt = f"Describe the dataset:\n\n{sample_data.head().to_markdown(index=False)}"

        # Make a request to the ChatGPT API with a smaller completion length
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=100,  # Adjust this value to fit within the model's maximum context length
            temperature=0.7,
            n=1
        )

        # Extract and display the generated response
        generated_text = response['choices'][0]['text']
        st.subheader("Sample Dataset Description:")
        st.markdown(generated_text)

# Streamlit App
def main():
    st.title("Finance Fraud Detection - Data Upload, Display, and Transformations")

    # Data Upload Section
    st.header("Upload Parquet File")

    # Upload Parquet file
    parquet_file = upload_parquet_file()

    if parquet_file is not None:
        # Display first few rows of the uploaded Parquet data
        # Check if df_parquet is already loaded
        if 'df_parquet' not in st.session_state:
            df_parquet = load_parquet_file(parquet_file)
            st.session_state.df_parquet = df_parquet  # Store in session state to avoid duplicate widget ID error
        else:
            df_parquet = st.session_state.df_parquet

        st.subheader("Preview of Uploaded Parquet Data")
        st.dataframe(df_parquet.head())

        transformation_file = st.file_uploader("Choose a Transformation JSON file", type=["json"], key="transformations")

        # Display Transformation JSON as a table
        if transformation_file is not None:
            st.subheader("Preview of Uploaded Transformation JSON Data")
            df_json = pd.read_json(transformation_file)
            st.dataframe(df_json)

        # Apply Transformations button
        if st.button("Apply Transformations"):
            # Read content of the uploaded file
            transformation_json = None
            if transformation_file is not None:
                transformation_json = transformation_file.read().decode("utf-8")

            # Apply transformations
            df_transformed = apply_transformations(df_parquet, transformation_json)

            # Display transformed data
            st.subheader("Transformed Data:")
            st.dataframe(df_transformed)

            # Export to SQL Database Section
            st.header("Export to SQL Database")

            # SQL Connection
            connection = sqlite3.connect(":memory:")

            # Options for exporting to SQL
            export_option = st.selectbox("Choose an option:", ["Create table and insert data", "Insert into an already existing table"])

            # Custom table name input
            custom_table_name = st.text_input("Enter a custom table name:")

            # Export button
            if st.button("Export to SQL Database"):
                try:
                    if export_option == "Create table and insert data":
                        table_name = custom_table_name if custom_table_name else "default_table"
                        export_to_sql(df_transformed, connection, table_name, create_table=True)
                        st.success(f"Data exported to SQL table '{table_name}'.")
                    elif export_option == "Insert into an already existing table":
                        existing_table_name = custom_table_name if custom_table_name else "default_table"
                        export_to_sql(df_transformed, connection, existing_table_name, create_table=False)
                        st.success(f"Data inserted into SQL table '{existing_table_name}'.")
                except Exception as e:
                    st.error(f"Error exporting to SQL Database: {e}")

            # Export Transformed Data Section
            st.header("Download Transformed Data")

            # CSV file name input for transformed data
            transformed_csv_file_name = st.text_input("Enter a CSV file name for Transformed Data:")

            # Export button for transformed data
            if st.button("Download Transformed Data as CSV"):
                try:
                    export_to_csv(df_transformed, f"{transformed_csv_file_name}.csv")
                    st.success(f"Transformed Data exported to CSV file '{transformed_csv_file_name}.csv'.")
                    st.markdown(get_download_link(df_transformed, f"{transformed_csv_file_name}.csv"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error exporting Transformed Data to CSV: {e}")

    # Describe Sample Dataset with ChatGPT API Section
    st.header("Describe Sample Dataset with ChatGPT API")

    # Describe button
    if st.button("Describe Cleaned Dataset"):
        if 'df_transformed' in st.session_state:
            # If the transformed dataset is available, describe it
            describe_dataset_with_chatgpt(st.session_state.df_transformed)
        elif 'df_parquet' in st.session_state:
            # If only the original dataset is available, describe it
            describe_dataset_with_chatgpt(st.session_state.df_parquet)

if __name__ == "__main__":
    main()

