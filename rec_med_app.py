import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the dataset
df = pd.read_csv("medicine.csv")

# Define a function to find alternative medicines based on reason and description
def find_alternative_medicines(input_reason_description, df):
    # Calculate TF-IDF vectors for all medicines using both reason and description
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Reason'] + ' ' + df['Description'])
    
    # Calculate cosine similarity between input reason/description and all medicines
    input_tfidf = tfidf_vectorizer.transform([input_reason_description])
    similarities = cosine_similarity(input_tfidf, tfidf_matrix)
    
    # Get indices of medicines with highest similarity scores
    top_indices = similarities.argsort()[0][-10:-1][::-1]  # Get top 5 similar medicines (excluding input)
    
    # Get similarity scores for top similar medicines
    similarity_scores = similarities[0][top_indices]
    
    # Return names of top similar medicines and their similarity scores
    top_medicines = df.iloc[top_indices][['Drug_Name', 'Reason', 'Description']]
    top_medicines['Similarity_Score'] = similarity_scores
    
    return top_medicines.to_dict(orient='records')

# Streamlit UI
st.title('Medicine Recommender')

# Set page width and layout
st.set_page_config(layout="wide")

# Set background color and padding
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main container
main_container = st.container()

# Sidebar container for input
with main_container:
    st.sidebar.subheader('Input')
    query_params = st.experimental_get_query_params()
    if 'reason' in query_params:
        input_reason_description = query_params['reason'][0]
        input_text = st.sidebar.text_input('Enter your reason or description:', value=input_reason_description)
    else:
        input_text = st.sidebar.text_input('Enter your reason or description:')
    button_clicked = st.sidebar.button('Find Medicines')

# Results container
with main_container:
    st.subheader('Results')

    # Button clicked, show results
    if button_clicked:
        if input_text:
            alternatives = find_alternative_medicines(input_text, df)[:5]  # Limit to top 5
            for medicine in alternatives:
                st.markdown(f"### {medicine['Drug_Name']}")
                st.write(f"**Reason:** {medicine['Reason']}")
                st.write(f"**Description:** {medicine['Description']}")
                st.write(f"**Similarity Score:** {medicine['Similarity_Score']:.2f}")
                st.write("")
        else:
            st.write("Please enter a reason or description.")


