import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    top_medicines = df.iloc[top_indices]['Drug_Name'].tolist()
    return top_medicines, similarity_scores

# Streamlit UI
st.title('Medicine Recommender')

# Parse query parameters for deep linking
query_params = st.experimental_get_query_params()
if 'reason' in query_params:
    input_reason_description = query_params['reason'][0]
    st.text_input('Enter your reason or description:', value=input_reason_description)
else:
    input_reason_description = st.text_input('Enter your reason or description:')

# Button to trigger recommendation
if st.button('Find Medicines'):
    if input_reason_description:
        alternatives, scores = find_alternative_medicines(input_reason_description, df)
        st.write(f"Medicine recommendation for '{input_reason_description}':")
        for alt, score in zip(alternatives, scores):
            st.write(f"{alt}: Similarity Score = {score:.2f}")
    else:
        st.write("Please enter a reason or description.")
