from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("medicine.csv")
app = Flask(__name__)


def find_alternative_medicines(input_reason_description, df):
    # Calculate TF-IDF vectors for all medicines using both reason and description
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Reason'] + ' ' + df['Description'])
    
    # Calculate cosine similarity between input reason/description and all medicines
    input_tfidf = tfidf_vectorizer.transform([input_reason_description])
    similarities = cosine_similarity(input_tfidf, tfidf_matrix)
    
    # Get indices of medicines with highest similarity scores
    top_indices = similarities.argsort()[0][-6:-1][::-1]  # Get top 5 similar medicines (excluding input)
    
    # Get similarity scores for top similar medicines
    similarity_scores = similarities[0][top_indices]
    
    # Return names of top similar medicines and their similarity scores
    top_medicines = df.iloc[top_indices][['Drug_Name', 'Reason', 'Description']]
    top_medicines['Similarity_Score'] = similarity_scores
    
    return top_medicines.to_dict(orient='records')

@app.route('/result', methods=['POST'])  
def contact():
    if request.method == 'POST':
        text = request.form['reason']
        res = find_alternative_medicines(text, df)
        # Return JSON response
        return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)
