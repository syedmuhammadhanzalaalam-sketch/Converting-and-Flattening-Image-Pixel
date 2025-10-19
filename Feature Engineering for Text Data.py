# 3.1 Task A.1: Vectorizing Text with CountVectorizer
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
# 1. Define Sample Text Data (Our "Corpus") 
documents = [ 
"The sun is shining today.", 
"The weather is good, the sun is great.", 
"A sunny day is a wonderful day." 
] 
print("--- Raw Text Documents ---")
for i, doc in enumerate(documents):
    print(f"Doc {i+1}: {doc}")
# 2. Apply Feature Engineering: CountVectorizer (Bag-of-Words)
# The CountVectorizer performs tokenization and vocabulary building.
vectorizer = CountVectorizer()
# fit_transform learns the vocabulary and converts the documents into feature vectors
X_text = vectorizer.fit_transform(documents) 
# 3. Analyze the Results 
feature_names = vectorizer.get_feature_names_out() 
text_matrix = X_text.toarray() 
print("\n--- Bag-of-Words (BoW) Transformation ---") 
print(f"Vocabulary (Features): {feature_names}") 
print(f"Shape of Feature Matrix: {text_matrix.shape}") 
# Create a DataFrame for a clean view (for student understanding) 
df_text = pd.DataFrame(text_matrix, columns=feature_names,  
index=[f"Doc {i+1}" for i in range(len(documents))]) 
print("\nBoW Numerical Feature Matrix (Machine Understandable Format):") 
print(df_text) 