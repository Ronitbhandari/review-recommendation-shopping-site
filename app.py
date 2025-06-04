import csv
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
import pandas as pd
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)


# load  data & model  
df = pd.read_csv('data/assignment3_II.csv')
with open('model/vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
with open('model/model.pkl', 'rb') as f:  clf = pickle.load(f)

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text: str) -> str:
    """
    Lowercase + tokenize on whitespace/punctuation, keep alphabetic tokens,
    lemmatize each token, then join back into a single space‐separated string.
    """
    tokens = word_tokenize((text or "").lower())
    lemmas = [lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha()]
    return " ".join(lemmas)

# Created two new columns that hold the lemmatized title & description
df['lemmatized_title'] = df['Clothes Title'].fillna('').map(lemmatize_text)
df['lemmatized_desc']  = df['Clothes Description'].fillna('').map(lemmatize_text)

reviews = {}


@app.route('/', methods=['GET'])
def index():
    # One entry per Clothing ID
    items_df = df.drop_duplicates(subset='Clothing ID')[[
        'Clothing ID',
        'Clothes Title',
        'Clothes Description'
    ]]
    items = items_df.to_dict(orient='records')
    return render_template('index.html', items=items)
    
#  Lemmatization‐Based Matching 
@app.route('/search', methods=['GET'])
def search():
    """
    We can search based on the string through clothing items. 
    I am using lemmatization for the search feature to match all the 
    possible items with the searched word.

    Steps:
      1) Lowercase & tokenize the user’s query.
      2) Lemmatize each alphabetic token.
      3) Build a regex that matches any one of those lemmas as whole words.
      4) Return all items whose precomputed lemmatized_title OR lemmatized_desc
         contains at least one of the query’s lemmas.
    """
    raw_query = request.args.get('q', '').strip()
    query = raw_query.lower()
    items, count = [], 0

    if query:
        # Tokenize & lemmatize the query
        query_tokens = [
            lemmatizer.lemmatize(tok) 
            for tok in word_tokenize(query) 
            if tok.isalpha()
        ]

        if query_tokens:
            # 2) Build a regex that matches any lemma as a whole word
            #    if query_tokens = ["dress","flow"], pattern = r'\b(?:dress|flow)\b'
            pattern = r'\b(?:' + "|".join(re.escape(tok) for tok in query_tokens) + r')\b'

            #  Filter: match precomputed lemmatized columns
            mask_title = df['lemmatized_title'].str.contains(pattern, regex=True)
            mask_desc  = df['lemmatized_desc'].str.contains(pattern, regex=True)

            filtered = (
                df[mask_title | mask_desc]
                .drop_duplicates(subset='Clothing ID')
                [['Clothing ID', 'Clothes Title', 'Clothes Description']]
            )

            items = filtered.to_dict(orient='records')
            count = len(items)

    return render_template(
        'index.html',
        items=items,
        query=raw_query,  
        count=count
    )   
    
@app.route('/item/<int:item_id>', methods=['GET'])
def item_detail(item_id):
    rows = df[df['Clothing ID'] == item_id]
    if rows.empty:
        abort(404)

    # Item‐level info (first row)
    item = rows.iloc[0][[
        'Clothing ID','Clothes Title','Clothes Description','Department Name'
    ]].to_dict()

    # All dataset reviews for this item
    existing_reviews = rows[[
        'Title','Review Text','Rating','Recommended IND','Age'
    ]].to_dict(orient='records')

    # Any new reviews added in this session (or empty list)
    new_reviews = reviews.get(item_id, [])

    return render_template(
        'item.html',
        item=item,
        existing_reviews=existing_reviews,
        new_reviews=new_reviews
    )
    
 
@app.route('/item/<int:item_id>/review', methods=['GET', 'POST'])
def new_review(item_id):
    #  Verify the item exists
    item_rows = df[df['Clothing ID'] == item_id]
    if item_rows.empty:
        abort(404)

    #  Handle the POST after “Confirm & Submit Review”
    if request.method == 'POST' and request.form.get('confirm'):
        #  Extract hidden form fields (populated by your JS)
        review_title = request.form.get('review_title_hidden', '').strip()
        review_text  = request.form.get('review_text_hidden', '').strip()
        age  = request.form.get('age_hidden', '').strip()
        rating       = request.form.get('rating_hidden', '').strip()
        final_label  = int(request.form.get('predicted_hidden', '0') or 0)

        # Look up item-level fields from the existing DataFrame
        base = item_rows.iloc[0]
        division     = base['Division Name']
        department   = base['Department Name']
        class_name   = base['Class Name']
        clothes_t    = base['Clothes Title']
        clothes_desc = base['Clothes Description']

        # a list matching the CSV column order:
        new_row_list = [
            item_id,
            age,            
            review_title,
            review_text,
            rating,
            final_label,
            0,                 
            division,
            department,
            class_name,
            clothes_t,
            clothes_desc
        ]

        # Append that one line to the CSV file without rewriting everything
        with open('data/assignment3_II.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(new_row_list)

        # Also stored it in the in-memory dict so it appears immediately
        reviews.setdefault(item_id, []).append({
            'title':     review_title,
            'text':      review_text,
            'age':       age,
            'rating':    rating,
            'predicted': final_label,
            'final':     final_label
        })

        return redirect(url_for('item_detail', item_id=item_id))

    #  render the form
    return render_template('new_review.html', item_id=item_id)
    
    
#  Predict Review Endpoint 
@app.route('/predict_review', methods=['POST'])
def predict_review():
    """
    Expects JSON payload: { "title": "...", "text": "..." }
    Returns JSON: { "predicted": 0 } or { "predicted": 1 }
    """
    data = request.get_json()
    if not data or 'title' not in data or 'text' not in data:
        return jsonify({"error": "Missing 'title' or 'text' in JSON"}), 400

    title = data['title'].strip()
    text  = data['text'].strip()

    # Combine title + text for prediction
    combined = f"{title} {text}"
    X_new    = vec.transform([combined])
    pred     = int(clf.predict(X_new)[0])

    return jsonify({"predicted": pred})

if __name__ == '__main__':
    app.run(debug=True)