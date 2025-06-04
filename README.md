# Milestone II: Clothing Reviews Web App

**Student:** `Ronit Bhandari`  
**Student ID:** `s4109169`

---

##  Project Overview
A Flask-based web application that allows online shoppers to:
1. **Search** clothing items by keyword (using lemmatization to match different word forms).  
2. **View** item details and existing reviews.  
3. **Add** a new review: the backend model automatically suggests a binary “recommend” label (0 = No, 1 = Yes), which the user can override before saving.  
4. **Persist** new reviews by appending them to the existing CSV so they remain available after server restarts.

---

##  Features

- **Keyword Search**  
  - Case-insensitive, lemmatization-based matching on “Clothes Title” and “Clothes Description.”  
  - Typing “dress” will match “dresses,” “dressing,” etc., because the app lemmatizes both the query and item text.

- **Item Listing & Details**  
  - Home page displays every unique clothing item as a responsive card grid.  
  - Clicking a card opens an item detail page showing:  
    - Title, description, department, and all dataset reviews.  
    - New session-added reviews under a separate “New Reviews” section.

- **Add Review**  
  1. Buyer enters **Review Title**, **Age**, **Review Text**, and **Rating**.  
  2. As they type, the app calls the backend model to **auto-predict** a “Recommended” label (Yes/No).  
  3. The predicted value appears in an editable field (“Yes” or “No”). The user can modify it if desired.  
  4. On submit, the review (including final label) is appended to the CSV file and displayed immediately.

- **Persisted Storage**  
  - All reviews (existing + new) live in a single CSV (`assignment3_II.csv`).  
  - New reviews are appended via Python’s `csv` module so they survive server restarts.

---

##  Tech Stack & Dependencies

- **Flask** (web framework)  
- **Pandas** (CSV loading/manipulation)  
- **NLTK** (lemmatization with `WordNetLemmatizer`)  
- **scikit-learn** (loaded `vectorizer.pkl` & `model.pkl` for review classification)  
- **Python 3.8+**  

All Python dependencies are listed in `requirements.txt`.  
