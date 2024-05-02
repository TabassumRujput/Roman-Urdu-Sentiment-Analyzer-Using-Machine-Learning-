from flask import Flask, render_template, request
import pickle
import nltk
import string
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re
import sklearn


app = Flask(__name__)

# Load the sentiment analysis model and  CountVectorizer vectorizer
with open('bnb.pkl', 'rb') as f:
    bnb = pickle.load(f)
with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:

        urdu_stopwords = ["kia", "me", "jis", "ap", "un", "o", "e", "he", "ha", "h", "k", "b", "mein", "aur", "ko",
                          "hai", "ki", "ka", "ke", "se", "ho", "to", "ne", "kar",
                          "kya", "ye", "hota", "hoti", "kiya", "hu", "hum", "tum", "aap", "wagera", "is",
                          "are", "am", "will", "the", "of", "and", "in", "for", "on", "with", "at", "by",
                          "an", "or", "but", "not", "that", "this", "it", "from", "to", "as", "was", "were", "be",
                          "been", 'ab', "being"
                                        "ai", "ayi", "hy", "hai", "main", "ki", "tha", "koi", "ko", "sy", "woh",
                          "bhi", "aur", "wo", "yeh", "rha", "hota", "ho", "ga", "ka", "le", "lye",
                          "kr", "kar", "lye", "liye", "hotay", "waisay", "gya", "gaya", "kch", "ab",
                          "thy", "thay", "houn", "hain", "han", "to", "is", "hi", "jo", "kya", "thi",
                          "se", "pe", "phr", "wala", "waisay", "us", "na", "ny", "hun", "rha", "raha",
                          "ja", "rahay", "abi", "uski", "ne", "haan", "acha", "nai", "sent", "photo",
                          "you", "kafi", "gai", "rhy", "kuch", "jata", "aye", "ya", "dono", "hoa",
                          "aese", "de", "wohi", "jati", "jb", "krta", "lg", "rahi", "hui", "karna",
                          "krna", "gi", "hova", "yehi", "jana", "jye", "chal", "mil", "tu", "hum",
                          "par", "hay", "kis", "sb", "gy", "dain", "krny", "tou"
                          ]

        if i not in urdu_stopwords and i not in string.punctuation:
            y.append(i)

    return " ".join(y)


import re


def remove_emojis(text):
    # Define a regular expression pattern to match emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    # Remove emojis from the text using the pattern
    return emoji_pattern.sub(r'', text)


# Output: 'I love Python! '


@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Preprocess the comment
        preprocessed_comment = transform_text(comment)
        preprocessed_comment = remove_emojis(preprocessed_comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = cv.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = bnb.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)