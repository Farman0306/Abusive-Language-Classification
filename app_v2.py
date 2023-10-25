from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
stemmer = nltk.SnowballStemmer("english")
import string
nltk.download('stopwords')
stopword=set(stopwords.words('english'))
nltk.download('punkt')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/predict', methods=['POST'])
def predict():
    def clean_text(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text=" ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text=" ".join(text)
        return text

    tfidf_model_file = open('tfidf.pickle', 'rb')
    tfidf_model = pickle.load(tfidf_model_file)

    rf_model_file = open('rf_model.pickle', 'rb')
    classifier = pickle.load(rf_model_file)

    if request.method == 'POST':
        message = request.form['message']
        data = message
        c_data = clean_text(data)
        clean_data = [c_data]
        vect = tfidf_model.transform(clean_data).toarray()
        my_prediction = classifier.predict(vect)
        return str(my_prediction[0])  # Convert the prediction to a string

    return ""

if __name__ == '__main__':
    app.run(debug=True)
