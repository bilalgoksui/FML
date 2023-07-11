import pandas as pd 
import numpy as np 
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
import json
from flask_cors import CORS
import pickle
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ast
import nltk
import random
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('RAPIDAPI_KEY')
api_host = os.environ.get('RAPIDAPI_HOST')

# nltk.download('punkt')
# nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)
genre_emotion_mapping = {
    'action': ['joy', 'surprise'],
    'adventure': ['joy', 'surprise'],
    'animation': ['joy', 'surprise'],
    'comedy': ['joy'],
    'crime': ['fear', 'sadness'],
    'documentary': ['joy', 'surprise'],
    'drama': ['sadness', 'joy'],
    'family': ['joy', 'love'],
    'fantasy': ['joy', 'surprise'],
    'foreign': ['joy', 'surprise'],
    'history': ['sadness', 'surprise'],
    'horror': ['fear', 'surprise'],
    'music': ['joy', 'surprise'],
    'mystery': ['surprise', 'fear'],
    'romance': ['love', 'joy'],
    'science fiction': ['surprise', 'joy'],
    'tv movie': ['joy', 'surprise'],
    'thriller': ['fear', 'surprise'],
    'war': ['fear', 'sadness'],
    'western': ['joy', 'surprise']
}
STOPWORDS = set(stopwords.words('english'))

MAX_LEN = 35

emotions_to_labels = {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4,'surprise': 5}

labels_to_emotions = {j:i for i,j in emotions_to_labels.items()}


movies_data = pd.read_csv("data/movies_data_updated_without_specific_line.csv", low_memory=False)

list_of_all_titles = movies_data['title'].tolist()
list_of_all_indices = movies_data.index.tolist()
list_of_all_genres = movies_data['genres'].tolist()
list_of_all_overview = movies_data['overview'].tolist()
selected_features = ['genres', 'overview', 'tagline', 'title', 'imdb_id']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['overview'] + ' ' + movies_data['tagline'] + ' ' + \
                    movies_data['title'] + ' ' + movies_data['imdb_id']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)


similarity = cosine_similarity(feature_vectors)

def movie_poster(imdb_id):
    url = "https://moviesdatabase.p.rapidapi.com/titles/x/titles-by-ids"

    querystring = {"idsList": imdb_id}
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": api_host
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    data = json.loads(response.text)

    url = data['results'][0]['primaryImage']['url']
    return url
def suggest(movie_name, emotion,re_suggest):
    listm = []
    movie_name = movie_name.lower()
    movie_name_tokens = word_tokenize(movie_name)
    movie_name_tokens = [word for word in movie_name_tokens if word not in STOPWORDS]
    movie_name = ' '.join(movie_name_tokens)
    

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)


    if not find_close_match:
        emotion_type=emotion
        emotion_type = emotion_type.lower()
        emotion_type_tokens = word_tokenize(movie_name)
        emotion_type_tokens = [word for word in emotion_type_tokens if word not in STOPWORDS]
        emotion_type = ' '.join(emotion_type_tokens)
        find_close_match = difflib.get_close_matches(emotion_type, list_of_all_overview)


    if  not find_close_match:
        listm = randomrec()
        return listm
    if find_close_match:
        close_match = find_close_match[0]
        
        index_of_the_movie = list_of_all_indices[list_of_all_titles.index(close_match)]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)


        n=0
        if re_suggest==1:
            n=5   
   
        for movie in sorted_similar_movies[n:15]:
            index = movie[0]
            title_from_index = movies_data.loc[movies_data.index == index, 'title'].values
            title_from_index = list(title_from_index)
            imdb_id_from_index = movies_data.loc[movies_data.index == index, 'imdb_id'].values
            imdb_id_from_index = list(imdb_id_from_index)
            genre_from_index = movies_data.loc[movies_data.index == index, 'genres'].values

            poster_url = movie_poster(imdb_id_from_index[0])

            genres = get_genre_by_emotion(emotion)

            genre_list = ast.literal_eval(genre_from_index[0])
            genre_list = [genre.lower() for genre in genre_list] 

            genre_list = [genre for genre in genre_list if genre in genres]
            num_genre_matches = len(genre_list)

            listm.append({'movie_title': list_of_all_titles[index],
                        'imdb_id': movies_data.loc[index, 'imdb_id'],
                        'movie_image': poster_url,
                        'genres': genre_from_index[0],
                        'genre_matches': num_genre_matches})

        listm = sorted(listm, key=lambda x: x['genre_matches'], reverse=True)

        
        return listm[0:5]  

def randomrec():
    random_movie_list   = []
    for _ in range(5):
    
        random_index = random.randint(0, len(list_of_all_titles) - 1)
        print(random_index)
        imdb_id_from_index = movies_data.loc[movies_data.index == random_index, 'imdb_id'].values
        imdb_id_from_index = list(imdb_id_from_index)
        poster_url = movie_poster(imdb_id_from_index[0])
        random_movie_list.append({'movie_title': list_of_all_titles[random_index],
                    'imdb_id': movies_data.loc[random_index, 'imdb_id'],
                    'movie_image': poster_url
                    })

    return random_movie_list


def text_preprocess(text, stop_words=False):

  text = re.sub(r'\W+', ' ', text).lower()

  tokens = word_tokenize(text)

  if stop_words:
    tokens = [token for token in tokens if token not in STOPWORDS]

  return tokens

def predict_emotion(texts):
    # daha önceden eğittiğimiz lstm modelini yüklüyoruz
    model = tf.keras.models.load_model('\my_trained_model.h5',compile=True)


    with open('\loaded_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle) 

   
    texts_prepr = [text_preprocess(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(texts_prepr)
    pad = pad_sequences(sequences, maxlen=MAX_LEN)


    predictions = model.predict(pad)
    labels = np.argmax(predictions, axis=1)

    for i, lbl in enumerate(labels):
        print(f'\'{texts[i]}\' --> {labels_to_emotions[lbl]}')
    
    return labels_to_emotions[lbl]

def get_genre_by_emotion(emotion):
    genres = []
    for genre, emotions in genre_emotion_mapping.items():
        if emotion in emotions:
            genres.append(genre)
    return genres

@app.route('/suggest', methods=['POST'])
def index():
    req_data = request.get_json()
    movie_name = req_data.get('movie_name')
    emotion_text = req_data.get('emotion_text')
    re_suggest = req_data.get('re_suggest')
    suprise = req_data.get('supriseme')

    if suprise ==1:
        data = randomrec()
        return jsonify(data)
    elif suprise ==0:
        if emotion_text:
            predicted_emotion = predict_emotion([emotion_text]) 
            if movie_name:
                data = suggest(movie_name,predicted_emotion,re_suggest) 
                return jsonify(data)
            else:
                return jsonify(error='Invalid movie_name'), 400
    else:
        return jsonify(error='Error'), 400

if __name__ == '__main__':
    app.run(debug=True, port=5655)

