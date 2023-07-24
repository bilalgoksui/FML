# FML
Movie Recommendation App
This app gives users movie recommendations through emotional content analysis. The user enters the emotion content, and the system recommends movies that match that emotion.

## How to use

1.Launch the application:

```
python app.py
```
2.After the application starts, send a POST request using an API client (Postman etc.):
```
POST http://localhost:5655/suggest
```

Example JSON request body:
```
{
    "movie_name": "Inception",
    "emotion_text": "I feel excited and curious.",
    "re_suggest": 0,
    "supriseme": 0
}
```

#### movie_name: The movie name on which to base recommendations.
#### emotion_text: Text that expresses the emotion the user is feeling.
#### re_suggest: A flag indicating whether to resuggest or not. If it is 1, the previous movie name is re-suggested.
#### supriseme: If 1, a surprise suggestion is made.

3. Get the results. The application will return a list of movies in JSON format.
   
## Requirements
Python 3.6 or above
Flask
Pandas
Numpy
Scikit-learn
NLTK
TensorFlow
Flask-CORS
dotenv
requests

## License
This project is distributed under the MIT license. See the LICENSE file for more information.
[MIT](https://choosealicense.com/licenses/mit/)
