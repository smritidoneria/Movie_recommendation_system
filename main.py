import streamlit as st
import pickle
import pandas as pd


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies=[];
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies


similarity = pickle.load(open('venv/similarity.pkl', 'rb'))
st.title('Movie Recommendation System')
movies_dict = pickle.load(open('venv/movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

selected_movie_name = st.selectbox('fg', movies['title'].values);
if st.button('recommend'):
    recommendations=recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)
