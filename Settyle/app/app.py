from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import random
import os

# initialize the Flask app
app = Flask(__name__)

# load the necessary objects
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

df_full = pd.read_pickle('model/df_full.pkl')
df = pd.read_pickle('model/df.pkl')
user_item_matrix = pd.read_pickle('model/user_item_matrix.pkl')
item_user_matrix = pd.read_pickle('model/item_user_matrix.pkl')


# to give it to homepahe
def get_top_products(df, n=10):
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    
    top_products = (df.groupby('product_id')['rating']
                    .mean()
                    .sort_values(ascending=False)
                    .head(2 * n)  
                    .index.tolist())
    
    randomized_top_products = random.sample(top_products, k=n)

    return df[df['product_id'].isin(randomized_top_products)][['product_id', 'product_name', 'img_link']].drop_duplicates()



# route for the home page
@app.route('/')
def index():
    top_products = get_top_products(df)
    return render_template('index.html', products=top_products.to_dict(orient='records'))

# route for product recommendations
@app.route('/recommend', methods=['POST'])
@app.route('/recommend', methods=['POST'])
def recommend():
    product_id = request.form.get('product_id')

    if not product_id:
        return "Error: No product ID provided.", 400

    product_ratings = item_user_matrix.loc[product_id]
    product_unrated_users = product_ratings[product_ratings == 0]
    product_unrated_users_id = product_unrated_users.index.tolist()
    predicted_ratings = [model.predict(user_id, product_id).est for user_id in product_unrated_users_id]
    top_indices = np.argsort(predicted_ratings)[::-1][:3]
    top_user_ids = [product_unrated_users_id[i] for i in top_indices]
    top_user_ids = df_full[df_full['user_id'].isin(top_user_ids)]['user_id'].unique().tolist()

    unique_product_ids = set()
    all_recommendations = []

    for user in top_user_ids:
        user_ratings = user_item_matrix.loc[user]
        user_unrated_products = user_ratings[user_ratings == 0]
        user_unrated_products_id = user_unrated_products.index.tolist()
        predicted_ratings = [model.predict(user, product_id).est for product_id in user_unrated_products_id]
        top_indices = np.argsort(predicted_ratings)[::-1][:5]
        top_product_ids = [user_unrated_products_id[i] for i in top_indices]
        
        for prod_id in top_product_ids:
            if prod_id not in unique_product_ids:
                unique_product_ids.add(prod_id)
                top_product_info = df_full[df_full['product_id'] == prod_id][['product_id', 'product_name', 'img_link']].iloc[0]
                all_recommendations.append(top_product_info.fillna({'img_link': 'default_image.jpg'}).to_dict())

        if len(unique_product_ids) >= 5:
            break

    if not all_recommendations:
        return "No recommendations available."

    return render_template('recommendations.html', recommendations=all_recommendations)


if __name__ == '__main__':
    app.run(debug=True)



