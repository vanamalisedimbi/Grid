import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Read dataset from text file
def read_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Calculate user similarity using cosine similarity
def calculate_user_similarity(user_profiles):
    user_similarity = cosine_similarity(user_profiles)
    user_similarity_df = pd.DataFrame(user_similarity, columns=user_profiles.index, index=user_profiles.index)
    return user_similarity_df

# Recommend products based on similar users
def recommend_products(active_user, user_similarity_df, interactions, num_recommendations=3):
    similar_users = user_similarity_df[active_user].sort_values(ascending=False)[1:num_recommendations+1].index
    recommended_products = {}

    for user in similar_users:
        products_interacted = interactions[interactions['User'] == user]['Product'].tolist()
        for product in products_interacted:
            if product not in interactions[interactions['User'] == active_user]['Product'].tolist():
                if product not in recommended_products:
                    recommended_products[product] = user_similarity_df.loc[user, active_user]
                else:
                    recommended_products[product] += user_similarity_df.loc[user, active_user]

    sorted_recommendations = sorted(recommended_products.items(), key=lambda x: x[1], reverse=True)
    return [product for product, _ in sorted_recommendations]

# Main function
def main():
    file_path = 'example_data.txt'
    df = read_dataset(file_path)

    user_profiles = pd.get_dummies(df['Product']).groupby(df['User']).mean()
    user_similarity_df = calculate_user_similarity(user_profiles)

    interactions = df[['User', 'Product']]
    unique_users = df['User'].unique()

    personalized_rankings = {}
    for user in unique_users:
        recommended_products_list = recommend_products(user, user_similarity_df, interactions)
        personalized_rankings[user] = recommended_products_list

    for user, rankings in personalized_rankings.items():
        print(f"{user}'s Personalized Rankings: {rankings}")

if _name_ == "_main_":
    main()