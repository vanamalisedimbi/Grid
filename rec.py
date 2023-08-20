import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def read_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def calculate_user_similarity(user_profiles):
    user_similarity = cosine_similarity(user_profiles)
    user_similarity_df = pd.DataFrame(user_similarity, columns=user_profiles.index, index=user_profiles.index)
    return user_similarity_df

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

def main():
    training_file_path = input("Enter the path of the training dataset file: ")
    training_df = read_dataset(training_file_path)

    user_profiles = pd.get_dummies(training_df['Product']).groupby(training_df['User']).mean()
    user_similarity_df = calculate_user_similarity(user_profiles)

    interactions = training_df[['User', 'Product']]
    unique_users = training_df['User'].unique()

    personalized_rankings = {}
    for user in unique_users:
        recommended_products_list = recommend_products(user, user_similarity_df, interactions)
        personalized_rankings[user] = recommended_products_list

    test_file_path = input("Enter the path of the test dataset file: ")
    test_df = read_dataset(test_file_path)

    accuracy_scores = []
    for user in unique_users:
        if user in test_df['User'].unique():
            actual_interactions = test_df[test_df['User'] == user]['Product'].tolist()
            recommended_interactions = personalized_rankings[user]

            correct_recommendations = set(actual_interactions) & set(recommended_interactions)
            accuracy = len(correct_recommendations) / len(actual_interactions) * 100
            accuracy_scores.append(accuracy)
            print(f"Accuracy for {user}: {accuracy:.2f}%")

    overall_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

if __name__ == "__main__":
    main()
