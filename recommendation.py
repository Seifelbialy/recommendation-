
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.user_item_matrix = np.zeros((num_users, num_items))
    
    def fit(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
    
    def predict(self, user_id, top_n=5):
        similarities = cosine_similarity(self.user_item_matrix, self.user_item_matrix)
        user_similarities = similarities[user_id]
        ranked_users = np.argsort(user_similarities)[::-1]  # Sort users by similarity in descending order
        user_ratings = self.user_item_matrix[user_id]
        recommendations = []
        for other_user_id in ranked_users:
            if other_user_id == user_id:
                continue
            other_user_rating = self.user_item_matrix[other_user_id]
            relevant_items = np.where(other_user_rating > 0)[0]  # Get items rated by the other user
            for item_id in relevant_items:
                if user_ratings[item_id] == 0:  # User hasn't rated this item
                    recommendations.append((item_id, user_similarities[other_user_id]))
            if len(recommendations) >= top_n:
                break
        recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort recommendations by similarity
        return [item_id for item_id, _ in recommendations[:top_n]]


# Example usage
user_item_matrix = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0]
])

cf = CollaborativeFiltering(num_users=5, num_items=5)
cf.fit(user_item_matrix)

user_id = 0  # ID of the user for whom we want to make recommendations
recommendations = cf.predict(user_id)
print("Recommendations for user", user_id, ":", recommendations)
