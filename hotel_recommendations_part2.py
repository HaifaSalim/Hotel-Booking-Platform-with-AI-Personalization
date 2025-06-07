import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import functools
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import random
from collections import Counter

def load_processed_data():
    """Load all processed data"""
 
    if not os.path.exists('processed_data'):
        raise FileNotFoundError("Processed data not found")
    
    user_data = pd.read_pickle('processed_data/user_data.pkl')
    hotel_data = pd.read_pickle('processed_data/hotel_data.pkl')
    room_data = pd.read_pickle('processed_data/room_data.pkl')
    
    
    user_preferences = pd.read_pickle('processed_data/user_preferences.pkl')
    hotel_features = pd.read_pickle('processed_data/hotel_features.pkl')
    
    user_similarity = pd.read_pickle('processed_data/user_similarity.pkl')
    content_similarity = pd.read_pickle('processed_data/content_similarity.pkl')
    
    user_hotel_matrix = pd.read_pickle('processed_data/user_hotel_matrix.pkl')
    
    user_factors = None
    hotel_factors = None
    if os.path.exists('processed_data/user_factors.npy'):
        user_factors = np.load('processed_data/user_factors.npy')
    
    if os.path.exists('processed_data/hotel_factors.npy'):
        hotel_factors = np.load('processed_data/hotel_factors.npy')
    
    return (user_data, hotel_data, room_data, user_preferences, hotel_features, 
            user_similarity, user_hotel_matrix, content_similarity, user_factors, hotel_factors)

def get_cluster_recommendations(centroids, hotel_features, feature_names, n_recommendations=5):
    """Get top hotel recommendations for cluster centroid"""
    cluster_recommendations = {}
    
  
    for cluster_id, centroid in enumerate(centroids):
        hotel_scores = {}
        
       
        centroid_df = pd.DataFrame([centroid], columns=feature_names)
       
        hotel_subset = hotel_features[feature_names].fillna(0)
        

        for hotel_id, features in hotel_subset.iterrows():
            similarity = cosine_similarity(
                centroid_df.values.reshape(1, -1),
                features.values.reshape(1, -1)
            )[0][0]
            
            hotel_scores[hotel_id] = similarity
      
        sorted_hotels = sorted(hotel_scores.items(), key=lambda x: x[1], reverse=True)
        top_hotels = sorted_hotels[:n_recommendations]
        
        cluster_recommendations[cluster_id] = [hotel_id for hotel_id, _ in top_hotels]
    
    return cluster_recommendations

def collaborative_filtering(user_data):
   
    if 'rating' not in user_data.columns or 'hotel_id' not in user_data.columns:
        return pd.DataFrame(), pd.DataFrame(), None, None
    
    
    user_hotel_matrix = user_data.pivot_table(
        index='user_id', 
        columns='hotel_id', 
        values='rating',
        fill_value=0
    )
 
    if user_hotel_matrix.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None
    
   
    user_hotel_sparse = csr_matrix(user_hotel_matrix.values)
    user_similarity = cosine_similarity(user_hotel_sparse)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_hotel_matrix.index,
        columns=user_hotel_matrix.index
    )
    
    predicted_ratings, user_factors, hotel_factors = matrix_factorization(user_hotel_matrix)
    
    return user_similarity_df, user_hotel_matrix, user_factors, hotel_factors


def content_based_filtering(user_data, hotel_data, room_data):
   
    user_preferences = extract_user_preferences(user_data)
    
    hotel_features = extract_hotel_features(hotel_data, room_data)
    
    feature_mapping = {
        'user_pool_preference': ['has_pool'],
        'user_spa_preference': ['has_spa'],
        'prefers_family_travel': ['has_family_rooms'],
        'family_size': ['room_occupancy_max'],
        'prefers_hotel_Resort Hotel': ['is_resort'],
        'prefers_hotel_City Hotel': ['is_resort'],  
        'avg_price_per_night': ['price', 'room_price_mean']
    }
    
    similarity_array = np.zeros((len(user_preferences), len(hotel_features)))
    

    for i, user_id in enumerate(user_preferences.index):
        user_prefs = user_preferences.loc[user_id]
        
        for j, hotel_id in enumerate(hotel_features.index):
            hotel_feats = hotel_features.loc[hotel_id]
            
            feature_scores = []
            feature_weights = []
            
            for pref, pref_value in user_prefs.items():
                if pref in feature_mapping and feature_mapping[pref]:
             
                    weight = 1.5 if pref in ['user_pool_preference', 'user_spa_preference'] else 1.0
                    
                    for hotel_feat in feature_mapping[pref]:
                        if hotel_feat in hotel_feats.index:
                            hotel_value = hotel_feats[hotel_feat]
                            
                            if pd.notna(pref_value) and pd.notna(hotel_value):
                                if pref == 'avg_price_per_night' and hotel_feat in ['price', 'room_price_mean']:
                                    if pref_value > 0 and hotel_value <= pref_value * 1.2:
                                        price_similarity = 1 - min(abs(pref_value - hotel_value) / pref_value, 1)
                                        feature_scores.append(price_similarity * weight)
                                        feature_weights.append(weight)
                                elif pref == 'prefers_hotel_City Hotel' and hotel_feat == 'is_resort':
                                    feature_scores.append((1 - hotel_value) * pref_value * weight)
                                    feature_weights.append(weight)
                                elif pref_value > 0 and hotel_value > 0:
                                    feature_scores.append(min(pref_value, hotel_value) * weight)
                                    feature_weights.append(weight)
            
            if feature_weights:
                similarity_array[i, j] = sum(feature_scores) / sum(feature_weights)
    
 
    similarity_matrix = pd.DataFrame(
        similarity_array,
        index=user_preferences.index,
        columns=hotel_features.index
    )
    
    return similarity_matrix, user_preferences, hotel_features

def matrix_factorization(user_hotel_matrix, n_factors=20, n_iterations=20, learning_rate=0.01, regularization=0.02):
    """
   matrix factorization using Singular Value Decomposition
    for collaborative filtering
    """

    sparse_matrix = csr_matrix(user_hotel_matrix.values)
    
  
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(sparse_matrix)
    
    hotel_factors = svd.components_.T
 
    predicted_ratings = np.dot(user_factors, hotel_factors.T)
    

    predicted_ratings = np.clip(predicted_ratings, 0, 5)
    
    
    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_hotel_matrix.index,
        columns=user_hotel_matrix.columns
    )
    
    return predicted_df, user_factors, hotel_factors

def collaborative_filtering(user_data):
   
    if 'rating' not in user_data.columns or 'hotel_id' not in user_data.columns:
        return pd.DataFrame(), pd.DataFrame(), None, None
    
 
    user_hotel_matrix = user_data.pivot_table(
        index='user_id', 
        columns='hotel_id', 
        values='rating',
        fill_value=0
    )
    
    if user_hotel_matrix.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None
    user_hotel_sparse = csr_matrix(user_hotel_matrix.values)
    user_similarity = cosine_similarity(user_hotel_sparse)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_hotel_matrix.index,
        columns=user_hotel_matrix.index
    )
    
    predicted_ratings, user_factors, hotel_factors = matrix_factorization(user_hotel_matrix)
    
    return user_similarity_df, user_hotel_matrix, user_factors, hotel_factors

def extract_user_preferences(user_data):
   
    user_ids = user_data['user_id'].unique()
    user_preferences = pd.DataFrame(index=user_ids)
    
  
    numerical_cols = [col for col in [
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
        'booking_changes', 'days_in_waiting_list', 'adr', 'total_of_special_requests',
        'total_nights', 'booking_spending'
    ] if col in user_data.columns]
    
    categorical_cols = [col for col in [
        'meal', 'reserved_room_type', 'assigned_room_type', 'hotel_type',
        'is_repeated_guest', 'is_canceled'
    ] if col in user_data.columns]
    
   
    agg_functions = {}
    for col in numerical_cols:
        agg_functions[col] = ['sum', 'mean', 'max']
    
    for col in categorical_cols:
        agg_functions[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else None
    
    if 'rating' in user_data.columns:
        agg_functions['rating'] = ['mean', 'min', 'max', 'count']
    

    if agg_functions:
     
        user_agg = user_data.groupby('user_id').agg(agg_functions)
        
    
        user_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in user_agg.columns]
        

        user_preferences = user_preferences.join(user_agg)
    
    explicit_prefs = [col for col in [
        'user_prefers_weekend', 'user_prefers_long_stay', 
        'user_pool_preference', 'user_spa_preference'
    ] if col in user_data.columns]
    
    if explicit_prefs:
        
        explicit_agg = user_data.groupby('user_id').agg({
            pref: lambda x: x.mode().iloc[0] if not x.mode().empty else False 
            for pref in explicit_prefs
        })
        
    
        for pref in explicit_prefs:
            explicit_agg[pref] = explicit_agg[pref].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})
        
        
        user_preferences = user_preferences.join(explicit_agg)
    
  
    weekend_cols = ['stays_in_weekend_nights_sum', 'stays_in_week_nights_sum']
    if all(col in user_preferences.columns for col in weekend_cols):
        total_nights = user_preferences['stays_in_weekend_nights_sum'] + user_preferences['stays_in_week_nights_sum']
        user_preferences['weekend_ratio'] = user_preferences['stays_in_weekend_nights_sum'] / total_nights.replace(0, 1)
        if 'user_prefers_weekend' not in user_preferences.columns:
            user_preferences['user_prefers_weekend'] = (user_preferences['weekend_ratio'] > 0.4).astype(int)
   
    if 'total_nights_mean' in user_preferences.columns:
        if 'user_prefers_long_stay' not in user_preferences.columns:
            user_preferences['user_prefers_long_stay'] = (user_preferences['total_nights_mean'] > 3).astype(int)
    
    if 'total_of_special_requests_mean' in user_preferences.columns:
        user_preferences['prefers_special_requests'] = (user_preferences['total_of_special_requests_mean'] > 1).astype(int)
    
    family_cols = ['adults_mean', 'children_mean']
    if all(col in user_preferences.columns for col in family_cols):
        user_preferences['family_size'] = user_preferences['adults_mean'] + user_preferences['children_mean']
        user_preferences['prefers_family_travel'] = (user_preferences['children_mean'] > 0).astype(int)
    
   
    if 'reserved_room_type' in user_preferences.columns:
        room_type_dummies = pd.get_dummies(user_preferences['reserved_room_type'], prefix='prefers_room')
        user_preferences = pd.concat([user_preferences, room_type_dummies], axis=1)
    
     
    if 'hotel_type' in user_preferences.columns:
        hotel_type_dummies = pd.get_dummies(user_preferences['hotel_type'], prefix='prefers_hotel')
        user_preferences = pd.concat([user_preferences, hotel_type_dummies], axis=1)
    
    
    price_cols = ['booking_spending_sum', 'total_nights_sum']
    if all(col in user_preferences.columns for col in price_cols):
        user_preferences['avg_price_per_night'] = user_preferences['booking_spending_sum'] / user_preferences['total_nights_sum'].replace(0, 1)
    elif 'adr_mean' in user_preferences.columns:
        user_preferences['avg_price_per_night'] = user_preferences['adr_mean']
    
    numerical_cols = user_preferences.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        scaler = MinMaxScaler()
        user_preferences[numerical_cols] = scaler.fit_transform(user_preferences[numerical_cols].fillna(0))
    
    return user_preferences

def extract_hotel_features(hotel_data, room_data):
   
    hotel_features = pd.DataFrame(index=hotel_data['hotel_id'])
     
    if 'price_per_night' in hotel_data.columns:
        hotel_features['price'] = hotel_data['price_per_night']
    
    if 'rating' in hotel_data.columns:
        hotel_features['rating_numeric'] = hotel_data['rating'].map({
            'Five': 5, 'Four': 4, 'Three': 3, 'Two': 2, 'One': 1,
            'FiveStar': 5, 'FourStar': 4, 'ThreeStar': 3, 'TwoStar': 2, 'OneStar': 1
        })

    if 'description' in hotel_data.columns:
        hotel_features['is_resort'] = hotel_data['description'].str.contains('resort', case=False, na=False).astype(int)
    
   
    if 'hotel_facilities' in hotel_data.columns:
        facilities = hotel_data['hotel_facilities']
        hotel_features['has_pool'] = facilities.str.contains('pool', case=False, na=False).astype(int)
        hotel_features['has_spa'] = facilities.str.contains('spa', case=False, na=False).astype(int)
        hotel_features['has_gym'] = facilities.str.contains('gym|fitness', case=False, na=False).astype(int)
        hotel_features['has_restaurant'] = facilities.str.contains('restaurant|dining', case=False, na=False).astype(int)
        hotel_features['has_wifi'] = facilities.str.contains('wifi|internet', case=False, na=False).astype(int)
        hotel_features['has_bar'] = facilities.str.contains('bar|lounge', case=False, na=False).astype(int)
        hotel_features['has_beach_access'] = facilities.str.contains('beach', case=False, na=False).astype(int)
        hotel_features['has_business_facilities'] = facilities.str.contains('business|conference|meeting', case=False, na=False).astype(int)
    
    if 'city_name' in hotel_data.columns:
        location_dummies = pd.get_dummies(hotel_data['city_name'], prefix='loc')
        hotel_features = pd.concat([hotel_features, location_dummies], axis=1)
    
     
    if 'attractions' in hotel_data.columns:
        hotel_features['attraction_count'] = hotel_data['attractions'].str.count(',') + 1
    
    if not room_data.empty and 'hotel_id' in room_data.columns:
    
        agg_dict = {
            'price': ['mean', 'min', 'max'],
            'occupancy': ['mean', 'max']
        }
        
        room_agg = room_data.groupby('hotel_id').agg(agg_dict)
   
        room_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in room_agg.columns]
       
        hotel_features = hotel_features.join(room_agg, how='left')
        
       
        for col in room_agg.columns:
            hotel_features.rename(columns={col: f'room_{col}'}, inplace=True)
     
        room_types_per_hotel = room_data.groupby('hotel_id')['room_type'].nunique()
        hotel_features = hotel_features.join(room_types_per_hotel.rename('room_type_variety'), how='left')
       
        has_family_room = room_data.groupby('hotel_id')['occupancy'].max() >= 4
        hotel_features = hotel_features.join(has_family_room.rename('has_family_rooms').astype(int), how='left')
        
        if 'amenities' in room_data.columns:
            luxury_keywords = ['suite', 'luxury', 'premium', 'deluxe', 'executive']
    
            luxury_mask = room_data['amenities'].str.lower().str.contains('|'.join(luxury_keywords), na=False)
            has_luxury = room_data[luxury_mask].groupby('hotel_id').size() > 0
           
            hotel_features = hotel_features.join(has_luxury.rename('has_luxury_rooms').astype(int), how='left')
    
   
    if 'rating_numeric' in hotel_features.columns:
        hotel_features['popularity_score'] = hotel_features['rating_numeric']
        
       
        if 'review_count' in hotel_data.columns:
       
            review_count_normalized = MinMaxScaler().fit_transform(hotel_data[['review_count']]).flatten()
            hotel_features['review_count_normalized'] = review_count_normalized
            hotel_features['popularity_score'] = 0.7 * hotel_features['rating_numeric'] + 0.3 * hotel_features['review_count_normalized']
    
    numerical_cols = hotel_features.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        scaler = MinMaxScaler()
        hotel_features[numerical_cols] = scaler.fit_transform(hotel_features[numerical_cols].fillna(0))
    
    return hotel_features


def create_hotel_clusters(hotel_features, n_clusters=5):
    """Create hotel clusters based on features for recommendations"""

    numerical_cols = hotel_features.select_dtypes(include=[np.number]).columns
    
    
    if numerical_cols.empty:
        return None, None, None
    

    cluster_data = hotel_features[numerical_cols].fillna(0)
 
    if len(cluster_data) < n_clusters:
        return None, None, None
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(cluster_data)
    
    hotel_clusters = pd.Series(clusters, index=hotel_features.index)
    
    centroids = kmeans.cluster_centers_
    
    return hotel_clusters, centroids, numerical_cols

def calculate_hotel_popularity(hotel_data, user_data):
    """Calculating popularity score for hotels """
    
    rating_map = {'Five': 5, 'Four': 4, 'Three': 3, 'Two': 2, 'One': 1,
                 'FiveStar': 5, 'FourStar': 4, 'ThreeStar': 3, 'TwoStar': 2, 'OneStar': 1}
    
    if 'hotel_id' in user_data.columns:
        hotel_bookings = user_data['hotel_id'].value_counts().to_dict()
    else:
        hotel_bookings = {}
    
   
    hotel_ratings = {}
    if 'rating' in hotel_data.columns:
        for _, row in hotel_data.iterrows():
            if 'hotel_id' in row and 'rating' in row:
                rating_str = row['rating']
                if isinstance(rating_str, str) and rating_str in rating_map:
                    hotel_ratings[row['hotel_id']] = rating_map[rating_str]
                elif pd.notna(rating_str):
                    hotel_ratings[row['hotel_id']] = float(rating_str)
    
 
    popularity_scores = {}
    for hotel_id in hotel_data['hotel_id']:
        
        booking_count = hotel_bookings.get(hotel_id, 0)
        max_bookings = max(hotel_bookings.values()) if hotel_bookings else 1
        normalized_bookings = booking_count / max_bookings if max_bookings > 0 else 0
  
        rating = hotel_ratings.get(hotel_id, 0)
        normalized_rating = rating / 5.0
        
        
        popularity_scores[hotel_id] = (0.7 * normalized_rating) + (0.3 * normalized_bookings)
    
    return popularity_scores

def build_trending_recommendations(hotel_data, user_data, location_filter=None, n_recommendations=10):
    """Build recommendations for trending hotels based on popularity and recency"""
    
    popularity_scores = calculate_hotel_popularity(hotel_data, user_data)
    
    
    if 'date' in user_data.columns:
        try:
            user_data['date'] = pd.to_datetime(user_data['date'])
            
        
            latest_bookings = user_data.groupby('hotel_id')['date'].max()
            
            most_recent_date = user_data['date'].max()
            days_since_booking = (most_recent_date - latest_bookings).dt.days
            
           
            max_days = days_since_booking.max() if not days_since_booking.empty else 1
            recency_scores = 1 - (days_since_booking / max_days) if max_days > 0 else 0
           
            for hotel_id in popularity_scores:
                if hotel_id in recency_scores:
                    popularity_scores[hotel_id] = (0.6 * popularity_scores[hotel_id]) + (0.4 * recency_scores[hotel_id])
        except:
            
            pass
    
    if location_filter:
        filtered_hotel_ids = []
        for hotel_id in popularity_scores.keys():
            hotel = hotel_data[hotel_data['hotel_id'] == hotel_id]
            if not hotel.empty:
                location_matches = False
                
                if 'city_name' in hotel.columns:
                    city_name = hotel['city_name'].iloc[0]
                    if pd.notna(city_name) and location_filter.lower() in str(city_name).lower():
                        location_matches = True
               
                if 'location' in hotel.columns and not location_matches:
                    location = hotel['location'].iloc[0]
                    if pd.notna(location) and location_filter.lower() in str(location).lower():
                        location_matches = True
                
                if location_matches:
                    filtered_hotel_ids.append(hotel_id)
        
        if filtered_hotel_ids:
            filtered_scores = {hotel_id: score for hotel_id, score in popularity_scores.items() 
                              if hotel_id in filtered_hotel_ids}
            popularity_scores = filtered_scores
    
    sorted_hotels = sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)
    top_trending = sorted_hotels[:n_recommendations]
    
    return [hotel_id for hotel_id, _ in top_trending]
    
def get_new_visitor_recommendations(new_visitor_info, user_data, hotel_data, room_data, n_recommendations=5,
                                   hotel_features=None, hotel_clusters=None, centroids=None, feature_names=None):
    """Get recommendations for new visitors without user IDs"""

    location_filter = new_visitor_info.get("location", None)
    
    
    if hotel_features is None:
        hotel_features = extract_hotel_features(hotel_data, room_data)
    
    if hotel_clusters is None or centroids is None or feature_names is None:
        hotel_clusters, centroids, feature_names = create_hotel_clusters(hotel_features)
    
    cluster_recommendations = {}
    if hotel_clusters is not None and centroids is not None:
        cluster_recommendations = get_cluster_recommendations(centroids, hotel_features, feature_names)
    
    trending_hotels = build_trending_recommendations(hotel_data, user_data, location_filter)
    
   
    if not new_visitor_info:
        return [get_hotel_details(hotel_id, hotel_data) for hotel_id in trending_hotels[:n_recommendations]]
    
    best_cluster = None
    
    visitor_prefs = {}
    
    if 'hotel_type' in new_visitor_info:
        visitor_prefs['hotel_type'] = new_visitor_info['hotel_type']
        
        if visitor_prefs['hotel_type'] == 'Resort Hotel':
            visitor_prefs['prefers_hotel_Resort Hotel'] = 1
            visitor_prefs['prefers_hotel_City Hotel'] = 0
        elif visitor_prefs['hotel_type'] == 'City Hotel':
            visitor_prefs['prefers_hotel_Resort Hotel'] = 0
            visitor_prefs['prefers_hotel_City Hotel'] = 1
    
    if 'amenities' in new_visitor_info:
        amenities = new_visitor_info['amenities']
        visitor_prefs['user_pool_preference'] = 1 if 'pool' in amenities else 0
        visitor_prefs['user_spa_preference'] = 1 if 'spa' in amenities else 0
    
    if 'price_range' in new_visitor_info:
        price_range = new_visitor_info['price_range']
        if isinstance(price_range, dict) and 'max' in price_range:
            visitor_prefs['avg_price_per_night'] = price_range['max']
        elif isinstance(price_range, (int, float)):
            visitor_prefs['avg_price_per_night'] = price_range
    
    if 'family_size' in new_visitor_info:
        visitor_prefs['family_size'] = new_visitor_info['family_size']
        visitor_prefs['prefers_family_travel'] = 1 if visitor_prefs['family_size'] > 1 else 0
    
    
    if hotel_clusters is not None and centroids is not None:
        
        visitor_features = []
        visitor_feature_names = []
        
      
        centroid_features = hotel_features.select_dtypes(include=[np.number]).columns
        
        for feature in centroid_features:
            if feature in visitor_prefs:
                visitor_features.append(visitor_prefs[feature])
                visitor_feature_names.append(feature)
        

        if visitor_features:
            visitor_vector = np.array(visitor_features).reshape(1, -1)
           
            centroid_subset = []
            for centroid in centroids:
                subset = []
                for feature in visitor_feature_names:
                    idx = list(centroid_features).index(feature)
                    subset.append(centroid[idx])
                centroid_subset.append(subset)
        
            distances = []
            for centroid in centroid_subset:
                centroid_vector = np.array(centroid).reshape(1, -1)
                distance = np.linalg.norm(visitor_vector - centroid_vector)
                distances.append(distance)
            
            best_cluster = np.argmin(distances)
    
    if best_cluster is not None and best_cluster in cluster_recommendations:
        recommended_hotels = cluster_recommendations[best_cluster]
    else:
    
        recommended_hotels = trending_hotels
    
    top_recommendations = recommended_hotels[:n_recommendations]
    
    detailed_recommendations = [get_hotel_details(hotel_id, hotel_data) for hotel_id in top_recommendations]
    
   
    if location_filter:
       
        filtered_recommendations = [
            hotel for hotel in detailed_recommendations
            if hotel.get('location', '').lower() == location_filter.lower()
            or location_filter.lower() in hotel.get('location', '').lower()
            or location_filter.lower() in hotel.get('city_name', '').lower()
        ]

       
        if filtered_recommendations:
            detailed_recommendations = filtered_recommendations
    
    return detailed_recommendations
def get_hotel_details(hotel_id, hotel_data):
    """Get information for a hotel"""
    hotel_row = hotel_data[hotel_data['hotel_id'] == hotel_id]
    
    if hotel_row.empty:
        return {'hotel_id': hotel_id, 'name': f'Hotel {hotel_id}', 'details': 'Not available'}
  
    rating = hotel_row['rating'].iloc[0] if 'rating' in hotel_row.columns else None
    if isinstance(rating, str):
        
        rating = rating.replace('Star', '').replace('star', '').strip()
        if rating in ['Five', 'Four', 'Three', 'Two', 'One']:
            rating = rating + ' Star'
    
    

    hotel_details = {
        'hotel_id': hotel_id,
        'id': hotel_id, 
        'name': hotel_row['name'].iloc[0] if 'name' in hotel_row.columns else f'Hotel {hotel_id}',
        'location': hotel_row['city_name'].iloc[0] if 'city_name' in hotel_row.columns else 'Unknown location',
        'price_per_night': hotel_row['price_per_night'].iloc[0] if 'price_per_night' in hotel_row.columns else 0,
        'rating': hotel_row['rating'].iloc[0] if 'rating' in hotel_row.columns else 'Not rated',
        'imageUrl': hotel_row['image_url'].iloc[0] if 'image_url' in hotel_row.columns else 'images/default_hotel.jpg'
    }
    
    detail_columns = ['description']
    for col in detail_columns:
        if col in hotel_row.columns:
            hotel_details[col] = hotel_row[col].iloc[0]
   
    if 'hotel_facilities' in hotel_row.columns:
        facilities = hotel_row['hotel_facilities'].iloc[0]
        if isinstance(facilities, str):
            hotel_details['facilities'] = [facility.strip() for facility in facilities.split(',')]
   
    if 'attractions' in hotel_row.columns:
        attractions = hotel_row['attractions'].iloc[0]
        if isinstance(attractions, str):
            hotel_details['nearby_attractions'] = [attraction.strip() for attraction in attractions.split(',')]
    
    return hotel_details
def get_personalized_recommendations(user_id, user_data, hotel_data, room_data,
                                   user_similarity=None, user_hotel_matrix=None,
                                   content_similarity=None, user_factors=None,
                                   hotel_factors=None, location_filter=None, n_recommendations=5):
    """Get personalized hotel recommendations for a specific user with reasons."""
    
    if user_id not in user_data['user_id'].values:
        return []
    
   
    user_info = user_data[user_data['user_id'] == user_id]
    
 
    if user_similarity is None or user_hotel_matrix is None or user_factors is None or hotel_factors is None:
        user_similarity, user_hotel_matrix, user_factors, hotel_factors = collaborative_filtering(user_data)
    
    if content_similarity is None:
        content_similarity, user_preferences, hotel_features = content_based_filtering(user_data, hotel_data, room_data)
    
    recommended_hotels = []
    recommendation_reasons = {} 

    if user_id in user_similarity.index and not user_similarity.empty:
        
        similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:6] 
        
        similar_user_hotels = []
        
        for similar_user_id, similarity in similar_users.items():
            if similarity > 0.2 and similar_user_id in user_hotel_matrix.index:
           
                user_ratings = user_hotel_matrix.loc[similar_user_id]
                rated_hotels = user_ratings[user_ratings > 0].sort_values(ascending=False)
                
                for hotel_id, rating in rated_hotels.items():
                    if rating >= 3:  
                        similar_user_hotels.append((hotel_id, rating * similarity))
        
        hotel_counts = Counter()
        for hotel_id, weighted_rating in similar_user_hotels:
            hotel_counts[hotel_id] += weighted_rating

        collab_top_hotels = [hotel_id for hotel_id, _ in hotel_counts.most_common(n_recommendations)]
        
   
        if collab_top_hotels:
            for hotel_id in collab_top_hotels:
                if hotel_id not in recommended_hotels:
                    recommended_hotels.append(hotel_id)
                    recommendation_reasons[hotel_id] = ["Recommended by users with similar preferences."]
    
   
    if user_id in content_similarity.index and not content_similarity.empty:
        user_prefs = content_similarity.loc[user_id]
        content_top_hotels = user_prefs.sort_values(ascending=False).index[:n_recommendations].tolist()
        
      
        for hotel_id in content_top_hotels:
            if hotel_id not in recommended_hotels:
                recommended_hotels.append(hotel_id)
                recommendation_reasons[hotel_id] = ["Matches your preferences based on hotel features."]
    
    if user_factors is not None and hotel_factors is not None and user_id in user_hotel_matrix.index:
 
        user_idx = list(user_hotel_matrix.index).index(user_id)
        
    
        user_vector = user_factors[user_idx].reshape(1, -1)
        predicted_ratings = np.dot(user_vector, hotel_factors.T).flatten()
        
       
        hotel_ids = user_hotel_matrix.columns
        hotel_ratings = list(zip(hotel_ids, predicted_ratings))
        
        sorted_hotels = sorted(hotel_ratings, key=lambda x: x[1], reverse=True)
        
        matrix_top_hotels = [hotel_id for hotel_id, _ in sorted_hotels[:n_recommendations]]
      
        for hotel_id in matrix_top_hotels:
            if hotel_id not in recommended_hotels:
                recommended_hotels.append(hotel_id)
                recommendation_reasons[hotel_id] = ["Predicted to match your preferences based on past behavior."]
    
    if len(recommended_hotels) < n_recommendations:
        trending_hotels = build_trending_recommendations(hotel_data, user_data, location_filter)
        
        for hotel_id in trending_hotels:
            if hotel_id not in recommended_hotels:
                recommended_hotels.append(hotel_id)
                recommendation_reasons[hotel_id] = ["Currently trending based on popularity and recency."]
                if len(recommended_hotels) >= n_recommendations:
                    break
    
    detailed_recommendations = []
    for hotel_id in recommended_hotels[:n_recommendations]:
        hotel_details = get_hotel_details(hotel_id, hotel_data)
        hotel_details['reasons'] = recommendation_reasons.get(hotel_id, ["No specific reasons available."])
        detailed_recommendations.append(hotel_details)
   
    if location_filter:
        filtered_recommendations = [
            hotel for hotel in detailed_recommendations
            if hotel.get('location', '').lower() == location_filter.lower()
            or location_filter.lower() in hotel.get('location', '').lower()
            or location_filter.lower() in hotel.get('city_name', '').lower()
        ]
        
        
        if filtered_recommendations:
            detailed_recommendations = filtered_recommendations
    
 
    return detailed_recommendations

def main_part2():

    (user_data, hotel_data, room_data, user_preferences, hotel_features, 
     user_similarity, user_hotel_matrix, content_similarity, user_factors, hotel_factors) = load_processed_data()
    
   
    
    user_id = data.get("user_id", None)
    recommendations = get_personalized_recommendations(user_id, user_data, hotel_data, room_data,
                                                      user_similarity=user_similarity, 
                                                      user_hotel_matrix=user_hotel_matrix,
                                                      content_similarity=content_similarity,
                                                      user_factors=user_factors,
                                                      hotel_factors=hotel_factors)
    
    print(f"Recommendations for user {user_id}:")
    for i, hotel in enumerate(recommendations, 1):
        print(f"{i}. {hotel['name']} - {hotel.get('rating', 'N/A')} ")
        if 'reasons' in hotel:
            print("   Reasons:")
            for reason in hotel['reasons']:
                print(f"   - {reason}")
        print()  
    
    new_visitor_info = {
        'hotel_type': 'Resort Hotel',
        'amenities': ['pool', 'spa'],
        'price_range': {'min': 100, 'max': 300},
        'family_size': 3
    }
    
    new_visitor_recommendations = get_new_visitor_recommendations(new_visitor_info, user_data, hotel_data, room_data,
                                                                 hotel_features=hotel_features)
    
    print("\nRecommendations for new visitor:")
    for i, hotel in enumerate(new_visitor_recommendations, 1):
        print(f"{i}. {hotel['name']} - {hotel.get('rating', 'N/A')} stars")
        if 'reasons' in hotel:
            print("   Reasons:")
            for reason in hotel['reasons']:
                print(f"   - {reason}")
        print()  

if __name__ == "__main__":
    main_part2()