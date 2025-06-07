import pandas as pd
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import functools
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import random
from collections import Counter


@functools.lru_cache(maxsize=32)
def load_data():
    user_data = pd.read_csv('user_datas.csv')
    hotel_data = pd.read_csv('hotel_data.csv')
    room_data = pd.read_csv('rooms_data.csv')
    
    return user_data, hotel_data, room_data
def handle_outliers(df, column, lower_quantile=0.01, upper_quantile=0.99):
    q_low = df[column].quantile(lower_quantile)
    q_high = df[column].quantile(upper_quantile)
    df[column] = df[column].clip(q_low, q_high)
    return df

def preprocess_data(user_data, hotel_data, room_data):
    
    if 'hotel_id' not in hotel_data.columns:
        hotel_data['hotel_id'] = hotel_data.index + 1
    
    
    is_resort = hotel_data['description'].str.contains('resort', case=False, na=False)
    
    resort_hotels = hotel_data[is_resort]['hotel_id'].tolist()
    city_hotels = hotel_data[~is_resort]['hotel_id'].tolist()
    hotel_types = {
        'Resort Hotel': resort_hotels,
        'City Hotel': city_hotels
    }
    for df in [user_data, hotel_data, room_data]:
   
        df.fillna({
            'rating': 0,
            'price_per_night': df['price_per_night'].median() if 'price_per_night' in df else None,
            'description': '',
            'hotel_facilities': ''
        }, inplace=True)

        room_data.drop_duplicates(inplace=True)

    room_data = handle_outliers(room_data, 'price')
    hotel_data = handle_outliers(hotel_data, 'price_per_night')

    room_to_hotel = {}
    if 'hotel_id' in room_data.columns:
        room_to_hotel = dict(zip(room_data['room_type'], room_data['hotel_id']))
    
    rating_map = {'FiveStar': 5, 'FourStar': 4, 'ThreeStar': 3, 'TwoStar': 2, 'OneStar': 1}

    hotel_ratings = {}
    if 'rating' in hotel_data.columns:
        for _, row in hotel_data.iterrows():
            if 'hotel_id' in row and 'rating' in row:
                hotel_ratings[row['hotel_id']] = row['rating']
    
    def convert_rating_to_numeric(rating):
        if pd.isna(rating):
            return 0
        if isinstance(rating, str) and rating in rating_map:
            return rating_map[rating]
        return int(rating) if pd.notna(rating) else 0
    

    if 'hotel_id' not in user_data.columns:
        
        def assign_hotel_id(row):
            hotel_type = row.get('hotel_type')
            room_type = row.get('reserved_room_type')
            
            if hotel_type in hotel_types and room_type in room_to_hotel:
                hotel_id = room_to_hotel[room_type]
                if hotel_id in hotel_types[hotel_type]:
                    return hotel_id
            
            if hotel_type in hotel_types and hotel_types[hotel_type]:
                
                if 'rating' in row and not pd.isna(row['rating']):
                    user_rating = convert_rating_to_numeric(row['rating'])
                    for h_id in hotel_types[hotel_type]:
                    
                        hotel_rating = hotel_ratings.get(h_id)
                        if hotel_rating and convert_rating_to_numeric(hotel_rating) == user_rating:
                            return h_id
                
             
                return np.random.choice(hotel_types[hotel_type])
            
       
            return np.random.choice(hotel_data['hotel_id'].tolist()) if not hotel_data.empty else None
        
      
        user_data['hotel_id'] = user_data.apply(assign_hotel_id, axis=1)
    
    return user_data, hotel_data, room_data



def matrix_factorization(user_hotel_matrix, n_factors=20, n_iterations=20, learning_rate=0.01, regularization=0.02):
    """
    Perform matrix factorization using Singular Value Decomposition (SVD)
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
def save_processed_data(user_data, hotel_data, room_data, user_preferences, hotel_features, 
                       user_similarity, user_hotel_matrix, content_similarity, user_factors, hotel_factors):
    """Save all processed data"""
  
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    user_data.to_pickle('processed_data/user_data.pkl')
    hotel_data.to_pickle('processed_data/hotel_data.pkl')
    room_data.to_pickle('processed_data/room_data.pkl')
    
    user_preferences.to_pickle('processed_data/user_preferences.pkl')
    hotel_features.to_pickle('processed_data/hotel_features.pkl')
    
    user_similarity.to_pickle('processed_data/user_similarity.pkl')
    content_similarity.to_pickle('processed_data/content_similarity.pkl')
    
    user_hotel_matrix.to_pickle('processed_data/user_hotel_matrix.pkl')
    
    if user_factors is not None:
        np.save('processed_data/user_factors.npy', user_factors)
    
    if hotel_factors is not None:
        np.save('processed_data/hotel_factors.npy', hotel_factors)
    
    print("Data processing complete")
def main_part1():
   
    user_data, hotel_data, room_data = load_data()
    
    
    user_data, hotel_data, room_data = preprocess_data(user_data, hotel_data, room_data)
    
  
    user_similarity, user_hotel_matrix, user_factors, hotel_factors = collaborative_filtering(user_data)
    content_similarity, user_preferences, hotel_features = content_based_filtering(user_data, hotel_data, room_data)
    
  
    save_processed_data(user_data, hotel_data, room_data, user_preferences, hotel_features, 
                       user_similarity, user_hotel_matrix, content_similarity, user_factors, hotel_factors)

if __name__ == "__main__":
    main_part1()

