import os
from flask import Flask, request, jsonify, session
from google import genai
from google.genai import types
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
from datetime import timedelta
import uuid

from hotel_recommendations_part2 import (
    load_processed_data,
    get_personalized_recommendations,
    get_new_visitor_recommendations,
    build_trending_recommendations,
    get_hotel_details,
)

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default-secret-key")
app.permanent_session_lifetime = timedelta(days=7)  


try:
    (user_data, hotel_data, room_data, user_preferences, hotel_features, 
     user_similarity, user_hotel_matrix, content_similarity, user_factors, hotel_factors) = load_processed_data()
    print("Successfully loaded recommendation data")
except Exception as e:
    print(f"Error loading recommendation data: {str(e)}")

def convert_to_native_types(data):
    """Convert NumPy types (int64, float64) to native Python types."""
    if isinstance(data, (np.int64, np.float64)):
        return data.item()  
    elif isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records") 
    else:
        return data
    
    
def get_recommendations_for_user(user_id=None, new_visitor_info=None, limit=5, location_specific=False):
    """Get hotel recommendations for a user or visitor."""
    try:
        if user_id:
            if user_id in user_data['user_id'].values:
               
                recommendations = get_personalized_recommendations(user_id, user_data, hotel_data, room_data,
                                                               user_similarity=user_similarity,
                                                               user_hotel_matrix=user_hotel_matrix,
                                                               content_similarity=content_similarity,
                                                               user_factors=user_factors,
                                                               hotel_factors=hotel_factors)
            else:
             
                if new_visitor_info:
                    recommendations = get_new_visitor_recommendations(new_visitor_info, user_data, hotel_data, room_data)
                else:
                    trending_hotels = build_trending_recommendations(hotel_data, user_data)
                    recommendations = [get_hotel_details(hotel_id, hotel_data) for hotel_id in trending_hotels[:limit]]
        else:
            
            if new_visitor_info:
                recommendations = get_new_visitor_recommendations(new_visitor_info, user_data, hotel_data, room_data)
            else:
             
                trending_hotels = build_trending_recommendations(hotel_data, user_data)
                recommendations = [get_hotel_details(hotel_id, hotel_data) for hotel_id in trending_hotels[:limit]]
                
        if location_specific and new_visitor_info and 'location' in new_visitor_info:
            location = new_visitor_info['location'].lower()
            recommendations = [hotel for hotel in recommendations
                            if hotel.get('location', '').lower() == location]
            
        recommendations_with_ids = []
        
        for hotel in recommendations:
            if isinstance(hotel, dict):
              
                if 'id' not in hotel and 'hotel_id' in hotel:
                    hotel['id'] = hotel['hotel_id']
                elif 'id' not in hotel and 'name' in hotel:
                     hotel['id'] = str(abs(hash(hotel['name'])))  
                recommendations_with_ids.append(hotel)
            else:
                
                hotel_details = get_hotel_details(hotel, hotel_data)
                if hotel_details:
                    hotel_details['id'] = str(hotel)  
                    recommendations_with_ids.append(hotel_details)
                
   
        return convert_to_native_types(recommendations_with_ids[:limit])
    except Exception as e:
        
        print(f"Error getting recommendations: {e}")
        return []

@app.route("/recommendations", methods=["POST"])
def recommendations_endpoint():
    data = request.json
    user_id = data.get("user_id", None)
    new_visitor_info = data.get("new_visitor_info", None)
    location_specific = data.get("location_specific", False)
    
    recommendations = get_recommendations_for_user(user_id, new_visitor_info, location_specific=location_specific)
    
    return jsonify({"recommendations": recommendations})


api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)

def initialize_user_session():
    """Initialize or retrieve user session data"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
        session['user_preferences'] = {}
    session.modified = True
    return session

def update_user_preferences(user_input, detected_preferences):
    """Update user preferences based on conversation"""
    if 'user_preferences' not in session:
        session['user_preferences'] = {}
  
    preference_keys = {
        'location': ['abu dhabi', 'dubai', 'sharjah', 'ajman', 'fujairah', 'ras al khaimah', 'umm al quwain'],
        'price_range': ['cheap', 'affordable', 'luxury', 'expensive', 'budget'],
        'amenities': ['pool', 'spa', 'gym', 'restaurant', 'wifi', 'beach'],
        'hotel_type': ['resort', 'city hotel', 'boutique', 'business'],
        'travel_purpose': ['business', 'vacation', 'family', 'romantic', 'solo']
    }
    
  
    for key, values in preference_keys.items():
        for value in values:
            if value in user_input.lower():
                session['user_preferences'][key] = value
                detected_preferences[key] = value
    
    if 'price' in user_input.lower() or 'budget' in user_input.lower():
        price_phrases = {
            'low': ['cheap', 'budget', 'low cost', 'affordable'],
            'medium': ['mid-range', 'reasonable', 'moderate'],
            'high': ['luxury', 'expensive', 'high-end', 'premium']
        }
        
        for range_type, phrases in price_phrases.items():
            if any(phrase in user_input.lower() for phrase in phrases):
                session['user_preferences']['price_range'] = range_type
                detected_preferences['price_range'] = range_type
    
    session.modified = True
    return detected_preferences

def extract_location(user_input):
    """Extract location from user input with improved detection"""
    location_keywords = ["in", "at", "to", "near", "around", "within", "close to"]
    locations = ["abu dhabi", "dubai", "sharjah", "ajman", "fujairah", "ras al khaimah", "umm al quwain"]
    
    user_input_lower = user_input.lower()
    
    for keyword in location_keywords:
        for loc in locations:
           
            if f"{keyword} {loc}" in user_input_lower:
                return loc
    

    for loc in locations:
        if loc in user_input_lower:
            return loc
    
    return None

def is_recommendation_request(user_input):
    """Check if user is asking for hotel recommendations"""
    recommendation_keywords = [
        "recommend", "suggestion", "hotel", "place to stay", 
        "where should i stay", "best hotel", "top hotel",
        "looking for", "find me", "suggest me", "options for",
        "places in", "hotels in", "accommodation"
    ]
    
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in recommendation_keywords)

def build_conversation_context():
    """Build context for the AI based on conversation history and preferences"""
    context = ""
    

    if 'user_preferences' in session and session['user_preferences']:
        context += "\n[USER PREFERENCES]:\n"
        for key, value in session['user_preferences'].items():
            context += f"- {key}: {value}\n"
    
    
    if 'conversation_history' in session and session['conversation_history']:
        context += "\n[CONVERSATION HISTORY]:\n"
        for msg in session['conversation_history'][-3:]:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context += f"{role}: {msg['content']}\n"
    
    return context

@app.route("/chat", methods=["POST"])
def chat_endpoint():
   
    user_session = initialize_user_session()
    data = request.json
    user_input = data.get("input", "").strip()
    user_id = data.get("user_id", None)
    new_visitor_info = data.get("new_visitor_info", {})
    
    if not user_input:
        return jsonify({"response": "Please provide some input."})
    
   
    session['conversation_history'].append({
        'role': 'user',
        'content': user_input,
        'timestamp': pd.Timestamp.now().isoformat()
    })
    
    detected_preferences = {}
    update_user_preferences(user_input, detected_preferences)

    location = extract_location(user_input)
    if location:
        new_visitor_info["location"] = location
        session['user_preferences']['location'] = location
        detected_preferences['location'] = location
    

    is_rec_request = is_recommendation_request(user_input)
    
 
    recommendation_context = ""
    if is_rec_request:
        recommendations = get_recommendations_for_user(
            user_id, 
            new_visitor_info or session.get('user_preferences', {}),
            limit=3
        )
        
        if recommendations:
            recommendation_context = "Based on our recommendation system, here are some hotels that might interest you:\n"
            for i, hotel in enumerate(recommendations, 1):
                hotel_info = f"{i}. {hotel.get('name', 'Unknown Hotel')} - "
                hotel_info += f"Rating: {hotel.get('rating', 'N/A')}/5, "
                hotel_info += f"Price: AED{hotel.get('price_per_night', 'N/A')}/night, "
                hotel_info += f"Location: {hotel.get('location', 'N/A')}"
                
                if 'facilities' in hotel:
                    hotel_info += f"\n   Facilities: {', '.join(hotel['facilities'][:3])}"
                
                recommendation_context += hotel_info + "\n"

    conversation_context = build_conversation_context()
    enhanced_input = f"{user_input}\n\n{conversation_context}"
    
    if recommendation_context:
        enhanced_input += f"\n[RECOMMENDATIONS]:\n{recommendation_context}"
    elif location and is_rec_request:
        enhanced_input += f"\n[LOCATION]: User is looking for hotels in {location.title()}."
    
    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=enhanced_input)],
        ),
    ]
    
 
    system_instruction = """
    You are a knowledgeable and friendly hotel booking assistant that helps users find and book hotels in the UAE. 
    You have access to a recommendation system that can suggest hotels based on user preferences and history.
    
    GUIDELINES:
    1. Be conversational, helpful, and professional
    2. Maintain context from previous messages in the conversation
    3. When presenting recommendations:
       - Highlight key features of each hotel
       - Mention why it might be a good fit based on user preferences
       - Present options clearly with relevant details
    4. If user asks about specific hotel details not provided:
       - Politely explain what information you do have
       - Offer to provide more recommendations with different criteria
    5. For booking inquiries:
       - Guide users to the platform's booking system
       - Don't make promises about availability
    6. Remember user preferences across the conversation
    7. Ask clarifying questions when needed to provide better recommendations
    8. **All prices are in AED (United Arab Emirates Dirham), and you should always communicate this to the user when discussing pricing.**
    
    TONE:
    - Warm and professional
    - Use simple, clear language
    - Be enthusiastic about helping
    - Avoid overly technical terms
    
    RESPONSE STRUCTURE:
    1. Acknowledge user's request
    2. Provide relevant information or recommendations
    3. Ask follow-up questions if needed
    4. Guide to next steps
    """
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_ONLY_HIGH"),
        ],
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text=system_instruction),
        ],
    )

    response_text = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text
    except Exception as e:
        response_text = "I apologize, but I'm having trouble processing your request right now. Please try again later."
        print(f"Error generating response: {str(e)}")
    
    session['conversation_history'].append({
        'role': 'assistant',
        'content': response_text,
        'timestamp': pd.Timestamp.now().isoformat()
    })
    session.modified = True
    
    return jsonify({
        "response": response_text,
        "session_id": session['session_id'],
        "detected_preferences": detected_preferences
    })

@app.route("/session", methods=["GET"])
def get_session():
    """Endpoint to retrieve current session """
    if 'session_id' not in session:
        return jsonify({"error": "No active session"}), 404
    
    return jsonify({
        "session_id": session['session_id'],
        "conversation_history": session.get('conversation_history', []),
        "user_preferences": session.get('user_preferences', {})
    })

@app.route("/clear_session", methods=["POST"])
def clear_session():
    """Endpoint to clear current session data"""
    session.clear()
    return jsonify({"status": "Session cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)