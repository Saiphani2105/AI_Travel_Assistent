import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Secure API key handling
api_key = "YOUR_GOOGLE_API_KEY"  # Store in .streamlit/secrets.toml

# System prompt for travel recommendations
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
    You are an AI travel planner. Given a source and destination, 
    suggest optimal travel options including cab, bus, train, and flights. 
    Estimate travel time and cost based on common travel patterns. 
    Consider the number of travelers and duration of stay for cost estimation. 
    Additionally, suggest must-visit local places at the destination.
    Provide results in a structured format with travel mode, estimated time, price, and recommended local attractions.
    """),
    ("human", """
    Plan travel from {source} to {destination} for {num_people} people staying {num_days} days. 
    Suggest transportation options, costs, and must-visit local attractions.
    """)
])

chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model='models/gemini-2.0-flash-exp')
parser = StrOutputParser()
chain = prompt_template | chat_model | parser

def get_travel_recommendations(source, destination, num_people, num_days):
    """Generates AI-powered travel recommendations."""
    user_input = {
        "source": source,
        "destination": destination,
        "num_people": num_people,
        "num_days": num_days
    }
    try:
        response = chain.invoke(user_input)
        return response if response else "No recommendations available."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("🌍 AI-Powered Travel Planner")
st.write("Enter your trip details to get personalized travel recommendations.")

col1, col2 = st.columns(2)

with col1:
    source = st.text_input("Source", placeholder="Enter city or airport")
    num_people = st.number_input("Number of Travelers", min_value=1, value=1, step=1)

with col2:
    destination = st.text_input("Destination", placeholder="Enter city or airport")
    num_days = st.number_input("Number of Days", min_value=1, value=1, step=1)

if st.button("Find Travel Options") and source and destination:
    with st.spinner("Fetching recommendations..."):
        travel_info = get_travel_recommendations(source, destination, num_people, num_days)
    
    st.subheader("Travel Recommendations")
    st.markdown(travel_info, unsafe_allow_html=True)
