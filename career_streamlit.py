# app.py - Simple Streamlit web app for BUK
import streamlit as st
import joblib
import pandas as pd

# BUK Course Mapping
course_mapping = {
    'Computing_Sciences': [
        'B.Sc. Computer Science',
        'B.Sc. Software Engineering',
        'B.Sc. Information Technology'
    ],
    'Engineering': [
        'B.Eng. Civil Engineering',
        'B.Eng. Electrical Engineering',
        'B.Eng. Mechanical Engineering'
    ],
    'Clinical_Sciences': [
        'MBBS Medicine',
        'B. Nursing',
        'BDS Dentistry'
    ],
    'Arts_Creative': [
        'B.A. Fine Arts',
        'B.A. Theatre and Performing Arts',
        'B.A. Music'
    ],
    'Business_Management': [
        'B.Sc. Business Administration',
        'B.Sc. Accounting',
        'B.Sc. Banking & Finance'
    ],
    'Law': [
        'LLB. Common and Islamic Law'
    ],
    'Education': [
        'B.Ed. Education',
        'B.A. Education'
    ],
    'Allied_Health_Sciences': [
        'B. Medical Laboratory Science',
        'B. Radiography',
        'B. Physiotherapy'
    ],
    'Agriculture_Environmental': [
        'B. Agriculture',
        'B.Sc. Environmental Management',
        'B. Food Science and Technology'
    ],
    'Pharmaceutical_Sciences': [
        'Doctor of Pharmacy',
        'B.Sc. Pharmacology',
    ],
    'Skilled_Vocational': [
        'B.Tech. Electrical Technology',
        'B.Tech. Mechanical Technology'
    ]
}

# Load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load('career_model.pkl')
    preprocessor = joblib.load('career_encoder.pkl')
    label_encoder = joblib.load('career_label_encoder.pkl')
    return preprocessor, label_encoder, model

preprocessor, label_encoder, model = load_model()

# App title
st.title("🎓 university course Recommendation System")
st.write("Get personalized course recommendations based on your profile")

# User input form
with st.form("student_form"):
    st.subheader("Student Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox(
            "Academic Category",
            ["science", "art", "commercial"]
        )
        
        primary_hobby = st.selectbox(
            "Primary Hobby/Interest",
            ["coding", "writing", "volunteering", "drawing", 
             "debate", "business_pitch", "reading", "sports_team"]
        )
        
        math_confidence = st.slider(
            "Math Confidence", 1, 5, 3,
            help="1: Very Low, 5: Very High"
        )
    
    with col2:
        likes_computers = st.radio(
            "Interest in Computers",
            ["yes", "no"]
        )
        
        verbal_confidence = st.slider(
            "Verbal Confidence", 1, 5, 3,
            help="1: Very Low, 5: Very High"
        )
    
    submitted = st.form_submit_button("Get Course Recommendation")

# When form is submitted
if submitted:
    # Prepare input
    user_data = {
        'category': category,
        'primary_hobby': primary_hobby,
        'math_confidence': math_confidence,
        'likes_computers': 1 if likes_computers == 'yes' else 0,
        'verbal_confidence': verbal_confidence
    }
    
    # Create DataFrame
    user_df = pd.DataFrame([user_data])
    
    # Preprocess and predict
    user_processed = preprocessor.transform(user_df)
    prediction_num = model.predict(user_processed)[0]
    career_name = label_encoder.inverse_transform([prediction_num])[0]
    
    # Display results
    st.success(f"*Recommended Career Group:* {career_name}")
    
    # Show BUK courses for this career group
    st.subheader(f"📚 Recommended  Courses in {career_name}:")


    
    if career_name in course_mapping:
        courses = course_mapping[career_name]
        for course in courses:
            st.markdown(f"• *{course}*")
    else:
        st.warning(f"No courses mapped for {career_name} yet.")