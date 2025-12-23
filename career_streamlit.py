import streamlit as st
import joblib
import pandas as pd

# Course Mapping
course_mapping = {
    'Computing_Sciences': ['B.Sc. Computer Science','B.Sc. Software Engineering','B.Sc. Information Technology'],
    'Engineering': ['B.Eng. Civil Engineering','B.Eng. Electrical Engineering','B.Eng. Mechanical Engineering'],
    'Clinical_Sciences': ['MBBS Medicine','B. Nursing','BDS Dentistry'],
    'Arts_Creative': ['B.A. Fine Arts','B.A. Theatre and Performing Arts','B.A. Music'],
    'Business_Management': ['B.Sc. Business Administration','B.Sc. Accounting','B.Sc. Banking & Finance'],
    'Law': ['LLB. Common and Islamic Law'],
    'Education': ['B.Ed. Education','B.A. Education'],
    'Allied_Health_Sciences': ['B. Medical Laboratory Science','B. Radiography','B. Physiotherapy'],
    'Agriculture_Environmental': ['B. Agriculture','B.Sc. Environmental Management','B. Food Science and Technology'],
    'Pharmaceutical_Sciences': ['Doctor of Pharmacy','B.Sc. Pharmacology',],
    'Skilled_Vocational': ['B.Tech. Electrical Technology','B.Tech. Mechanical Technology']
}

# Load model and encoders
model = joblib.load('career_model.pkl')
preprocessor = joblib.load('career_encoder.pkl')
label_encoder = joblib.load('career_label_encoder.pkl')

# Simple title
st.title("🎓 university course Recommendation System")
st.write("Get personalized course recommendations based on your profile")

# User input form
st.subheader("Student Profile")
    
category = st.selectbox("Academic Category",["science", "art", "commercial"])
hobby = st.selectbox("Primary Hobby/Interest",["coding", "writing", "volunteering", "drawing",
                                                "debate", "business_pitch", "reading", "sports_team"])
math = st.slider("Math Confidence", 1, 5, 3)
likes_computers = st.radio("Interest in Computers",["yes", "no"])
verbal = st.slider("Verbal Confidence", 1, 5, 3)


# When button is clicked
if st.button("Get Course Recommendation"):
    # Prepare the answers
    sample = {
        'category': category,
        'primary_hobby': hobby,
        'math_confidence': math,
        'likes_computers': 1 if likes_computers == 'yes' else 0,
        'verbal_confidence': verbal
    }
    
    # Create dataframe
    sample_df = pd.DataFrame([sample])
    
    #preprocess sample_df
    processed_sample_df = preprocessor.transform(sample_df)
    
    #Recommend
    recommend = model.predict(processed_sample_df)[0]
    
    # Convert number to career group
    career_group = label_encoder.inverse_transform([recommend])[0]


    # Display results
    st.success(f"*Recommended Career Group:* {career_group}")
    
    # Show  courses for this career group
    st.subheader(f"📚 Recommended  Courses in {career_group}:")
    courses = course_mapping[career_group]
    for course in courses:
        st.markdown(f".{course}")











