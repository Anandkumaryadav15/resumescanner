import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# User and Admin credentials
users = {
    "user": "user123",
    "admin": "admin123"
}

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Logged in as {username}")
        else:
            st.error("Invalid username or password")

def logout():
    st.sidebar.title("Logout")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.success("Logged out successfully")

# Web app
def main():
    st.title("Resume Screening App")

    # Login functionality
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = None

    if not st.session_state['logged_in']:
        login()
    else:
        logout()

        uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

        if uploaded_file is not None:
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

            cleaned_resume = clean_resume(resume_text)
            input_features = tfidfd.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")
            st.write("Predicted Category:", category_name)

            # Determine if candidate is selected or not
            selected_categories = [
                "Java Developer", "Python Developer", "Data Science", "DevOps Engineer", "Blockchain"
            ]
            candidate_status = "Selected" if category_name in selected_categories else "Not Selected"
            st.write("Candidate Status:", candidate_status)

            # Pie chart data
            labels = ['Selected', 'Not Selected']
            sizes = [1, 0] if candidate_status == "Selected" else [0, 1]
            colors = ['#4CAF50', '#FF4136']
            explode = (0.1, 0)  # explode the first slice if selected

            # Plotting the pie chart
            fig, ax = plt.subplots()
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            st.pyplot(fig)

# Python main
if __name__ == "__main__":
    main()
