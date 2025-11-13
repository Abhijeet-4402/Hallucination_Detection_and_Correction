import streamlit as st
import requests

st.title("AI Hallucination Detector")
st.markdown("This app checks answers for hallucinations using a Flask backend.")

API_URL = "http://127.0.0.1:5000/detect_hallucination"

user_question = st.text_input("❓ Enter your question here:")

if st.button("🚀 Get Fact-Checked Answer"):
    if user_question:
        with st.spinner(f"Querying Backend for: '{user_question}'..."):
            try:
                response = requests.post(API_URL, json={'question': user_question})
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis Complete!")
                    
                    raw_answer = result.get("raw_answer", "N/A")
                    corrected_answer = result.get("corrected_answer", "N/A")
                    confidence_score = result.get("confidence_score", 0.0)
                    
                    st.subheader(f"Results for: *{user_question}*")
                    st.markdown("---")

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.error("🚨 Raw Answer (Potential Hallucination)")
                        st.code(raw_answer, language='text')
                        st.caption("This is the initial, ungrounded response from the LLM.")

                    with col2:
                        st.success("✅ Corrected Answer (Grounded Evidence)")
                        st.code(corrected_answer, language='text')
                        st.caption("This answer has been regenerated using evidence from Wikipedia.")

                    st.markdown("---")
                    st.metric(
                        label="Confidence Score",
                        value=f"{int(confidence_score * 100)}%",
                        delta="Higher is better"
                    )
                    st.text("Citations: Retrieved from Wikipedia/ChromaDB (Simulated)")
                else:
                    st.error(f"Error connecting to backend API: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Flask backend. Make sure `api.py` is running on http://127.0.0.1:5000.")
    else:
        st.warning("Please enter a question to start the detection process!")
