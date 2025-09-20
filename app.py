import streamlit as st
import pandas as pd
import plotly.express as px
from model import EmotionAnalyzer
from utils import plot_emotion_distribution, create_wordcloud, plot_frequent_words
import os


def main():
    st.title("Hinglish Tweet Emotion Analysis")
    st.write("Analyze emotions in Hinglish tweets using NLP and Machine Learning")

    # Load the pre-trained model
    analyzer = EmotionAnalyzer()
    model_path = "models/emotion_analyzer.pkl"

    # Try to load the model
    try:
        analyzer.load_model(model_path)
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error("Error loading model. Please contact support.")
        st.stop()

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Predict Emotions", "Visualizations"])

    # Prediction Tab
    with tab1:
        st.header("Predict Emotions in Tweets")

        user_input = st.text_area("Enter a Hinglish tweet:", height=100)
        if st.button("Predict Emotion"):
            with st.spinner("Analyzing..."):
                prediction = analyzer.predict(user_input)
                # Display prediction with custom styling
                st.markdown(f"### Predicted Emotion: :blue[{prediction.upper()}]")
                # Show preprocessed text
                with st.expander("See how your text was processed"):
                    cleaned_text = analyzer.preprocessor.preprocess(user_input)
                    st.code(cleaned_text, language="text")

    # Visualization Tab
    with tab2:
        st.header("Tweet Analysis Visualizations")

        # Load the sample dataset for visualization
        try:
            # Load and preprocess the data
            df = pd.read_csv("data.csv")
            # Process the data
            processed_df = analyzer.prepare_data(df)

            # Emotion distribution
            st.subheader("Distribution of Emotions in Tweets")
            emotion_counts = df["Emotion"].value_counts().reset_index()
            emotion_counts.columns = ["Emotion", "Count"]  # Rename columns for plotting
            fig = px.bar(
                emotion_counts,
                x="Emotion",
                y="Count",
                title="Distribution of Emotions",
            )
            st.plotly_chart(fig)

            # Word clouds for each emotion
            st.subheader("Word Clouds by Emotion")
            for emotion in df["Emotion"].unique():
                emotion_texts = processed_df[processed_df["Emotion"] == emotion][
                    "cleaned_text"
                ]
                st.pyplot(
                    create_wordcloud(emotion_texts, f"Common Words in {emotion} tweets")
                )

            # Frequent words analysis
            st.subheader("Most Frequent Words Analysis")
            selected_emotion = st.selectbox(
                "Select emotion to see frequent words", sorted(df["Emotion"].unique())
            )
            emotion_texts = processed_df[processed_df["Emotion"] == selected_emotion][
                "cleaned_text"
            ]
            st.plotly_chart(plot_frequent_words(emotion_texts, selected_emotion))

        except Exception as e:
            st.error(f"Error loading visualization data: {str(e)}")


if __name__ == "__main__":
    main()
