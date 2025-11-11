import streamlit as st
import pandas as pd 
import os
import requests
import time
from dotenv import load_dotenv



df = pd.read_csv("ai_news_with_sentiment_scores.csv")
df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
df.dropna(subset=['publishedAt'], inplace=True) # Drop rows where conversion failed

st.set_page_config(layout="wide")

#  Slack Connectivity
load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Function to send Slack alerts
def send_slack_alert(message: str):
    if not SLACK_WEBHOOK_URL:
        print("⚠️ No Slack webhook URL set.")
        st.sidebar.warning("Slack Webhook URL not configured.", icon="⚠️")
        return
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=10)
        if response.status_code == 200:
            print("✅ Alert sent successfully.")
        else:
            st.sidebar.warning(f"❌ Failed to send alert: {response.status_code} {response.text}")
    except Exception as e:
        st.warning(f"❌ Error sending alert: {e}")


def slack_message_label_filter():
    # Send Slack alerts
    alerts_sent = 0
    for i, row in label_filtered_df.head(5).iterrows():
        title = row.get("title", "Untitled")
        url = row.get("url", "")
        score = row.get("sentiment_score", "N/A")
        sentiment = row.get("sentiment", "Unknown")

        message = (
            f"*News Alert Triggered (Label Filter)!*\n"
            f"*Title:* {title}\n"
            f"*Sentiment:* {sentiment} ({score})\n"
            f"*URL:* {url}\n"
        )
        send_slack_alert(message)
        alerts_sent += 1
        time.sleep(1)

    if alerts_sent > 0:
        st.toast(f"Successfully sent {alerts_sent} Slack alerts!", icon="✅")

def slack_message_score_filter():
    # Send Slack alerts
    alerts_sent = 0
    for i, row in score_filtered_df.head(5).iterrows():
        title = row.get("title", "Untitled")
        url = row.get("url", "")
        score = row.get("sentiment_score", "N/A")
        sentiment = row.get("sentiment", "Unknown")

        message = (
            f"*News Alert Triggered (Score Filter)!*\n"
            f"*Title:* {title}\n"
            f"*Sentiment:* {sentiment} ({score})\n"
            f"*URL:* {url}\n"
        )
        send_slack_alert(message)
        alerts_sent += 1
        time.sleep(1)

    if alerts_sent > 0:
        st.toast(f"Successfully sent {alerts_sent} Slack alerts!", icon="✅")

#  Sidebar Navigation and Filters 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Dashboard", "Visualizations"))

st.sidebar.divider()
st.sidebar.subheader("Filters")

# Label Filter (sidebar)
Positive = st.sidebar.checkbox("Positive", value = True)
Negative = st.sidebar.checkbox("Negative", value = True)
Neutral = st.sidebar.checkbox("Neutral", value = True)
sentiment_filter = []
if Positive:
    sentiment_filter.append("Positive")
if Negative:
    sentiment_filter.append("Negative")
if Neutral:
    sentiment_filter.append("Neutral")

label_filtered_df = df[df["sentiment"].isin(sentiment_filter)][["title","sentiment_score","sentiment", "url"]].sort_values(by="sentiment_score")

# Score Filter (sidebar)
score_range = st.sidebar.slider("Sentiment Score Range", -1.0, 1.0, (-1.0, 1.0))

score_filtered_df = df[
    (df['sentiment_score'].between(score_range[0], score_range[1]))
][["title","sentiment_score","sentiment", "url"]].sort_values(by="sentiment_score")


# Page display

if page == "Dashboard":
    
    st.title("News Sentiment Dashboard")
    st.divider()

    # Major metrics
    sentiment_counts = df["sentiment"].value_counts()
    
    # get counts, defaulting to 0 if a sentiment is not present
    positive_count = sentiment_counts.get("Positive", 0)
    negative_count = sentiment_counts.get("Negative", 0)
    
    total_count = len(df)
    
    # Calculate ratios safely, avoiding ZeroDivisionError
    if total_count > 0:
        positive_ratio = (positive_count / total_count) * 100
        negative_ratio = (negative_count / total_count) * 100
        avg_score = f"{(sum(df["sentiment_score"])/total_count):.2f}"
    else:
        positive_ratio = 0.0
        negative_ratio = 0.0
        avg_score = "N/A" # Handle case for empty dataframe

    # Display important metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total News", total_count)
    col1.metric("Average Sentiment Score", avg_score)
    col2.metric(":green[Positive]", positive_count)
    col2.metric(":green[Positive %]", f"{positive_ratio:.1f}%")
    col3.metric(":red[Negative]", negative_count)
    col3.metric(":red[Negative %]", f"{negative_ratio:.1f}%")
    
    st.divider()
    st.subheader("Latest News")
    st.dataframe(df[["title","processed_text","sentiment_score","sentiment"]].head())
    
    st.divider()
    st.subheader(":orange[News Filtered by Label]")
    with st.expander("See all News Filtered by Label"):
        st.dataframe(label_filtered_df)
    st.text(f"{len(label_filtered_df)} {sentiment_filter} news articles available")

    st.markdown(
    "<h4>Click below to send Slack alerts about the filtered news articles</h4>",
    unsafe_allow_html=True)
    slack_button_label = st.button(label="Send Slack alert ⚠️", on_click=slack_message_label_filter, key="slack_label")

    st.divider()

    st.subheader(":orange[News Filtered by Score]")
    with st.expander("See all News Filtered by Score"):
        st.dataframe(score_filtered_df)
    st.text(f"{len(score_filtered_df)} news articles available with applied filters")

    st.markdown(
    "<h4>Click below to send Slack alerts about the filtered news articles</h4>",
    unsafe_allow_html=True)
    slack_button_score = st.button(label="Send Slack alert ⚠️", on_click=slack_message_score_filter, key="slack_score")

    st.divider()


elif page == "Visualizations":

    #  Visualization
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import plotly.express as px
    
    from forecast import model_training
    
    
    st.title("Visualizations")
    st.divider()

    sentiment_counts = df.value_counts("sentiment")

    # bar chart
    st.subheader("Sentiment Distribution Bar Chart")
    st.bar_chart(sentiment_counts)
    st.divider()

    # sentiment pie chart
    st.subheader("Sentiment Distribution Pie Chart")
    if not sentiment_counts.empty:
        chart_data = sentiment_counts.reset_index(name='count')
        
        # color map
        color_map = {
            'Positive': '#00FF00',
            'Negative': '#FF0000',
            'Neutral': "#DADADA"
        }
        
        # Plotly Express pie chart
        fig = px.pie(
            chart_data, 
            names='sentiment', 
            values='count',
            title='Sentiment Distribution',
            color='sentiment',              # Use 'sentiment' column for colors
            color_discrete_map=color_map    
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data to display in pie chart.")
    st.divider()

    # word cloud
    st.subheader("Word Cloud")
    texts = df["processed_text"].dropna().tolist()
    if texts:
        all_words = " ".join(texts).split()
        wordcloud = WordCloud(width=600, height=300, background_color=None).generate(" ".join(all_words))

        fig3, ax3 = plt.subplots(figsize=(5,4))
        ax3.imshow(wordcloud)
        ax3.axis("off")
        ax3.set_title("WordCloud of AI News")
        st.pyplot(fig3)
    else:
        st.write("Not enough text data to generate a word cloud.")
    st.divider()
    
    # trend and forecast
    
    # Aggregate sentiment counts by 15-minute intervals
    sentiment_over_time = df.groupby([df['publishedAt'].dt.floor('15min'), 'sentiment']) \
                            .size().unstack(fill_value=0).sort_index()

    trend_col, forecast_col = st.columns(2)
    with trend_col:
        st.subheader("Sentiment Trends Over Time")
        st.line_chart(sentiment_over_time, color = ["#FF0000", "#DDDDDD", "#00FF00"])

    
    forecast_df = model_training(df) # This function is from 'forecast.py'
    with forecast_col:
        st.subheader("Sentiment Trend Forecast - Prophet")
        st.line_chart(forecast_df, color = ["#FF0000", "#FFFFFF", "#00FF00"])
    
    st.divider()