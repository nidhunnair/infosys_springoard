import pandas as pd
import streamlit as st

df = pd.read_csv("ai_news_with_sentiment_scores.csv")

def model_training(df):
    from prophet import Prophet
    import pandas as pd

    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')

    batch = df.groupby([df['publishedAt'].dt.floor('15min'), 'sentiment']) \
              .size().unstack(fill_value=0).sort_index()

    future_steps = 14 
    sentiments = ['Negative', 'Neutral', 'Positive']

    fitted_models = {}
    forecast_df = pd.DataFrame()

    for col in sentiments:
        df_prophet = batch[[col]].reset_index()
        df_prophet.columns = ['ds', 'y']
        
        model = Prophet()
        model.add_seasonality(name='hourly_wave', period=(1/24), fourier_order=3)
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=future_steps, freq='15min')
        forecast = model.predict(future)
        
        fitted_models[col] = {'model': model, 'forecast': forecast}
        
        forecast_df[col] = forecast['yhat'][-future_steps:]


    forecast_df['ds'] = pd.date_range(
    start=forecast['ds'].max(),  # last timestamp in forecast
    periods= future_steps,       
    freq='D'                     # 1 day interval
)
    forecast_df = forecast_df.set_index('ds')

    return forecast_df

if __name__ == "__main__":
    forecast_df = model_training(df)

    st.line_chart(forecast_df)

