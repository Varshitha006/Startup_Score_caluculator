import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#assigning weights to the columns in dataset
weights = {
    "team_experience": 0.10,
    "market_size_million_usd": 0.15,
    "monthly_active_users": 0.25,
    "monthly_burn_rate_inr": 0.10,
    "funds_raised_inr": 0.15,
    "valuation_inr": 0.25
}

# Scoring function 
def normalize_and_score(df, weights):
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    for col in weights:
        df_scaled[col] = scaler.fit_transform(df[[col]])
        if col == "monthly_burn_rate_inr":
            df_scaled[col] = 1 - df_scaled[col]
    df_scaled["score"] = sum(df_scaled[col] * weight for col, weight in weights.items())
    df["score"] = df_scaled["score"] * 100
    return df

# Loading the dataset
df = pd.read_csv(r"C:\Users\varsh\OneDrive\Desktop\Startup_Scoring_Dataset.csv")
df = normalize_and_score(df, weights)

# Streamlit app
st.title("🚀 Startup Success Evaluator")
st.write("Enter new startup values to get a success score:")

user_input = {}
for col in weights:
    user_input[col] = st.number_input(f"{col.replace('_', ' ').title()}", min_value=0.0)

if st.button("Evaluate Startup"):
    input_df = pd.DataFrame([user_input])
    full_df = pd.concat([df.drop(columns=["score"]), input_df], ignore_index=True)
    full_scored_df = normalize_and_score(full_df, weights)
    final_score = full_scored_df.iloc[-1]["score"]

    st.metric("Predicted Score", round(final_score, 2))
    if final_score > 75:
        st.success("High Potential")
    elif final_score > 50:
        st.warning(" Moderate Potential")
    else:
        st.error("Low Potential")
