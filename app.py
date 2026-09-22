import streamlit as st
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

DATA_FILE = "india_housing_prices(1).csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def train_model(df):
    target = "Price_in_Lakhs"
    drop_cols = ["ID", target]
    features = [c for c in df.columns if c not in drop_cols]

    # Keep deployment light and avoid huge one-hot matrices.
    data = df.sample(n=min(15000, len(df)), random_state=42).copy()
    X = data[features].copy()
    y = data[target].astype(float)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=50,
            max_depth=16,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return model

try:
    df = load_data()
except Exception:
    st.error("Dataset not found. Make sure india_housing_prices(1).csv is in the GitHub repository.")
    st.stop()

st.title("🏠 Real Estate Investment Advisor")
st.write("Enter property details to estimate the house price.")

with st.sidebar:
    st.header("Project Information")
    st.write(f"**Dataset:** {len(df):,} records")
    st.write("**Target:** Price_in_Lakhs")
    st.write("**Model:** Random Forest Regressor")
    st.write("**Deployment:** Streamlit")

def options(column):
    return sorted(df[column].dropna().astype(str).unique().tolist())

c1, c2, c3 = st.columns(3)

with c1:
    state = st.selectbox("State", options("State"))
    city = st.selectbox("City", options("City"))
    locality = st.selectbox("Locality", options("Locality"))
    property_type = st.selectbox("Property Type", options("Property_Type"))
    bhk = st.number_input("BHK", int(df["BHK"].min()), int(df["BHK"].max()), int(df["BHK"].median()))
    size = st.number_input("Size (Sq Ft)", int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max()), int(df["Size_in_SqFt"].median()))
    year = st.number_input("Year Built", int(df["Year_Built"].min()), 2026, int(df["Year_Built"].median()))

with c2:
    furnished = st.selectbox("Furnished Status", options("Furnished_Status"))
    floor = st.number_input("Floor No", int(df["Floor_No"].min()), int(df["Floor_No"].max()), int(df["Floor_No"].median()))
    total_floors = st.number_input("Total Floors", int(df["Total_Floors"].min()), int(df["Total_Floors"].max()), int(df["Total_Floors"].median()))
    age = st.number_input("Age of Property", int(df["Age_of_Property"].min()), int(df["Age_of_Property"].max()), int(df["Age_of_Property"].median()))
    schools = st.number_input("Nearby Schools", int(df["Nearby_Schools"].min()), int(df["Nearby_Schools"].max()), int(df["Nearby_Schools"].median()))
    hospitals = st.number_input("Nearby Hospitals", int(df["Nearby_Hospitals"].min()), int(df["Nearby_Hospitals"].max()), int(df["Nearby_Hospitals"].median()))
    transport = st.selectbox("Public Transport", options("Public_Transport_Accessibility"))

with c3:
    parking = st.selectbox("Parking Space", options("Parking_Space"))
    security = st.selectbox("Security", options("Security"))
    amenities = st.selectbox("Amenities", options("Amenities"))
    facing = st.selectbox("Facing", options("Facing"))
    owner = st.selectbox("Owner Type", options("Owner_Type"))
    availability = st.selectbox("Availability Status", options("Availability_Status"))

# Price_per_SqFt is derived from the target in this dataset, so use the dataset median.
median_ppsf = float(df["Price_per_SqFt"].median())

input_data = pd.DataFrame([{
    "State": state,
    "City": city,
    "Locality": locality,
    "Property_Type": property_type,
    "BHK": bhk,
    "Size_in_SqFt": size,
    "Price_per_SqFt": median_ppsf,
    "Year_Built": year,
    "Furnished_Status": furnished,
    "Floor_No": floor,
    "Total_Floors": total_floors,
    "Age_of_Property": age,
    "Nearby_Schools": schools,
    "Nearby_Hospitals": hospitals,
    "Public_Transport_Accessibility": transport,
    "Parking_Space": parking,
    "Security": security,
    "Amenities": amenities,
    "Facing": facing,
    "Owner_Type": owner,
    "Availability_Status": availability
}])

if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
    with st.spinner("Preparing the prediction model..."):
        model = train_model(df)

    prediction = float(model.predict(input_data)[0])

    st.success(f"### Estimated House Price: ₹ {prediction:,.2f} Lakhs")

    st.subheader("Property Summary")
    st.dataframe(
        input_data.T.rename(columns={0: "Value"}),
        use_container_width=True
    )

st.divider()
st.caption("Machine-learning based Real Estate Investment Advisor")
