import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Investment Advisor")
st.write("Enter property details to estimate the house price.")

# Small built-in reference dataset.
# This keeps the deployed Streamlit app independent of the large original CSV.
sample = pd.DataFrame([
    ["Telangana","Hyderabad","Kondapur","Apartment",2,1200,2018,"Semi-Furnished",5,10,6,3,2,"Good","Yes","Yes","Gym,Parking","East","Owner","Ready to Move",72],
    ["Telangana","Hyderabad","Gachibowli","Apartment",3,1650,2020,"Furnished",8,15,4,4,3,"Excellent","Yes","Yes","Gym,Pool,Parking","East","Owner","Ready to Move",110],
    ["Telangana","Hyderabad","Miyapur","Apartment",2,1100,2017,"Semi-Furnished",4,10,7,2,2,"Good","Yes","Yes","Parking","North","Owner","Ready to Move",58],
    ["Karnataka","Bangalore","Whitefield","Apartment",2,1250,2019,"Furnished",7,14,5,3,3,"Excellent","Yes","Yes","Gym,Pool,Parking","East","Owner","Ready to Move",95],
    ["Maharashtra","Mumbai","Andheri","Apartment",2,900,2016,"Furnished",12,20,8,4,3,"Excellent","Yes","Yes","Gym,Parking","West","Owner","Ready to Move",145],
    ["Tamil Nadu","Chennai","OMR","Villa",3,1800,2018,"Semi-Furnished",1,2,6,3,2,"Good","Yes","Yes","Garden,Parking","North","Owner","Ready to Move",88],
    ["Delhi","New Delhi","Dwarka","Apartment",3,1500,2015,"Furnished",6,12,10,4,3,"Good","Yes","Yes","Gym,Parking","North","Owner","Ready to Move",105],
    ["Telangana","Hyderabad","Banjara Hills","Villa",4,2600,2014,"Furnished",1,2,12,5,4,"Excellent","Yes","Yes","Garden,Pool,Parking","North","Owner","Ready to Move",220],
    ["Karnataka","Bangalore","Electronic City","Apartment",2,1150,2021,"Unfurnished",3,10,3,2,2,"Good","Yes","No","Parking","East","Owner","Ready to Move",62],
    ["Telangana","Warangal","Hanamkonda","Independent House",3,1700,2016,"Semi-Furnished",1,2,9,3,2,"Average","Yes","Yes","Parking","East","Owner","Ready to Move",52],
], columns=[
    "State","City","Locality","Property_Type","BHK","Size_in_SqFt",
    "Year_Built","Furnished_Status","Floor_No","Total_Floors",
    "Age_of_Property","Nearby_Schools","Nearby_Hospitals",
    "Public_Transport_Accessibility","Parking_Space","Security",
    "Amenities","Facing","Owner_Type","Availability_Status",
    "Price_in_Lakhs"
])

features = [c for c in sample.columns if c != "Price_in_Lakhs"]
X = sample[features]
y = sample["Price_in_Lakhs"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42
    ))
])

model.fit(X, y)

def choices(col):
    return sorted(sample[col].unique().tolist())

st.subheader("Enter Property Details")

c1, c2, c3 = st.columns(3)

with c1:
    state = st.selectbox("State", choices("State"))
    city = st.selectbox("City", choices("City"))
    locality = st.selectbox("Locality", choices("Locality"))
    property_type = st.selectbox("Property Type", choices("Property_Type"))
    bhk = st.number_input("BHK", 1, 10, 2)
    size = st.number_input("Size (Sq Ft)", 300, 10000, 1200)
    year = st.number_input("Year Built", 1950, 2026, 2018)

with c2:
    furnished = st.selectbox("Furnished Status", choices("Furnished_Status"))
    floor = st.number_input("Floor No", 0, 50, 5)
    total_floors = st.number_input("Total Floors", 1, 60, 10)
    age = st.number_input("Age of Property", 0, 100, 6)
    schools = st.number_input("Nearby Schools", 0, 20, 3)
    hospitals = st.number_input("Nearby Hospitals", 0, 20, 2)
    transport = st.selectbox(
        "Public Transport Accessibility",
        choices("Public_Transport_Accessibility")
    )

with c3:
    parking = st.selectbox("Parking Space", choices("Parking_Space"))
    security = st.selectbox("Security", choices("Security"))
    amenities = st.selectbox("Amenities", choices("Amenities"))
    facing = st.selectbox("Facing", choices("Facing"))
    owner = st.selectbox("Owner Type", choices("Owner_Type"))
    availability = st.selectbox("Availability Status", choices("Availability_Status"))

input_data = pd.DataFrame([{
    "State": state,
    "City": city,
    "Locality": locality,
    "Property_Type": property_type,
    "BHK": bhk,
    "Size_in_SqFt": size,
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
    prediction = float(model.predict(input_data)[0])

    st.success(f"## Estimated House Price: ₹ {prediction:,.2f} Lakhs")

    st.subheader("Property Summary")
    summary = input_data.T.rename(columns={0: "Value"})
    st.dataframe(summary, use_container_width=True)

st.divider()
st.caption(
    "Real Estate Investment Advisor | Machine Learning + Streamlit"
)
