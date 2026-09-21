import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODELS AND DATA
# ---------------------------------------------------

@st.cache_resource
def load_models():
    classification_model = joblib.load("classification_model.pkl")
    regression_model = joblib.load("regression_model.pkl")
    return classification_model, regression_model


@st.cache_data
def load_data():
    return pd.read_csv("india_housing_prices.csv")


classification_model, regression_model = load_models()
df = load_data()


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🏠 Real Estate Investment Advisor")
st.write(
    "Predict whether a property is a good investment "
    "and estimate its price after 5 years."
)

st.divider()


# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------

st.sidebar.header("🔎 Property Filters")

states = ["All"] + sorted(df["State"].dropna().astype(str).unique().tolist())
selected_state = st.sidebar.selectbox("State", states)

if selected_state != "All":
    filtered_df = df[df["State"].astype(str) == selected_state]
else:
    filtered_df = df.copy()


cities = ["All"] + sorted(
    filtered_df["City"].dropna().astype(str).unique().tolist()
)
selected_city = st.sidebar.selectbox("City", cities)

if selected_city != "All":
    filtered_df = filtered_df[
        filtered_df["City"].astype(str) == selected_city
    ]


property_types = ["All"] + sorted(
    filtered_df["Property_Type"].dropna().astype(str).unique().tolist()
)
selected_property_type = st.sidebar.selectbox(
    "Property Type",
    property_types
)

if selected_property_type != "All":
    filtered_df = filtered_df[
        filtered_df["Property_Type"].astype(str) == selected_property_type
    ]


bhk_values = sorted(
    pd.to_numeric(filtered_df["BHK"], errors="coerce")
    .dropna()
    .unique()
    .tolist()
)

if len(bhk_values) > 0:
    selected_bhk = st.sidebar.selectbox(
        "BHK",
        ["All"] + [int(x) for x in bhk_values]
    )

    if selected_bhk != "All":
        filtered_df = filtered_df[
            pd.to_numeric(filtered_df["BHK"], errors="coerce")
            == selected_bhk
        ]


# ---------------------------------------------------
# FILTERED PROPERTY DATA
# ---------------------------------------------------

st.subheader("🔎 Available Properties")

st.write(f"Number of properties matching filters: **{len(filtered_df)}**")

if len(filtered_df) > 0:
    display_columns = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Furnished_Status"
    ]

    display_columns = [
        col for col in display_columns if col in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[display_columns].head(100),
        use_container_width=True
    )
else:
    st.warning("No properties match the selected filters.")


st.divider()


# ---------------------------------------------------
# PROPERTY INPUT
# ---------------------------------------------------

st.subheader("🏡 Enter Property Details")

col1, col2 = st.columns(2)


with col1:

    state = st.selectbox(
        "State",
        sorted(df["State"].dropna().astype(str).unique().tolist())
    )

    city = st.selectbox(
        "City",
        sorted(df["City"].dropna().astype(str).unique().tolist())
    )

    locality = st.selectbox(
        "Locality",
        sorted(df["Locality"].dropna().astype(str).unique().tolist())
    )

    property_type = st.selectbox(
        "Property Type",
        sorted(df["Property_Type"].dropna().astype(str).unique().tolist())
    )

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    size = st.number_input(
        "Size in SqFt",
        min_value=100.0,
        value=1000.0,
        step=50.0
    )

    price = st.number_input(
        "Price in Lakhs",
        min_value=1.0,
        value=50.0,
        step=1.0
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1950,
        max_value=2026,
        value=2015,
        step=1
    )

    furnished_status = st.selectbox(
        "Furnished Status",
        sorted(
            df["Furnished_Status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    floor_no = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )


with col2:

    total_floors = st.number_input(
        "Total Floors",
        min_value=1,
        max_value=100,
        value=5,
        step=1
    )

    age = st.number_input(
        "Age of Property",
        min_value=0,
        max_value=100,
        value=10,
        step=1
    )

    nearby_schools = st.number_input(
        "Nearby Schools",
        min_value=0,
        value=3,
        step=1
    )

    nearby_hospitals = st.number_input(
        "Nearby Hospitals",
        min_value=0,
        value=2,
        step=1
    )

    public_transport = st.selectbox(
        "Public Transport Accessibility",
        sorted(
            df["Public_Transport_Accessibility"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    parking = st.selectbox(
        "Parking Space",
        sorted(
            df["Parking_Space"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    security = st.selectbox(
        "Security",
        sorted(
            df["Security"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    amenities = st.selectbox(
        "Amenities",
        sorted(
            df["Amenities"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    facing = st.selectbox(
        "Facing",
        sorted(
            df["Facing"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    owner_type = st.selectbox(
        "Owner Type",
        sorted(
            df["Owner_Type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    availability_status = st.selectbox(
        "Availability Status",
        sorted(
            df["Availability_Status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )


# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

st.divider()

if st.button("🔮 Predict Investment & Future Price", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "State": [state],
        "City": [city],
        "Locality": [locality],
        "Property_Type": [property_type],
        "BHK": [bhk],
        "Size_in_SqFt": [size],
        "Price_in_Lakhs": [price],
        "Year_Built": [year_built],
        "Furnished_Status": [furnished_status],
        "Floor_No": [floor_no],
        "Total_Floors": [total_floors],
        "Age_of_Property": [age],
        "Nearby_Schools": [nearby_schools],
        "Nearby_Hospitals": [nearby_hospitals],
        "Public_Transport_Accessibility": [public_transport],
        "Parking_Space": [parking],
        "Security": [security],
        "Amenities": [amenities],
        "Facing": [facing],
        "Owner_Type": [owner_type],
        "Availability_Status": [availability_status]
    })


    # ------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------

    try:

        investment_prediction = classification_model.predict(input_data)[0]

        if investment_prediction == 1:

            st.success("### ✅ Good Investment")

        else:

            st.warning("### ⚠️ Not a Good Investment")


        # Confidence
        if hasattr(classification_model, "predict_proba"):

            probabilities = classification_model.predict_proba(
                input_data
            )[0]

            confidence = max(probabilities) * 100

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

    except Exception as e:

        st.error(
            "Classification prediction error. "
            "Please check that the input columns match the trained model."
        )

        st.exception(e)


    # ------------------------------------------------
    # REGRESSION
    # ------------------------------------------------

    try:

        predicted_price = regression_model.predict(input_data)[0]

        st.subheader("💰 Estimated Property Price After 5 Years")

        st.metric(
            "Predicted Future Price",
            f"₹ {predicted_price:.2f} Lakhs"
        )

    except Exception as e:

        st.error(
            "Regression prediction error. "
            "Please check that the input columns match the trained model."
        )

        st.exception(e)


# ---------------------------------------------------
# DATA VISUALIZATION
# ---------------------------------------------------

st.divider()

st.subheader("📊 Property Market Analysis")


chart_col1, chart_col2 = st.columns(2)


with chart_col1:

    st.write("### Average Price by City")

    city_price = (
        df.groupby("City")["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(city_price)


with chart_col2:

    st.write("### Average Price by Property Type")

    property_price = (
        df.groupby("Property_Type")["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(property_price)


st.write("### 📈 Size vs Property Price")

size_price = (
    df[["Size_in_SqFt", "Price_in_Lakhs"]]
    .dropna()
    .sort_values("Size_in_SqFt")
    .set_index("Size_in_SqFt")
)

st.line_chart(size_price)


# ---------------------------------------------------
# CITY × PROPERTY TYPE HEATMAP
# ---------------------------------------------------

st.write("### 🔥 City vs Property Type Price Analysis")

heatmap_data = pd.pivot_table(
    df,
    values="Price_in_Lakhs",
    index="City",
    columns="Property_Type",
    aggfunc="mean"
)

st.dataframe(
    heatmap_data.round(2),
    use_container_width=True
)


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "Real Estate Investment Advisor | "
    "Machine Learning + Streamlit"
)