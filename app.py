import pandas as pd
import streamlit as st
import random
import joblib
import src.feature_engineering
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Egypt Used Car Price Predictor",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css(file_name="styles.css"):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return joblib.load('models/used_car_price_pipeline.joblib')

@st.cache_data
def load_cleaned_data():
    return pd.read_csv('data/cleaned_data.csv')

def preprocess(df):
    df = df.copy()

    df["Color"] = df["Color"].replace(src.feature_engineering.color_mapping)
    df['Mileage'] = df['Mileage'].fillna(df['Mileage'].mean())

    return df

df = preprocess(load_cleaned_data())
load_css()
pipeline = load_pipeline()


# Helper Functions
def get_mode(series):
    return series.mode()[0]


def yes_to_bool(series):
    return get_mode(series) == "Yes"


def bool_to_yes_no(value):
    return "Yes" if value else "No"


def predict_price(user_inputs, pipeline_model):
    if not user_inputs:
        return None
    input_df = pd.DataFrame([user_inputs])
    try:
        prediction = pipeline_model.predict(input_df)[0]
        return round(prediction, 0)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

    
# Header Section
col_title, col_github = st.columns([4, 1])

with col_title:
    st.markdown(
        "<div class='main-title'>🚗 Egypt Used Car Price Predictor</div>",
        unsafe_allow_html=True)

    st.markdown(
    """
    <div class='sub-title'>
        Estimate the market value of a used car in Egypt using a machine learning
        model trained on more than <b>21,000</b> cleaned vehicle listings.
        Adjust the vehicle specifications to receive an instant price estimate
        and explore market insights.
    </div>
    """,
    unsafe_allow_html=True
    )


with col_github:
    st.write(" ")
    st.markdown(
        """
        <a href="https://github.com/MazenSr/Egypt-Used-Car-Price-Predictor" target="_blank" class="github-btn">
            <svg height="16" width="16" viewBox="0 0 16 16" style="vertical-align: text-bottom; fill: currentColor; margin-right: 5px;">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
            View on GitHub
        </a>
    """, unsafe_allow_html=True)

st.space(12)


# Sidebar Filtering Controls
st.sidebar.header("⚙️ Vehicle Specifications") 

makes = sorted(df['Make'].unique())
top_10 = df['Make'].value_counts().head(10).index.tolist()

if "default_make" not in st.session_state:
    st.session_state.default_make = random.choice(top_10)

selected_make = st.sidebar.selectbox("Brand", makes, index=makes.index(st.session_state.default_make))

slicing_df = df[df["Make"] == selected_make]

models = sorted(slicing_df["Model"].unique())
default_model = get_mode(slicing_df["Model"])
selected_model = st.sidebar.selectbox("Model", models, index=models.index(default_model),
                                      help=f"Defaults to the most common model for {selected_make}.")

slicing_df = slicing_df[slicing_df["Model"] == selected_model]

years = slicing_df["Year"]
min_year = int(years.min())
max_year = int(years.max())

if min_year == max_year:
    selected_year = min_year
    st.sidebar.write(f"Manufacturing Year : {selected_year}")
    st.sidebar.caption(f"Only {selected_year} models available in dataset.")
else:
    selected_year = st.sidebar.slider(
        "Manufacturing Year",
        min_value=min_year,
        max_value=max_year,
        value=max_year,
    )

mileages = slicing_df['Mileage']
selected_mileage = st.sidebar.number_input(
    "Mileage (km)", 
    min_value=0, 
    max_value=500000, 
    value=int(mileages.mean()), 
    step=5000,
    help=f"Average for {selected_make} {selected_model}: ~{mileages.mean():,.0f} km"
)

colors = sorted(slicing_df['Color'].unique())
default_color = get_mode(slicing_df['Color'])
selected_color = st.sidebar.selectbox("Color", colors, index=colors.index(default_color),
                                       help=f"Defaults to the most common color for {selected_make} {selected_model}.")


st.sidebar.subheader("Features & Options")

selected_transmission = st.sidebar.checkbox("Automatic Transmission", 
                                    value=yes_to_bool(slicing_df["Automatic Transmission"]))

selected_air_conditioner = st.sidebar.checkbox("Air Conditioner", 
                                       value=yes_to_bool(slicing_df["Air Conditioner"]))

selected_power_steering = st.sidebar.checkbox("Power Steering", 
                                      value=yes_to_bool(slicing_df["Power Steering"]))

selected_remote_control = st.sidebar.checkbox("Remote Control", 
                                      value=yes_to_bool(slicing_df["Remote Control"]))

st.sidebar.caption(f"Defaults are based on the selected {selected_make} {selected_model}.")


# Main Content Tabs 
tab_predict, tab_market = st.tabs(
    ["📊 Prediction", "📈 Market Insights"]
)

# Tab 1: Prediction
with tab_predict:

    st.subheader("Vehicle Configuration")

    features_columns = st.columns(5)
    features_columns[0].metric("Brand", selected_make)
    features_columns[1].metric("Model", selected_model)
    features_columns[2].metric("Year", selected_year)
    features_columns[3].metric("Mileage", f"{selected_mileage:,}")
    features_columns[4].metric("Color", selected_color)

    st.write(" ")
    st.markdown("**Selected Options:**")

    options_columns = st.columns(4)
    options_columns[0].info(
        f"⚙️ Automatic: {'Yes' if selected_transmission else 'No'}"
    )
    options_columns[1].info(
        f"❄️ Air Conditioner: {'Yes' if selected_air_conditioner else 'No'}"
    )
    options_columns[2].info(
        f"🎯 Power Steering: {'Yes' if selected_power_steering else 'No'}"
    )
    options_columns[3].info(
        f"🔑 Remote Control: {'Yes' if selected_remote_control else 'No'}"
    )

    st.space(20)


    predict_clicked = st.button(
        "Calculate Estimated Value", type="primary", use_container_width=True
    )

    if predict_clicked:
        input_data = {
            "Make": selected_make,
            "Model": selected_model,
            "Year": selected_year,
            "Mileage": selected_mileage,
            "Color": selected_color,
            "Automatic Transmission": bool_to_yes_no(selected_transmission),
            "Air Conditioner": bool_to_yes_no(selected_air_conditioner),
            "Power Steering": bool_to_yes_no(selected_power_steering),
            "Remote Control": bool_to_yes_no(selected_remote_control),
            "Date Displayed": pd.NaT,
        }

        predicted_val = predict_price(input_data, pipeline)

        if predicted_val is not None:
            st.markdown('<div id="prediction-result"></div>',
                         unsafe_allow_html=True)

            with st.container(border=True):
                st.success("Price estimation completed successfully.")
                st.metric(
                    label="Estimated Price", value=f"{predicted_val:,.0f} EGP"
                )
                
            components.html(
                """
                <script>
                    const element = window.parent.document.getElementById('prediction-result');
                    if (element) {
                        element.scrollIntoView({
                            behavior: 'smooth',
                            block: 'center'
                        });}
                </script>
                """,
                height=0
            )
    else:
        predicted_val = predict_price(None, pipeline)

st.caption(
        """Disclaimer: The estimated price is generated by a machine learning model
        trained on over 21,000 cleaned used car listings from Egypt through 2024.
        Actual market values may vary based on vehicle condition, maintenance history,
        accident record, location, negotiation, and changes in market conditions after 2024."""
    )       


# Tab 2: Insights
with tab_market:
    st.subheader(f"Market Analysis : {selected_make} {selected_model}")

    car_count = len(slicing_df)
    min_price = slicing_df["Price"].min()
    max_price = slicing_df["Price"].max()


    col1, col2 = st.columns(2)

    with col1:
        st.metric("Min Price in Data", f"{min_price:,.0f} EGP")
        st.metric("Dataset Car Count", f"{car_count:,} units")

    with col2:
        st.metric("Max Price in Data", f"{max_price:,.0f} EGP")
        st.metric(
            "Current Model Prediction",
            f"{predicted_val:,.0f} EGP" if predicted_val else "N/A")

    st.divider()


    st.markdown(f"### 📈 Price Distribution for {selected_make} {selected_model}")
    fig_dist = px.histogram(
            slicing_df,
            x="Price",
            nbins=50,
            labels={"Price": "Price (EGP)"},
            color_discrete_sequence=["#beb314"],
        )
    fig_dist.update_layout(bargap=0.1, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()


    st.markdown("### 📉 Mileage vs. Price Analysis")
    fig_scatter = px.scatter(
            slicing_df,
            x="Mileage",
            y="Price",
            color="Year",
            hover_data=["Year", "Color", "Automatic Transmission"],
            title=f"Relationship Between Mileage and Price for {selected_make} {selected_model}",
            labels={"Mileage": "Mileage (km)", "Price": "Price (EGP)"},
            color_continuous_scale="Viridis",
            trendline="ols",
        )
    fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()


    st.markdown(f"### 📊 Price Trends by Year for {selected_make} {selected_model}")
    year_price_df = (
            slicing_df.groupby("Year")["Price"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )
    year_price_df.columns = ["Year", "Average Price", "Min Price", "Max Price"]

    st.line_chart(
            data=year_price_df,
            x="Year",
            y=["Average Price", "Min Price", "Max Price"],
            color=["#00ff00", "#ff0000", "#0000ff"],
        )

# Footer
st.divider()
st.markdown(
    """
    <div class="custom-footer">
        <p>
            Built with <b>Streamlit</b>, <b>Scikit-learn</b> & <b>XGBoost</b> |
            Data source: <b>Hatla2ee</b> used car listings.
        </p>
        <p>
            Developed by <b>Mazen Mahmoud</b> •
            <a href="https://github.com/MazenSr" target="_blank">GitHub</a> •
            <a href="https://www.linkedin.com/in/mazen-mahmoud-ds/" target="_blank">LinkedIn</a> •
            <a href="mailto:mazen.mahmoud420409@gmail.com">Contact Me</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)    