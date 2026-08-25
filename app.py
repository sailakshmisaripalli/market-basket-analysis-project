
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp { background:#f4f7fb; }

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#172554 0%,#1e3a8a 100%);
}
section[data-testid="stSidebar"] * { color:white !important; }

section[data-testid="stSidebar"] .stButton button {
    text-align:left !important;
    justify-content:flex-start !important;
    font-size:16px !important;
    font-weight:600 !important;
    padding:10px 14px !important;
    margin:2px 0 !important;
    border-radius:8px !important;
    width:100%;
}
section[data-testid="stSidebar"] .stButton button p {
    text-align:left !important;
    font-size:16px !important;
    font-weight:600 !important;
}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background:#2563eb !important;
    border:1px solid #2563eb !important;
}
section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    background:transparent !important;
    border:1px solid transparent !important;
}
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background:rgba(255,255,255,.10) !important;
    border:1px solid rgba(255,255,255,.18) !important;
}

.submenu-wrapper {
    margin-left:16px;
    border-left:2px solid rgba(255,255,255,.25);
    padding-left:8px;
    margin-top:-2px;
    margin-bottom:4px;
}
.submenu-wrapper .stButton button {
    font-size:15px !important;
    font-weight:500 !important;
    padding:7px 12px !important;
}

.sidebar-title { font-size:26px; font-weight:700; margin-bottom:5px; }
.sidebar-subtitle { font-size:15px; opacity:.9; margin-bottom:30px; }

.section-title {
    font-size:27px;
    font-weight:700;
    color:#17346d;
    margin-top:24px;
    margin-bottom:18px;
}

.hero {
    background:linear-gradient(135deg,#172554,#2563eb);
    padding:38px;
    border-radius:0 0 22px 22px;
    color:white;
    margin-bottom:28px;
    box-shadow:0 10px 30px rgba(23,37,84,.18);
}
.hero h1 { font-size:38px; margin-bottom:12px; color:white; }
.hero p { font-size:17px; margin:0; color:white; }

.metric-card {
    background:white;
    border-radius:16px;
    padding:25px;
    min-height:160px;
    border:1px solid #e5e7eb;
    box-shadow:0 5px 18px rgba(15,23,42,.07);
    text-align:center;
}
.metric-title {
    color:#64748b;
    font-size:13px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:.5px;
    margin-bottom:12px;
}
.metric-value { color:#17346d; font-size:30px; font-weight:800; }
.metric-description { color:#64748b; font-size:13px; margin-top:8px; }

.white-card {
    background:white;
    border-radius:16px;
    padding:26px;
    border:1px solid #e5e7eb;
    box-shadow:0 5px 18px rgba(15,23,42,.06);
    margin-bottom:20px;
}
.white-card h2 { color:#1e293b; }

.success-box {
    background:#dcfce7;
    border:1px solid #bbf7d0;
    color:#166534;
    padding:15px 18px;
    border-radius:10px;
    margin:10px 0;
    font-weight:600;
}
.info-box {
    background:#eff6ff;
    border:1px solid #bfdbfe;
    color:#1e40af;
    padding:16px;
    border-radius:10px;
}

.recommendation-card {
    background:white;
    border-radius:14px;
    padding:20px;
    border-left:5px solid #2563eb;
    box-shadow:0 5px 15px rgba(15,23,42,.07);
    margin-bottom:14px;
}
.recommendation-title {
    color:#17346d;
    font-size:18px;
    font-weight:700;
    margin-bottom:12px;
}
.recommendation-metric {
    display:inline-block;
    margin-right:20px;
    color:#475569;
    font-size:14px;
}
.recommendation-metric strong { color:#1e293b; }

.footer {
    text-align:center;
    color:#64748b;
    font-size:13px;
    padding:30px 0;
}
.small-note { color:#64748b; font-size:13px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET
# ============================================================

def find_dataset():
    possible_files = [
        "online_retail_small.csv",
        "online_retail.csv",
        "online_retail.csv.csv",
        "Online Retail.csv",
        "OnlineRetail.csv",
        "data/online_retail.csv",
        "dataset/online_retail.csv"
    ]
    for file in possible_files:
        if os.path.exists(file):
            return file
    csv_files = glob.glob("*.csv")
    return csv_files[0] if csv_files else None


@st.cache_data
def load_data():
    file_path = find_dataset()
    if file_path is None:
        return None, None
    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
    except Exception:
        df = pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df, file_path


df, file_path = load_data()

if df is None:
    st.error("Dataset not found. Please place your CSV file in the same folder as app.py.")
    st.stop()


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_column(columns, names):
    names = [x.lower().strip() for x in names]
    for col in columns:
        if str(col).lower().strip() in names:
            return col
    return None


invoice_col = detect_column(df.columns, ["invoice","invoiceno","invoice no","invoice_no"])
stock_col = detect_column(df.columns, ["stockcode","stock_code","stock code"])
description_col = detect_column(df.columns, ["description"])
quantity_col = detect_column(df.columns, ["quantity","qty"])
customer_col = detect_column(df.columns, ["customerid","customer_id","customer id"])
unit_price_col = detect_column(df.columns, ["unitprice","unit_price","unit price"])

if invoice_col is None or description_col is None:
    st.error("Required columns not found. Dataset must contain Invoice/InvoiceNo and Description.")
    st.write(df.columns.tolist())
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

@st.cache_data
def prepare_data(data, invoice_column, quantity_column, description_column):
    cleaned = data.copy()

    cleaned[invoice_column] = cleaned[invoice_column].astype(str)
    cleaned = cleaned[~cleaned[invoice_column].str.upper().str.startswith("C")]

    if quantity_column and quantity_column in cleaned.columns:
        cleaned[quantity_column] = pd.to_numeric(cleaned[quantity_column], errors="coerce")
        cleaned = cleaned[cleaned[quantity_column] > 0]

    cleaned = cleaned[cleaned[description_column].notna()]
    cleaned[description_column] = cleaned[description_column].astype(str).str.strip()
    cleaned = cleaned[cleaned[description_column] != ""]

    if unit_price_col and unit_price_col in cleaned.columns:
        cleaned[unit_price_col] = pd.to_numeric(cleaned[unit_price_col], errors="coerce")

    return cleaned


cleaned_df = prepare_data(df, invoice_col, quantity_col, description_col)

total_rows = len(df)
total_columns = len(df.columns)
unique_products = int(df[description_col].nunique())
missing_values = int(df.isnull().sum().sum())
valid_records = len(cleaned_df)


# ============================================================
# TRANSACTIONS / BASKET
# ============================================================

@st.cache_data
def create_transactions(data, invoice_column, description_column):
    transactions = (
        data.groupby(invoice_column)[description_column]
        .apply(lambda x: list(set(x.dropna())))
        .tolist()
    )
    return [x for x in transactions if x]


@st.cache_data
def create_basket(transactions):
    if not transactions:
        return pd.DataFrame()
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(encoded, columns=encoder.columns_)


@st.cache_data
def generate_rules(basket, min_support, min_confidence):
    if basket.empty:
        return pd.DataFrame(), pd.DataFrame()

    if basket.shape[1] > 1500:
        counts = basket.sum(axis=0)
        basket = basket[counts.nlargest(1500).index]

    frequent = apriori(
        basket,
        min_support=max(min_support, 0.02),
        use_colnames=True,
        max_len=2,
        low_memory=True
    )

    if frequent.empty:
        return frequent, pd.DataFrame()

    rules = association_rules(
        frequent,
        metric="confidence",
        min_threshold=min_confidence
    )

    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence"], ascending=False)

    return frequent, rules


# ============================================================
# ML DATA / TRAINING
# ============================================================

@st.cache_data
def prepare_ml_data(data, quantity_column, unitprice_column):
    if quantity_column is None or unitprice_column is None:
        return None

    d = data.copy()
    d[quantity_column] = pd.to_numeric(d[quantity_column], errors="coerce")
    d[unitprice_column] = pd.to_numeric(d[unitprice_column], errors="coerce")
    d = d[(d[quantity_column] > 0) & (d[unitprice_column] > 0)].copy()

    if len(d) < 20:
        return None

    d["TransactionValue"] = d[quantity_column] * d[unitprice_column]
    median_value = float(d["TransactionValue"].median())
    d["HighValue"] = (d["TransactionValue"] >= median_value).astype(int)

    X = d[[quantity_column, unitprice_column]].copy()
    X.columns = ["Quantity", "UnitPrice"]
    y = d["HighValue"]

    return X, y, median_value


@st.cache_resource
def train_ml_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic = LogisticRegression(random_state=42, max_iter=1000)
    logistic.fit(X_train_scaled, y_train)
    logistic_pred = logistic.predict(X_test_scaled)

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [
            accuracy_score(y_test, logistic_pred),
            accuracy_score(y_test, rf_pred)
        ],
        "Precision": [
            precision_score(y_test, logistic_pred, zero_division=0),
            precision_score(y_test, rf_pred, zero_division=0)
        ],
        "Recall": [
            recall_score(y_test, logistic_pred, zero_division=0),
            recall_score(y_test, rf_pred, zero_division=0)
        ],
        "F1 Score": [
            f1_score(y_test, logistic_pred, zero_division=0),
            f1_score(y_test, rf_pred, zero_division=0)
        ]
    })

    return results, logistic, rf, scaler, X_test, y_test, logistic_pred, rf_pred


# ============================================================
# DL DATA / TRAINING
# ============================================================

@st.cache_data
def prepare_dl_data(data, quantity_column, unitprice_column):
    if quantity_column is None or unitprice_column is None:
        return None

    d = data.copy()
    d[quantity_column] = pd.to_numeric(d[quantity_column], errors="coerce")
    d[unitprice_column] = pd.to_numeric(d[unitprice_column], errors="coerce")
    d = d[(d[quantity_column] > 0) & (d[unitprice_column] > 0)].copy()

    if len(d) < 50:
        return None

    d["TransactionValue"] = d[quantity_column] * d[unitprice_column]
    median_value = float(d["TransactionValue"].median())
    d["HighValue"] = (d["TransactionValue"] >= median_value).astype(int)

    X = d[[quantity_column, unitprice_column]].copy()
    X.columns = ["Quantity", "UnitPrice"]
    y = d["HighValue"]

    return X, y, median_value


@st.cache_resource
def train_deep_learning_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Input(shape=(2,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=.20,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    probabilities = model.predict(X_test_scaled, verbose=0).ravel()
    predictions = (probabilities >= .5).astype(int)

    return (
        model, scaler, history, X_test, y_test,
        predictions, probabilities
    )


# ============================================================
# SESSION STATE
# ============================================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Home"

if "model_menu_open" not in st.session_state:
    st.session_state.model_menu_open = False

if "ml_trained" not in st.session_state:
    st.session_state.ml_trained = False

if "dl_trained" not in st.session_state:
    st.session_state.dl_trained = False


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">🛒 Market Basket</div>
    <div class="sidebar-subtitle">Analysis Dashboard</div>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 Navigation")

    model_pages = ["🧠 Deep Learning", "🤖 ML Prediction"]

    def nav_button(label, key, target=None):
        target = target or label
        active = st.session_state.active_page == target
        if st.button(
            label,
            key=key,
            width="stretch",
            type="primary" if active else "secondary"
        ):
            st.session_state.active_page = target
            st.rerun()

    nav_button("🏠 Home", "nav_home")
    nav_button("📊 Dashboard", "nav_dashboard")
    nav_button("🛍️ Recommendations", "nav_recommendations")

    model_active = st.session_state.active_page in model_pages
    if model_active:
        st.session_state.model_menu_open = True

    arrow = "▼" if st.session_state.model_menu_open else "▶"

    if st.button(
        f"🧠 MODEL INTELLIGENCE {arrow}",
        key="nav_model_group",
        width="stretch",
        type="primary" if model_active else "secondary"
    ):
        st.session_state.model_menu_open = not st.session_state.model_menu_open
        st.rerun()

    if st.session_state.model_menu_open:
        st.markdown('<div class="submenu-wrapper">', unsafe_allow_html=True)
        nav_button("├── 🧠 Deep Learning", "nav_dl", "🧠 Deep Learning")
        nav_button("└── 🤖 ML Prediction", "nav_ml", "🤖 ML Prediction")
        st.markdown("</div>", unsafe_allow_html=True)

    nav_button("📄 Dataset", "nav_dataset")
    nav_button("🎓 About Project", "nav_about")

    page = st.session_state.active_page

    st.markdown("---")
    st.markdown("### ⚙️ Algorithms")
    st.markdown("""
    <div style="background:#1d3b82;padding:15px;border-radius:10px;line-height:1.5;">
    <b>Apriori</b><br>Association Rule Mining<br><br>
    <b>Logistic Regression</b><br>Classification<br><br>
    <b>Random Forest</b><br>Machine Learning Classification<br><br>
    <b>Feedforward Neural Network</b><br>Deep Learning Classification
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero" style="text-align:center;padding:50px;">
        <h1>🛒 Welcome To Market Basket Analysis</h1>
        <p>Discover purchasing patterns, generate product recommendations,
        and classify transaction values using Machine Learning & Deep Learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🚀 Project Features</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    cards = [
        ("📊", "Dashboard", "Insights", "Dataset statistics, preprocessing and transaction patterns."),
        ("🛍️", "Recommendations", "Smart", "Discover products frequently purchased together using Apriori."),
        ("🤖", "ML Models", "Predictive", "Compare Logistic Regression and Random Forest.")
    ]

    for col, (icon, title, value, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top:5px solid #2563eb;">
                <div style="font-size:30px">{icon}</div>
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-description">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:35px;background:linear-gradient(135deg,#172554,#2563eb);
    padding:35px;border-radius:20px;color:white;">
        <div style="font-size:14px;font-weight:700;letter-spacing:1px;">
        🧠 ADVANCED PREDICTION MODULE</div>
        <div style="font-size:30px;font-weight:800;margin:8px 0;">
        Deep Learning Intelligence</div>
        <div style="font-size:17px;line-height:1.6;">
        A Feedforward Neural Network uses Quantity and Unit Price
        to classify transactions into Normal-Value and High-Value categories.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔗 Complete Project Workflow</div>', unsafe_allow_html=True)

    w1, w2, w3, w4 = st.columns(4)
    workflow = [
        ("📄", "Dataset", "Load & clean retail data"),
        ("🛍️", "Apriori", "Discover product associations"),
        ("🤖", "Machine Learning", "Classify transaction values"),
        ("🧠", "Deep Learning", "Neural network classification")
    ]

    for col, (icon, title, desc) in zip([w1,w2,w3,w4], workflow):
        with col:
            st.markdown(f"""
            <div class="white-card" style="text-align:center;min-height:145px;">
                <div style="font-size:30px">{icon}</div>
                <h4 style="color:#17346d">{title}</h4>
                <p class="small-note">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">© 2026 | Market Basket Analysis Project |
    Python • Streamlit • Apriori • Machine Learning • Deep Learning</div>
    """, unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🛒 Market Basket Analysis</h1>
        <p>Discover purchasing patterns and understand retail transaction behaviour.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    metrics = [
        ("Total Records", f"{total_rows:,}", "Rows in original dataset"),
        ("Columns", str(total_columns), "Available attributes"),
        ("Unique Products", f"{unique_products:,}", "Different products"),
        ("Missing Values", f"{missing_values:,}", "Before preprocessing")
    ]

    for col,(title,value,desc) in zip([c1,c2,c3,c4],metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-description">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧹 Data Preparation</div>', unsafe_allow_html=True)
    p1,p2 = st.columns(2)
    with p1:
        st.markdown(f'<div class="success-box">✅ Valid records: {valid_records:,}</div>', unsafe_allow_html=True)
    with p2:
        removed = total_rows - valid_records
        st.markdown(f'<div class="success-box">✅ Removed/filtered records: {removed:,}</div>', unsafe_allow_html=True)

    # NEW: dataset visual insights
    st.markdown('<div class="section-title">📈 Data Insights</div>', unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)

    with chart1:
        if description_col:
            top_products = cleaned_df[description_col].value_counts().head(10).sort_values()
            st.subheader("🏆 Top 10 Products")
            st.bar_chart(top_products)

    with chart2:
        if quantity_col and unit_price_col:
            temp = cleaned_df[[quantity_col, unit_price_col]].copy()
            temp[quantity_col] = pd.to_numeric(temp[quantity_col], errors="coerce")
            temp[unit_price_col] = pd.to_numeric(temp[unit_price_col], errors="coerce")
            temp["TransactionValue"] = temp[quantity_col] * temp[unit_price_col]
            temp = temp.replace([np.inf,-np.inf], np.nan).dropna()
            if not temp.empty:
                st.subheader("💰 Transaction Value Distribution")
                st.bar_chart(temp["TransactionValue"].clip(upper=temp["TransactionValue"].quantile(.99)).head(1000))

    # Country insight
    if "Country" in cleaned_df.columns:
        st.subheader("🌍 Top Countries by Records")
        country_counts = cleaned_df["Country"].value_counts().head(10)
        st.bar_chart(country_counts)

    st.markdown('<div class="section-title">🧺 Transaction Basket</div>', unsafe_allow_html=True)

    transactions = create_transactions(cleaned_df, invoice_col, description_col)
    st.success(f"Transaction basket created successfully: {len(transactions):,} transactions")

    with st.expander("📋 View sample transactions"):
        for i, transaction in enumerate(transactions[:10]):
            st.write(f"**Transaction {i+1}:** " + ", ".join(transaction[:10]))


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🛍️ Recommendations":

    st.markdown('<div class="section-title">🛍️ Product Recommendation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="white-card">
        <h2>Find Products Purchased Together</h2>
        <p>Select a product and use Apriori to discover products
        commonly purchased with it.</p>
    </div>
    """, unsafe_allow_html=True)

    transactions = create_transactions(cleaned_df, invoice_col, description_col)

    if not transactions:
        st.error("Unable to create transaction basket.")
        st.stop()

    all_products = sorted(set(p for t in transactions for p in t))
    selected_product = st.selectbox("🔎 Select a product", all_products)

    col1,col2 = st.columns(2)
    with col1:
        min_support = st.slider("Minimum Support", .01, .10, .02, .01)
    with col2:
        min_confidence = st.slider("Minimum Confidence", .10, .90, .30, .05)

    generate = st.button("🔍 Generate Product Recommendations", type="primary", width="stretch")

    if generate:
        with st.spinner("Running Apriori algorithm..."):
            basket = create_basket(transactions)
            frequent_itemsets, rules = generate_rules(basket, min_support, min_confidence)

        if rules.empty:
            st.warning("No association rules found. Try lower support/confidence.")
        else:
            selected_rules = rules[
                rules["antecedents"].apply(lambda x: selected_product in x)
            ].copy()

            if selected_rules.empty:
                selected_rules = rules[
                    rules["consequents"].apply(lambda x: selected_product in x)
                ].copy()

            if selected_rules.empty:
                st.warning(f"No recommendations found for '{selected_product}'.")
            else:
                selected_rules = selected_rules.sort_values(
                    ["confidence","lift"], ascending=False
                )

                st.success(f"Recommendation analysis completed for '{selected_product}'.")

                r1,r2,r3 = st.columns(3)
                r1.metric("📌 Rules Found", len(selected_rules))
                r2.metric("🎯 Best Confidence", f"{selected_rules['confidence'].max():.2%}")
                r3.metric("🚀 Best Lift", f"{selected_rules['lift'].max():.2f}")

                st.markdown('<div class="section-title">🏆 Top Recommendations</div>', unsafe_allow_html=True)

                rows = []
                number = 1

                for _, rule in selected_rules.head(10).iterrows():
                    ants = list(rule["antecedents"])
                    cons = list(rule["consequents"])
                    recs = [x for x in cons if x != selected_product]
                    if not recs:
                        recs = [x for x in ants if x != selected_product]
                    if not recs:
                        continue

                    rec = recs[0]
                    rows.append({
                        "Rank": number,
                        "Recommended Product": rec,
                        "Support": rule["support"],
                        "Confidence": rule["confidence"],
                        "Lift": rule["lift"]
                    })

                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="recommendation-title">#{number} 🛍️ {rec}</div>
                        <span class="recommendation-metric"><strong>Support:</strong> {rule['support']:.2%}</span>
                        <span class="recommendation-metric"><strong>Confidence:</strong> {rule['confidence']:.2%}</span>
                        <span class="recommendation-metric"><strong>Lift:</strong> {rule['lift']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    number += 1

                if rows:
                    export_df = pd.DataFrame(rows)
                    csv_data = export_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Recommendations CSV",
                        csv_data,
                        "product_recommendations.csv",
                        "text/csv",
                        type="secondary"
                    )

                st.markdown('<div class="section-title">📈 Association Rules</div>', unsafe_allow_html=True)

                display = selected_rules.copy()
                display["Antecedent"] = display["antecedents"].apply(lambda x: ", ".join(x))
                display["Recommended Product"] = display["consequents"].apply(lambda x: ", ".join(x))
                display["Support"] = display["support"].map(lambda x: f"{x:.2%}")
                display["Confidence"] = display["confidence"].map(lambda x: f"{x:.2%}")
                display["Lift"] = display["lift"].map(lambda x: f"{x:.2f}")
                display = display[[
                    "Antecedent","Recommended Product","Support","Confidence","Lift"
                ]].head(10)

                st.dataframe(display, width="stretch", hide_index=True)


# ============================================================
# DEEP LEARNING
# ============================================================

elif page == "🧠 Deep Learning":

    st.markdown("""
    <div class="hero">
        <h1>🧠 Deep Learning Prediction</h1>
        <p>Feedforward Neural Network based transaction value classification.</p>
    </div>
    """, unsafe_allow_html=True)

    dl_result = prepare_dl_data(cleaned_df, quantity_col, unit_price_col)

    if dl_result is None:
        st.error("Quantity and UnitPrice columns are required for Deep Learning.")
        st.stop()

    X_dl, y_dl, median_dl = dl_result

    st.markdown(f"""
    <div class="white-card">
        <h2>Feedforward Neural Network</h2>
        <p><b>Input:</b> Quantity and Unit Price</p>
        <p><b>Target:</b> Normal-Value / High-Value</p>
        <p><b>Classification Threshold:</b> {median_dl:.2f}</p>
        <p><b>Architecture:</b> 2 → 64 → 32 → 1</p>
    </div>
    """, unsafe_allow_html=True)

    a1,a2,a3,a4 = st.columns(4)
    a1.metric("Input Layer","2 Features")
    a2.metric("Hidden Layer 1","64")
    a3.metric("Hidden Layer 2","32")
    a4.metric("Output Layer","1")

    train_dl = st.button("🚀 Train Deep Learning Model", type="primary", width="stretch")

    if train_dl:
        with st.spinner("Training Feedforward Neural Network..."):
            result = train_deep_learning_model(X_dl, y_dl)
        (
            dl_model, dl_scaler, dl_history, X_test_dl,
            y_test_dl, dl_predictions, dl_probabilities
        ) = result

        st.session_state.dl_result = result
        st.session_state.dl_trained = True

    if st.session_state.dl_trained and "dl_result" in st.session_state:
        (
            dl_model, dl_scaler, dl_history, X_test_dl,
            y_test_dl, dl_predictions, dl_probabilities
        ) = st.session_state.dl_result

        st.success("✅ Deep Learning model is ready.")

        acc = accuracy_score(y_test_dl, dl_predictions)
        prec = precision_score(y_test_dl, dl_predictions, zero_division=0)
        rec = recall_score(y_test_dl, dl_predictions, zero_division=0)
        f1 = f1_score(y_test_dl, dl_predictions, zero_division=0)

        st.markdown('<div class="section-title">📊 Deep Learning Performance</div>', unsafe_allow_html=True)
        p1,p2,p3,p4 = st.columns(4)
        p1.metric("Accuracy", f"{acc:.2%}")
        p2.metric("Precision", f"{prec:.2%}")
        p3.metric("Recall", f"{rec:.2%}")
        p4.metric("F1 Score", f"{f1:.2%}")

        st.markdown('<div class="section-title">📈 Training Accuracy</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame({
            "Training Accuracy": dl_history.history["accuracy"],
            "Validation Accuracy": dl_history.history["val_accuracy"]
        })
        st.line_chart(hist_df)

        st.markdown('<div class="section-title">📉 Training Loss</div>', unsafe_allow_html=True)
        loss_df = pd.DataFrame({
            "Training Loss": dl_history.history["loss"],
            "Validation Loss": dl_history.history["val_loss"]
        })
        st.line_chart(loss_df)

        st.markdown('<div class="section-title">🔍 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test_dl, dl_predictions)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal","Actual High-Value"],
            columns=["Predicted Normal","Predicted High-Value"]
        )
        st.dataframe(cm_df, width="stretch")

        st.markdown('<div class="section-title">🔮 DL Transaction Prediction</div>', unsafe_allow_html=True)

        q1,q2 = st.columns(2)
        with q1:
            input_quantity = st.number_input("🛒 Quantity", min_value=1.0, value=2.0, step=1.0, key="dl_qty")
        with q2:
            input_unit_price = st.number_input("💰 Unit Price", min_value=.01, value=10.0, step=.50, key="dl_price")

        transaction_value = input_quantity * input_unit_price
        st.info(f"Transaction Value = ₹{transaction_value:,.2f} | Threshold = ₹{median_dl:,.2f}")

        if st.button("🔮 Predict Transaction", type="primary", width="stretch", key="dl_predict"):
            input_data = np.array([[input_quantity,input_unit_price]])
            scaled = dl_scaler.transform(input_data)
            probability = float(dl_model.predict(scaled, verbose=0)[0][0])
            prediction = int(probability >= .5)

            if prediction:
                st.success("🟢 HIGH-VALUE TRANSACTION")
            else:
                st.info("🔵 NORMAL-VALUE TRANSACTION")

            st.metric("High-Value Probability", f"{probability:.2%}")
            st.progress(float(probability))
            st.write(f"**Transaction Value:** ₹{transaction_value:,.2f}")
            st.write(f"**Classification Threshold:** ₹{median_dl:,.2f}")

        st.markdown("""
        <div class="white-card">
            <h2>📚 Model Explanation</h2>
            <p><b>ReLU:</b> Used in hidden layers to learn non-linear patterns.</p>
            <p><b>Sigmoid:</b> Produces the probability of High-Value class.</p>
            <p><b>Adam:</b> Optimizer used for efficient training.</p>
            <p><b>Binary Cross Entropy:</b> Loss function for binary classification.</p>
            <p><b>EarlyStopping:</b> Stops training when validation loss stops improving.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "🤖 ML Prediction":

    st.markdown('<div class="section-title">🤖 Machine Learning Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="white-card">
        <h2>Transaction Value Classification</h2>
        <p>This module compares Logistic Regression and Random Forest
        for Normal-Value / High-Value transaction classification.</p>
    </div>
    """, unsafe_allow_html=True)

    ml_result = prepare_ml_data(cleaned_df, quantity_col, unit_price_col)

    if ml_result is None:
        st.error("Required Quantity and UnitPrice columns were not found.")
        st.stop()

    X, y, median_value = ml_result

    st.info(f"Transactions with value ≥ {median_value:.2f} are classified as High-Value.")

    m1,m2,m3 = st.columns(3)
    m1.metric("ML Records", f"{len(X):,}")
    m2.metric("Features", X.shape[1])
    m3.metric("Classes", 2)

    train_button = st.button("🚀 Train & Compare ML Models", type="primary", width="stretch")

    if train_button:
        with st.spinner("Training Logistic Regression and Random Forest..."):
            ml_result_model = train_ml_models(X,y)
        st.session_state.ml_result = ml_result_model
        st.session_state.ml_trained = True

    if st.session_state.ml_trained and "ml_result" in st.session_state:
        (
            results, logistic_model, random_forest_model, scaler,
            X_test, y_test, logistic_pred, rf_pred
        ) = st.session_state.ml_result

        st.success("✅ Machine Learning models trained successfully!")

        st.markdown('<div class="section-title">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

        display_results = results.copy()
        for col in ["Accuracy","Precision","Recall","F1 Score"]:
            display_results[col] = display_results[col].map(lambda x:f"{x:.2%}")

        st.dataframe(display_results, width="stretch", hide_index=True)

        best_idx = results["Accuracy"].idxmax()
        best_model = results.loc[best_idx,"Model"]
        best_accuracy = results.loc[best_idx,"Accuracy"]

        st.success(f"🏆 Best Model: {best_model} with {best_accuracy:.2%} accuracy.")

        # NEW interactive ML prediction
        st.markdown('<div class="section-title">🔮 Live ML Prediction</div>', unsafe_allow_html=True)

        i1,i2 = st.columns(2)
        with i1:
            pred_qty = st.number_input("🛒 Quantity", min_value=1.0, value=2.0, step=1.0, key="ml_qty")
        with i2:
            pred_price = st.number_input("💰 Unit Price", min_value=.01, value=10.0, step=.50, key="ml_price")

        if st.button("🔮 Predict with Both ML Models", type="primary", width="stretch", key="ml_predict"):
            sample = pd.DataFrame({"Quantity":[pred_qty],"UnitPrice":[pred_price]})

            lr_prob = float(logistic_model.predict_proba(
                scaler.transform(sample)
            )[0,1])
            lr_class = int(lr_prob >= .5)

            rf_prob = float(random_forest_model.predict_proba(sample)[0,1])
            rf_class = int(rf_prob >= .5)

            a,b = st.columns(2)
            with a:
                st.subheader("Logistic Regression")
                st.success("🟢 HIGH-VALUE" if lr_class else "🔵 NORMAL-VALUE")
                st.metric("High-Value Probability", f"{lr_prob:.2%}")

            with b:
                st.subheader("Random Forest")
                st.success("🟢 HIGH-VALUE" if rf_class else "🔵 NORMAL-VALUE")
                st.metric("High-Value Probability", f"{rf_prob:.2%}")

        st.markdown('<div class="section-title">🔍 Confusion Matrix</div>', unsafe_allow_html=True)

        cm1,cm2 = st.columns(2)

        with cm1:
            st.subheader("Logistic Regression")
            cm_lr = confusion_matrix(y_test, logistic_pred)
            st.dataframe(
                pd.DataFrame(
                    cm_lr,
                    index=["Actual Normal","Actual High-Value"],
                    columns=["Predicted Normal","Predicted High-Value"]
                ),
                width="stretch"
            )

        with cm2:
            st.subheader("Random Forest")
            cm_rf = confusion_matrix(y_test, rf_pred)
            st.dataframe(
                pd.DataFrame(
                    cm_rf,
                    index=["Actual Normal","Actual High-Value"],
                    columns=["Predicted Normal","Predicted High-Value"]
                ),
                width="stretch"
            )

        st.markdown("""
        <div class="white-card">
            <h2>📚 Model Explanation</h2>
            <p><b>Logistic Regression:</b> A linear classification model that estimates class probability.</p>
            <p><b>Random Forest:</b> An ensemble of decision trees that captures non-linear relationships.</p>
            <p><b>Features:</b> Quantity and Unit Price.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# DATASET
# ============================================================

elif page == "📄 Dataset":

    st.markdown('<div class="section-title">📄 Dataset Information</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="white-card">
        <h2>Online Retail Dataset</h2>
        <p><b>Total Rows:</b> {total_rows:,}</p>
        <p><b>Total Columns:</b> {total_columns}</p>
        <p><b>Unique Products:</b> {unique_products:,}</p>
        <p><b>Cleaned Records:</b> {valid_records:,}</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(20), width="stretch", hide_index=True)

    column_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [int(df[c].isnull().sum()) for c in df.columns],
        "Unique Values": [int(df[c].nunique()) for c in df.columns]
    })

    st.subheader("📊 Dataset Columns")
    st.dataframe(column_info, width="stretch", hide_index=True)

    st.subheader("⬇️ Downloads")

    cleaned_csv = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Cleaned Dataset",
        cleaned_csv,
        "cleaned_online_retail.csv",
        "text/csv"
    )

    # ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "🎓 About Project":

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.html("""
    <div class="hero">

        <h1>🎓 About This Project</h1>

        <p>
            Market Basket Analysis & Product Association System
        </p>

    </div>
    """)

    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.html("""
    <div class="white-card">

        <h2>🛒 Market Basket Analysis & Product Association System</h2>

        <p style="
            color:#475569;
            font-size:16px;
            line-height:1.7;
        ">
            This project analyzes retail transaction data to discover
            products frequently purchased together and to classify
            transaction value using Machine Learning and Deep Learning.
        </p>

    </div>
    """)

    # --------------------------------------------------------
    # ASSOCIATION RULE MINING
    # --------------------------------------------------------

    st.html("""
    <div class="section-title">
        🛍️ Association Rule Mining
    </div>

    <div class="white-card">

        <h2>🛍️ Apriori Algorithm</h2>

        <p>
            The Apriori algorithm is used to identify frequently
            purchased product combinations and generate association rules.
        </p>

        <div style="
            background:#eff6ff;
            padding:18px;
            border-radius:12px;
            border-left:5px solid #2563eb;
            margin-top:15px;
        ">

            <p>
                <b>🔹 Algorithm:</b> Apriori
            </p>

            <p>
                <b>🔹 Metrics:</b>
                Support • Confidence • Lift
            </p>

            <p>
                <b>🔹 Purpose:</b>
                Discover relationships between products purchased together.
            </p>

        </div>

    </div>
    """)

    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    st.html("""
    <div class="section-title">
        🤖 Machine Learning
    </div>

    <div class="white-card">

        <h2>🤖 Predictive Machine Learning</h2>

        <p>
            Machine Learning models are used to classify retail
            transactions into Normal-Value and High-Value categories.
        </p>

        <div style="
            background:#f8fafc;
            padding:20px;
            border-radius:12px;
            margin-top:15px;
        ">

            <p>
                <b>🔹 Algorithms:</b>
                Logistic Regression, Random Forest
            </p>

            <p>
                <b>🔹 Features:</b>
                Quantity, Unit Price
            </p>

            <p>
                <b>🔹 Target:</b>
                Transaction Value
            </p>

            <p>
                <b>🔹 Output:</b>
                Normal-Value / High-Value
            </p>

        </div>

    </div>
    """)

    # --------------------------------------------------------
    # DEEP LEARNING
    # --------------------------------------------------------

    st.html("""
    <div class="section-title">
        🧠 Deep Learning
    </div>

    <div class="white-card">

        <h2>🧠 Feedforward Neural Network</h2>

        <p>
            A Feedforward Neural Network is used as an advanced
            classification model for transaction value prediction.
        </p>

        <div style="
            background:linear-gradient(135deg,#eff6ff,#f8fafc);
            padding:22px;
            border-radius:14px;
            margin-top:18px;
            border:1px solid #dbeafe;
        ">

            <p>
                <b>🔹 Algorithm:</b>
                Feedforward Neural Network
            </p>

            <p>
                <b>🔹 Framework:</b>
                TensorFlow / Keras
            </p>

            <p>
                <b>🔹 Architecture:</b>
                2 → 64 → 32 → 1
            </p>

            <p>
                <b>🔹 Input Features:</b>
                Quantity, Unit Price
            </p>

            <p>
                <b>🔹 Activations:</b>
                ReLU + Sigmoid
            </p>

            <p>
                <b>🔹 Optimizer:</b>
                Adam
            </p>

            <p>
                <b>🔹 Output:</b>
                Normal-Value / High-Value
            </p>

        </div>

    </div>
    """)

    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.html("""
    <div class="section-title">
        💻 Technology Stack
    </div>

    <div class="white-card">

        <div style="
            display:flex;
            flex-wrap:wrap;
            gap:12px;
            margin-top:10px;
        ">

            <span style="
                background:#eff6ff;
                color:#1e40af;
                padding:10px 16px;
                border-radius:20px;
                font-weight:600;
            ">
                🐍 Python
            </span>

            <span style="
                background:#eff6ff;
                color:#1e40af;
                padding:10px 16px;
                border-radius:20px;
                font-weight:600;
            ">
                🐼 Pandas
            </span>

            <span style="
                background:#eff6ff;
                color:#1e40af;
                padding:10px 16px;
                border-radius:20px;
                font-weight:600;
            ">
                🔢 NumPy
            </span>

            <span style="
                background:#eff6ff;
                color:#1e40af;
                padding:10px 16px;
                border-radius:20px;
                font-weight:600;
            ">
                🤖 Scikit-learn
            </span>

            <span style="
                background:#eff6ff;
                color:#1e40af;
                padding:10px 16px;
                border-radius:20px;
                font-weight:600;
            ">
                🛍️ mlxtend
            </span>

            <span style="
                background:#eff6ff;
                color:#1e40af;
                padding:10px 16px;
                border-radius:20px;
                font-weight:600;
            ">
                🧠 TensorFlow
            </span>

            <span style="
                background:#eff6ff;
                color:#1e40af;
                padding:10px 16px;
                border-radius:20px;
                font-weight:600;
            ">
                ⚡ Keras
            </span>

            <span style="
                background:#eff6ff;
                color:#1e40af;
                padding:10px 16px;
                border-radius:20px;
                font-weight:600;
            ">
                🌐 Streamlit
            </span>

        </div>

    </div>
    """)

    # --------------------------------------------------------
    # PROJECT WORKFLOW
    # --------------------------------------------------------

    st.html("""
    <div class="section-title">
        ⚙️ Project Workflow
    </div>
    """)

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.html("""
        <div class="white-card"
             style="text-align:center; min-height:150px;">

            <div style="font-size:38px;">📄</div>

            <h3 style="color:#17346d;">
                Dataset
            </h3>

            <p style="color:#64748b;">
                Retail transaction data
            </p>

        </div>
        """)

    with f2:
        st.html("""
        <div class="white-card"
             style="text-align:center; min-height:150px;">

            <div style="font-size:38px;">🧹</div>

            <h3 style="color:#17346d;">
                Preprocessing
            </h3>

            <p style="color:#64748b;">
                Clean and prepare data
            </p>

        </div>
        """)

    with f3:
        st.html("""
        <div class="white-card"
             style="text-align:center; min-height:150px;">

            <div style="font-size:38px;">🛍️</div>

            <h3 style="color:#17346d;">
                Apriori
            </h3>

            <p style="color:#64748b;">
                Find product associations
            </p>

        </div>
        """)

    with f4:
        st.html("""
        <div class="white-card"
             style="text-align:center; min-height:150px;">

            <div style="font-size:38px;">🤖</div>

            <h3 style="color:#17346d;">
                Prediction
            </h3>

            <p style="color:#64748b;">
                ML & DL classification
            </p>

        </div>
        """)

    # --------------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------------

    st.html("""
    <div class="section-title">
        🎯 Project Objective
    </div>

    <div class="white-card"
         style="
            border-left:5px solid #2563eb;
            background:linear-gradient(135deg,#ffffff,#eff6ff);
         ">

        <h2>🎯 Objective</h2>

        <p style="
            color:#475569;
            font-size:16px;
            line-height:1.8;
        ">
            Identify product associations, generate intelligent
            product recommendations, compare predictive models,
            and provide an interactive analytics dashboard for
            retail transaction analysis.
        </p>

    </div>
    """)

    # --------------------------------------------------------
    # PROJECT SUMMARY
    # --------------------------------------------------------

    st.html("""
    <div class="section-title">
        🏆 Project Summary
    </div>

    <div class="white-card">

        <div style="
            background:#dcfce7;
            border:1px solid #bbf7d0;
            color:#166534;
            padding:18px;
            border-radius:12px;
            font-weight:600;
        ">

            ✅ Association Rule Mining completed using Apriori<br><br>

            ✅ Product recommendations generated using
            Support, Confidence and Lift<br><br>

            ✅ Machine Learning implemented using
            Logistic Regression and Random Forest<br><br>

            ✅ Deep Learning implemented using
            Feedforward Neural Network<br><br>

            ✅ Interactive Streamlit dashboard developed

        </div>

    </div>
    """)





# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
🛒 Market Basket Analysis | College Project |
Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)