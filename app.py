import streamlit as st 
import pandas as pd 
import numpy as np 
import os 
import glob 
import matplotlib.pyplot as plt 
 
 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input 
from tensorflow.keras.callbacks import EarlyStopping 
 
# Apriori libraries 
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import apriori, association_rules 
 
# Machine Learning libraries 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import ( 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix 
) 
 
 
# ============================================================ 
# PAGE CONFIGURATION 
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
 
    /* Main page */ 
    .stApp { 
        background-color: #f4f7fb; 
    } 
 
    /* Sidebar */ 
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #172554 0%, #1e3a8a 100%); 
    } 
 
    section[data-testid="stSidebar"] * { 
        color: white !important; 
    } 
 
    /* Sidebar navigation */ 
    section[data-testid="stSidebar"] div[role="radiogroup"] label { 
        font-size: 17px !important; 
        font-weight: 600 !important; 
        padding: 10px 8px !important; 
        margin: 4px 0 !important; 
        border-radius: 8px !important; 
    } 
 
    section[data-testid="stSidebar"] div[role="radiogroup"] label p { 
        font-size: 17px !important; 
        font-weight: 600 !important; 
    } 
 
    section[data-testid="stSidebar"] div[role="radiogroup"] { 
        gap: 6px !important; 
    } 
 
    /* Sidebar headings */ 
    section[data-testid="stSidebar"] h3 { 
        font-size: 18px !important; 
        font-weight: 700 !important; 
    } 
 
    /* Sidebar title */ 
    .sidebar-title { 
        font-size: 26px; 
        font-weight: 700; 
        margin-bottom: 5px; 
    } 
 
    .sidebar-subtitle { 
        font-size: 15px; 
        opacity: 0.9; 
        margin-bottom: 30px; 
    } 
 
    /* Page headings */ 
    .section-title { 
        font-size: 27px; 
        font-weight: 700; 
        color: #17346d; 
        margin-top: 20px; 
        margin-bottom: 18px; 
    } 
 
    /* Hero */ 
    .hero { 
        background: linear-gradient(135deg, #172554, #2563eb); 
        padding: 38px; 
        border-radius: 0px 0px 22px 22px; 
        color: white; 
        margin-bottom: 32px; 
        box-shadow: 0 10px 30px rgba(23,37,84,0.18); 
    } 
 
    .hero h1 { 
        font-size: 38px; 
        margin-bottom: 12px; 
        color: white; 
    } 
 
    .hero p { 
        font-size: 17px; 
        margin: 0; 
        color: white; 
    } 
 
    /* Metric cards */ 
    .metric-card { 
        background: white; 
        border-radius: 16px; 
        padding: 25px; 
        min-height: 170px; 
        border: 1px solid #e5e7eb; 
        box-shadow: 0 5px 18px rgba(15,23,42,0.07); 
        text-align: center; 
    } 
 
    .metric-title { 
        color: #64748b; 
        font-size: 14px; 
        font-weight: 700; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
        margin-bottom: 12px; 
    } 
 
    .metric-value { 
        color: #17346d; 
        font-size: 32px; 
        font-weight: 800; 
        margin-bottom: 8px; 
    } 
 
    .metric-description { 
        color: #64748b; 
        font-size: 13px; 
    } 
 
    /* White cards */ 
    .white-card { 
        background: white; 
        border-radius: 16px; 
        padding: 28px; 
        border: 1px solid #e5e7eb; 
        box-shadow: 0 5px 18px rgba(15,23,42,0.06); 
        margin-bottom: 20px; 
    } 
 
    .white-card h2 { 
        color: #1e293b; 
        margin-bottom: 12px; 
    } 
 
    /* Success box */ 
    .success-box { 
        background: #dcfce7; 
        border: 1px solid #bbf7d0; 
        color: #166534; 
        padding: 15px 18px; 
        border-radius: 10px; 
        margin: 10px 0; 
        font-weight: 600; 
    } 
 
    /* Info box */ 
    .info-box { 
        background: #eff6ff; 
        border: 1px solid #bfdbfe; 
        color: #1e40af; 
        padding: 16px; 
        border-radius: 10px; 
        margin: 10px 0; 
    } 
 
    /* Recommendation */ 
    .recommendation-card { 
        background: white; 
        border-radius: 14px; 
        padding: 20px; 
        border-left: 5px solid #2563eb; 
        box-shadow: 0 5px 15px rgba(15,23,42,0.07); 
        margin-bottom: 14px; 
    } 
 
    .recommendation-title { 
        color: #17346d; 
        font-size: 18px; 
        font-weight: 700; 
        margin-bottom: 12px; 
    } 
 
    .recommendation-metric { 
        display: inline-block; 
        margin-right: 20px; 
        color: #475569; 
        font-size: 14px; 
    } 
 
    .recommendation-metric strong { 
        color: #1e293b; 
    } 
 
    /* Footer */ 
    .footer { 
        text-align: center; 
        color: #64748b; 
        font-size: 13px; 
        padding: 30px 0; 
    } 
 
</style> 
""", unsafe_allow_html=True) 
 
 
# ============================================================ 
# FIND DATASET 
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
 
    if len(csv_files) > 0: 
        return csv_files[0] 
 
    return None 
 
 
# ============================================================ 
# LOAD DATASET 
# ============================================================ 
 
@st.cache_data 
def load_data(): 
 
    file_path = find_dataset() 
 
    if file_path is None: 
        return None, None 
 
    try: 
        df = pd.read_csv( 
            file_path, 
            encoding="ISO-8859-1" 
        ) 
    except Exception: 
        df = pd.read_csv(file_path) 
 
    return df, file_path 
 
 
df, file_path = load_data() 
 
 
# ============================================================ 
# SIDEBAR 
# ============================================================ 
 
with st.sidebar: 
 
    st.markdown(""" 
    <div class="sidebar-title"> 
        🛒 Market Basket 
    </div> 
 
    <div class="sidebar-subtitle"> 
        Analysis Dashboard 
    </div> 
    """, unsafe_allow_html=True) 
 
    st.markdown("### 📌 Navigation") 
 
    page = st.radio( 
        "Navigation", 
        [ 
            "🏠 Home", 
            "📊 Dashboard", 
            "🛍️ Recommendations", 
            "🧠 Deep Learning", 
            "🤖 ML Prediction", 
            "📄 Dataset", 
            "🎓 About Project" 
        ], 
        label_visibility="collapsed" 
    ) 
 
    st.markdown("---") 
 
    st.markdown("### ⚙️ Algorithms") 
 
    st.markdown(""" 
    <div style=" 
        background:#1d3b82; 
        padding:15px; 
        border-radius:10px; 
        margin-top:10px; 
        line-height:1.5; 
    "> 
        <b>Apriori Algorithm</b><br> 
        Association Rule Mining 
        <br><br> 
        <b>Logistic Regression</b><br> 
        Classification 
        <br><br> 
        <b>Random Forest</b><br> 
    Machine Learning Classification 
 
    <br><br> 
 
    <b>Feedforward Neural Network</b><br> 
    Deep Learning Classification 
 
    </div> 
    """, unsafe_allow_html=True) 
 
 
# ============================================================ 
# DATASET CHECK 
# ============================================================ 
 
if df is None: 
 
    st.error( 
        "Dataset not found. Please place your CSV file in the same folder as app.py." 
    ) 
 
    st.stop() 
 
 
# ============================================================ 
# STANDARDIZE COLUMN NAMES 
# ============================================================ 
 
df.columns = [str(col).strip() for col in df.columns] 
 
 
# ============================================================ 
# IDENTIFY COLUMNS 
# ============================================================ 
 
invoice_col = None 
stock_col = None 
description_col = None 
quantity_col = None 
customer_col = None 
unit_price_col = None 
 
for col in df.columns: 
 
    lower = col.lower().strip() 
 
    if lower in ["invoice", "invoiceno", "invoice no", "invoice_no"]: 
        invoice_col = col 
 
    elif lower in ["stockcode", "stock_code", "stock code"]: 
        stock_col = col 
 
    elif lower == "description": 
        description_col = col 
 
    elif lower == "quantity": 
        quantity_col = col 
 
    elif lower in ["customerid", "customer_id", "customer id"]: 
        customer_col = col 
 
    elif lower in ["unitprice", "unit_price", "unit price"]: 
        unit_price_col = col 
 
 
# ============================================================ 
# CHECK REQUIRED COLUMNS 
# ============================================================ 
 
if invoice_col is None or description_col is None: 
 
    st.error( 
        "Required columns not found. " 
        "The dataset must contain Invoice/InvoiceNo and Description columns." 
    ) 
 
    st.write("Available columns:") 
    st.write(df.columns.tolist()) 
 
    st.stop() 
 
 
# ============================================================ 
# BASIC DATA INFORMATION 
# ============================================================ 
 
total_rows = len(df) 
total_columns = len(df.columns) 
 
if description_col: 
    unique_products = df[description_col].nunique() 
else: 
    unique_products = 0 
 
missing_values = int(df.isnull().sum().sum()) 
 
 
# ============================================================ 
# DATA PREPARATION 
# ============================================================ 
 
@st.cache_data 
def prepare_data( 
    data, 
    invoice_column, 
    quantity_column, 
    description_column 
): 
 
    cleaned = data.copy() 
 
    # Remove cancelled invoices 
    if invoice_column in cleaned.columns: 
 
        cleaned[invoice_column] = cleaned[invoice_column].astype(str) 
 
        cleaned = cleaned[ 
            ~cleaned[invoice_column] 
            .str.upper() 
            .str.startswith("C") 
        ] 
 
    # Remove invalid quantities 
    if ( 
        quantity_column is not None 
        and quantity_column in cleaned.columns 
    ): 
 
        cleaned[quantity_column] = pd.to_numeric( 
            cleaned[quantity_column], 
            errors="coerce" 
        ) 
 
        cleaned = cleaned[ 
            cleaned[quantity_column] > 0 
        ] 
 
    # Remove missing descriptions 
    if description_column in cleaned.columns: 
 
        cleaned = cleaned[ 
            cleaned[description_column].notna() 
        ] 
 
        cleaned[description_column] = ( 
            cleaned[description_column] 
            .astype(str) 
            .str.strip() 
        ) 
 
        cleaned = cleaned[ 
            cleaned[description_column] != "" 
        ] 
 
    return cleaned 
 
 
cleaned_df = prepare_data( 
    df, 
    invoice_col, 
    quantity_col, 
    description_col 
) 
 
valid_records = len(cleaned_df) 
 
 
# ============================================================ 
# CREATE TRANSACTIONS 
# ============================================================ 
 
@st.cache_data 
def create_transactions( 
    data, 
    invoice_column, 
    description_column 
): 
 
    if invoice_column not in data.columns: 
        return [] 
 
    transactions = ( 
        data.groupby(invoice_column)[description_column] 
        .apply(lambda x: list(set(x.dropna()))) 
        .tolist() 
    ) 
 
    transactions = [ 
        transaction 
        for transaction in transactions 
        if len(transaction) > 0 
    ] 
 
    return transactions 
 
 
# ============================================================ 
# CREATE ONE-HOT BASKET 
# ============================================================ 
 
@st.cache_data 
def create_basket(transactions): 
 
    if not transactions: 
        return pd.DataFrame() 
 
    encoder = TransactionEncoder() 
 
    encoded_array = encoder.fit( 
        transactions 
    ).transform(transactions) 
 
    basket = pd.DataFrame( 
        encoded_array, 
        columns=encoder.columns_ 
    ) 
 
    return basket 
 
 
# ============================================================ 
# APRIORI 
# ============================================================ 
 
@st.cache_data 
def generate_rules( 
    basket, 
    min_support, 
    min_confidence 
): 
 
    if basket.empty: 
        return pd.DataFrame(), pd.DataFrame() 
 
    # Limit products for faster processing 
    if basket.shape[1] > 1500: 
 
        product_counts = basket.sum(axis=0) 
 
        top_products = product_counts.nlargest(1500).index 
 
        basket = basket[top_products] 
 
    # Apriori 
    frequent_itemsets = apriori( 
        basket, 
        min_support=max(min_support, 0.02), 
        use_colnames=True, 
        max_len=2, 
        low_memory=True 
    ) 
 
    if frequent_itemsets.empty: 
        return frequent_itemsets, pd.DataFrame() 
 
    # Association Rules 
    rules = association_rules( 
        frequent_itemsets, 
        metric="confidence", 
        min_threshold=min_confidence 
    ) 
 
    if not rules.empty: 
 
        rules = rules.sort_values( 
            by=["lift", "confidence"], 
            ascending=False 
        ) 
 
    return frequent_itemsets, rules 
 
 
# ============================================================ 
# MACHINE LEARNING DATA 
# ============================================================ 
 
@st.cache_data 
def prepare_ml_data(data): 
 
    ml_data = data.copy() 
 
    # Detect quantity 
    quantity_column = None 
    unitprice_column = None 
 
    for col in ml_data.columns: 
 
        lower = str(col).lower().strip() 
 
        if lower in [ 
            "quantity", 
            "qty" 
        ]: 
            quantity_column = col 
 
        if lower in [ 
            "unitprice", 
            "unit_price", 
            "unit price" 
        ]: 
            unitprice_column = col 
 
    if ( 
        quantity_column is None 
        or unitprice_column is None 
    ): 
        return None 
 
    # Convert numeric 
    ml_data[quantity_column] = pd.to_numeric( 
        ml_data[quantity_column], 
        errors="coerce" 
    ) 
 
    ml_data[unitprice_column] = pd.to_numeric( 
        ml_data[unitprice_column], 
        errors="coerce" 
    ) 
 
    # Remove invalid values 
    ml_data = ml_data[ 
        (ml_data[quantity_column] > 0) 
        & 
        (ml_data[unitprice_column] > 0) 
    ].copy() 
 
    if len(ml_data) < 20: 
        return None 
 
    # Transaction value 
    ml_data["TransactionValue"] = ( 
        ml_data[quantity_column] 
        * 
        ml_data[unitprice_column] 
    ) 
 
    # Median transaction value 
    median_value = ml_data[ 
        "TransactionValue" 
    ].median() 
 
    # Target 
    ml_data["HighValue"] = ( 
        ml_data["TransactionValue"] 
        >= median_value 
    ).astype(int) 
 
    # Features 
    X = ml_data[ 
        [ 
            quantity_column, 
            unitprice_column 
        ] 
    ].copy() 
 
    X.columns = [ 
        "Quantity", 
        "UnitPrice" 
    ] 
 
    y = ml_data["HighValue"] 
 
    return X, y, median_value 
 
 
# ============================================================ 
# TRAIN MACHINE LEARNING MODELS 
# ============================================================ 
 
@st.cache_data 
def train_ml_models(X, y): 
 
    # Split 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, 
        y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y 
    ) 
 
    # ======================================================== 
    # LOGISTIC REGRESSION 
    # ======================================================== 
 
    scaler = StandardScaler() 
 
    X_train_scaled = scaler.fit_transform(X_train) 
 
    X_test_scaled = scaler.transform(X_test) 
 
    logistic_model = LogisticRegression( 
        random_state=42, 
        max_iter=1000 
    ) 
 
    logistic_model.fit( 
        X_train_scaled, 
        y_train 
    ) 
 
    logistic_pred = logistic_model.predict( 
        X_test_scaled 
    ) 
 
    # ======================================================== 
    # RANDOM FOREST 
    # ======================================================== 
 
    random_forest_model = RandomForestClassifier( 
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1 
    ) 
 
    random_forest_model.fit( 
        X_train, 
        y_train 
    ) 
 
    rf_pred = random_forest_model.predict( 
        X_test 
    ) 
 
    # ======================================================== 
    # MODEL METRICS 
    # ======================================================== 
 
    results = pd.DataFrame({ 
 
        "Model": [ 
            "Logistic Regression", 
            "Random Forest" 
        ], 
 
        "Accuracy": [ 
            accuracy_score( 
                y_test, 
                logistic_pred 
            ), 
            accuracy_score( 
                y_test, 
                rf_pred 
            ) 
        ], 
 
        "Precision": [ 
            precision_score( 
                y_test, 
                logistic_pred, 
                zero_division=0 
            ), 
            precision_score( 
                y_test, 
                rf_pred, 
                zero_division=0 
            ) 
        ], 
 
        "Recall": [ 
            recall_score( 
                y_test, 
                logistic_pred, 
                zero_division=0 
            ), 
            recall_score( 
                y_test, 
                rf_pred, 
                zero_division=0 
            ) 
        ], 
 
        "F1 Score": [ 
            f1_score( 
                y_test, 
                logistic_pred, 
                zero_division=0 
            ), 
            f1_score( 
                y_test, 
                rf_pred, 
                zero_division=0 
            ) 
        ] 
    }) 
 
    return ( 
        results, 
        logistic_model, 
        random_forest_model, 
        scaler, 
        X_test, 
        y_test, 
        logistic_pred, 
        rf_pred 
    ) 
# ============================================================ 
# DEEP LEARNING DATA PREPARATION 
# ============================================================ 
 
@st.cache_data 
def prepare_dl_data(data): 
 
    dl_data = data.copy() 
 
    quantity_column = None 
    unitprice_column = None 
 
    for col in dl_data.columns: 
 
        lower = str(col).lower().strip() 
 
        if lower in ["quantity", "qty"]: 
            quantity_column = col 
 
        elif lower in [ 
            "unitprice", 
            "unit_price", 
            "unit price" 
        ]: 
            unitprice_column = col 
 
    if quantity_column is None or unitprice_column is None: 
        return None 
 
    dl_data[quantity_column] = pd.to_numeric( 
        dl_data[quantity_column], 
        errors="coerce" 
    ) 
 
    dl_data[unitprice_column] = pd.to_numeric( 
        dl_data[unitprice_column], 
        errors="coerce" 
    ) 
 
    dl_data = dl_data[ 
        (dl_data[quantity_column] > 0) & 
        (dl_data[unitprice_column] > 0) 
    ].copy() 
 
    if len(dl_data) < 50: 
        return None 
 
    dl_data["TransactionValue"] = ( 
        dl_data[quantity_column] * 
        dl_data[unitprice_column] 
    ) 
 
    median_value = dl_data[ 
        "TransactionValue" 
    ].median() 
 
    dl_data["HighValue"] = ( 
        dl_data["TransactionValue"] >= median_value 
    ).astype(int) 
 
    X = dl_data[ 
        [ 
            quantity_column, 
            unitprice_column 
        ] 
    ].copy() 
 
    X.columns = [ 
        "Quantity", 
        "UnitPrice" 
    ] 
 
    y = dl_data["HighValue"] 
 
    return X, y, median_value 
# ============================================================ 
# TRAIN DEEP LEARNING MODEL 
# ============================================================ 
 

 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, 
        y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y 
    ) 
 
    scaler = StandardScaler() 
 
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 
 
    # ======================================================== 
    # NEURAL NETWORK 
    # ======================================================== 
 
    model = Sequential([ 
        Input(shape=(X_train_scaled.shape[1],)), 
 
        Dense( 
            64, 
            activation="relu" 
        ), 
 
        Dense( 
            32, 
            activation="relu" 
        ), 
 
        Dense( 
            16, 
            activation="relu" 
        ), 
 
        Dense( 
            1, 
            activation="sigmoid" 
        ) 
    ]) 
 
    model.compile( 
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy"] 
    ) 
 
    early_stop = EarlyStopping( 
        monitor="val_loss", 
        patience=5, 
        restore_best_weights=True 
    ) 
 
    history = model.fit( 
        X_train_scaled, 
        y_train, 
        validation_split=0.20, 
        epochs=30, 
        batch_size=32, 
        callbacks=[early_stop], 
        verbose=0 
    ) 
 
    probabilities = model.predict( 
        X_test_scaled, 
        verbose=0 
    ) 
 
    predictions = ( 
        probabilities >= 0.5 
    ).astype(int).flatten() 
 
    accuracy = accuracy_score( 
        y_test, 
        predictions 
    ) 
 
    precision = precision_score( 
        y_test, 
        predictions, 
        zero_division=0 
    ) 
 
    recall = recall_score( 
        y_test, 
        predictions, 
        zero_division=0 
    ) 
 
    f1 = f1_score( 
        y_test, 
        predictions, 
        zero_division=0 
    ) 
 
    cm = confusion_matrix( 
        y_test, 
        predictions 
    ) 
 
    return ( 
        model, 
        scaler, 
        history, 
        X_test, 
        y_test, 
        predictions, 
        accuracy, 
        precision, 
        recall, 
        f1, 
        cm 
    ) 
if page == "🏠 Home": 
    # Hero section 
    st.html(""" 
    <div class="hero" style=" 
        background: linear-gradient(135deg, #1e3a8a, #2563eb); 
        padding: 50px; 
        border-radius: 0px 0px 25px 25px; 
        color: white; 
        text-align: center; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
    "> 
        <h1 style="font-size: 45px; margin-bottom: 15px;">🏠 Welcome To Market Basket Analysis</h1> 
        <p style="font-size: 20px; opacity: 0.9;"> 
            Explore Market Basket Analysis with insights, recommendations, and predictive models. 
        </p> 
    </div> 
    """) 
 
    # Feature cards 
    c1, c2, c3 = st.columns(3) 
 
    with c1: 
        st.html(""" 
        <div class="metric-card"> 
            <div class="metric-title">📊 Dashboard</div> 
            <div class="metric-value">Insights</div> 
            <div class="metric-description">Visualize dataset patterns</div> 
        </div> 
        """) 
 
    with c2: 
        st.html(""" 
        <div class="metric-card"> 
            <div class="metric-title">🛍️ Recommendations</div> 
            <div class="metric-value">Smart</div> 
            <div class="metric-description">Discover products bought together</div> 
        </div> 
        """) 
 
    with c3: 
        st.html(""" 
        <div class="metric-card"> 
            <div class="metric-title">🤖 ML Models</div> 
            <div class="metric-value">Predictive</div> 
            <div class="metric-description">Classify high-value transactions</div> 
        </div> 
        """) 
 
    # Footer 
    st.html(""" 
    <div class="footer"> 
        © 2026 nri | Market Basket Analysis Project 
    </div> 
    """) 
 
 
     
 
 
 
 
# ============================================================ 
# DASHBOARD PAGE 
# ============================================================ 
 
if page == "📊 Dashboard": 
 
    st.html(""" 
    <div class="hero"> 
 
        <h1> 
            🛒 Market Basket Analysis 
        </h1> 
 
        <p> 
            Discover purchasing patterns and recommend products 
            using Association Rule Mining. 
        </p> 
 
    </div> 
    """) 
 
    st.html(""" 
    <div class="section-title"> 
        📊 Dataset Overview 
    </div> 
    """) 
 
    c1, c2, c3, c4 = st.columns(4) 
 
    with c1: 
 
        st.html(f""" 
        <div class="metric-card"> 
 
            <div class="metric-title"> 
                Total Transactions 
            </div> 
 
            <div class="metric-value"> 
                {total_rows:,} 
            </div> 
 
            <div class="metric-description"> 
                Records in dataset 
            </div> 
 
        </div> 
        """) 
 
    with c2: 
 
        st.html(f""" 
        <div class="metric-card"> 
 
            <div class="metric-title"> 
                Dataset Columns 
            </div> 
 
            <div class="metric-value"> 
                {total_columns} 
            </div> 
 
            <div class="metric-description"> 
                Available attributes 
            </div> 
 
        </div> 
        """) 
 
    with c3: 
 
        st.html(f""" 
        <div class="metric-card"> 
 
            <div class="metric-title"> 
                Unique Products 
            </div> 
 
            <div class="metric-value"> 
                {unique_products:,} 
            </div> 
 
            <div class="metric-description"> 
                Different products 
            </div> 
 
        </div> 
        """) 
 
    with c4: 
 
        st.html(f""" 
        <div class="metric-card"> 
 
            <div class="metric-title"> 
                Missing Values 
            </div> 
 
            <div class="metric-value"> 
                {missing_values:,} 
            </div> 
 
            <div class="metric-description"> 
                Before preprocessing 
            </div> 
 
        </div> 
        """) 
 
    # Data preparation 
 
    st.html(""" 
    <div class="section-title"> 
        🧹 Data Preparation 
    </div> 
    """) 
 
    c1, c2 = st.columns(2) 
 
    with c1: 
 
        st.html(f""" 
        <div class="success-box"> 
            ✅ Valid records: {valid_records:,} 
        </div> 
        """) 
 
    with c2: 
 
        st.html(""" 
        <div class="success-box"> 
            ✅ Cancelled and invalid transactions removed 
        </div> 
        """) 
 
    # Basket 
 
    st.html(""" 
    <div class="section-title"> 
        🧺 Transaction Basket 
    </div> 
    """) 
 
    with st.spinner( 
        "Creating transaction basket..." 
    ): 
 
        transactions = create_transactions( 
            cleaned_df, 
            invoice_col, 
            description_col 
        ) 
 
    st.success( 
        f"Transaction basket created successfully: " 
        f"{len(transactions):,} transactions" 
    ) 
 
    if len(transactions) > 0: 
 
        with st.expander( 
            "📋 View sample transactions" 
        ): 
 
            for i, transaction in enumerate( 
                transactions[:10] 
            ): 
 
                st.write( 
                    f"**Transaction {i + 1}:** " 
                    + ", ".join(transaction[:10]) 
                ) 
 
 
# ============================================================ 
# RECOMMENDATIONS PAGE 
# ============================================================ 
 
elif page == "🛍️ Recommendations": 
 
    st.html(""" 
    <div class="section-title"> 
        🛍️ Product Recommendation 
    </div> 
    """) 
 
    st.html(""" 
    <div class="white-card"> 
 
        <h2> 
            Find Products Purchased Together 
        </h2> 
 
        <p> 
            Select a product below and use the Apriori algorithm 
            to discover products that are commonly purchased with it. 
        </p> 
 
    </div> 
    """) 
 
    with st.spinner( 
        "Preparing transaction basket..." 
    ): 
 
        transactions = create_transactions( 
            cleaned_df, 
            invoice_col, 
            description_col 
        ) 
 
    if not transactions: 
 
        st.error( 
            "Unable to create transaction basket." 
        ) 
 
        st.stop() 
 
    # Product list 
 
    all_products = sorted( 
        list( 
            set( 
                product 
                for transaction in transactions 
                for product in transaction 
            ) 
        ) 
    ) 
 
    if not all_products: 
 
        st.error( 
            "No products found in the dataset." 
        ) 
 
        st.stop() 
 
    selected_product = st.selectbox( 
        "🔎 Select a product", 
        all_products 
    ) 
 
    # Controls 
 
    col1, col2 = st.columns(2) 
 
    with col1: 
 
        min_support = st.slider( 
            "Minimum Support", 
            min_value=0.01, 
            max_value=0.10, 
            value=0.02, 
            step=0.01 
        ) 
 
    with col2: 
 
        min_confidence = st.slider( 
            "Minimum Confidence", 
            min_value=0.10, 
            max_value=0.90, 
            value=0.30, 
            step=0.05 
        ) 
 
    generate = st.button( 
        "🔍 Generate Product Recommendations", 
        type="primary", 
        width="stretch" 
    ) 
 
    if generate: 
 
        with st.spinner( 
            "Running Apriori algorithm... Please wait." 
        ): 
 
            basket = create_basket( 
                transactions 
            ) 
 
            frequent_itemsets, rules = generate_rules( 
                basket, 
                min_support, 
                min_confidence 
            ) 
 
        if rules.empty: 
 
            st.warning( 
                "No association rules found with the selected " 
                "support and confidence values." 
            ) 
 
            st.info( 
                "Try Minimum Support = 0.02 and " 
                "Minimum Confidence = 0.20." 
            ) 
 
        else: 
 
            selected_rules = rules[ 
                rules["antecedents"].apply( 
                    lambda x: 
                    selected_product in x 
                ) 
            ].copy() 
 
            if selected_rules.empty: 
 
                selected_rules = rules[ 
                    rules["consequents"].apply( 
                        lambda x: 
                        selected_product in x 
                    ) 
                ].copy() 
 
            if selected_rules.empty: 
 
                st.warning( 
                    f"No recommendations found for " 
                    f"'{selected_product}'. " 
                    f"Try another product or lower the thresholds." 
                ) 
 
            else: 
 
                selected_rules = selected_rules.sort_values( 
                    by=[ 
                        "confidence", 
                        "lift" 
                    ], 
                    ascending=False 
                ) 
 
                st.success( 
                    f"Recommendation analysis completed for " 
                    f"'{selected_product}'." 
                ) 
 
                st.html(""" 
                <div class="section-title"> 
                    🎯 Recommendation Results 
                </div> 
                """) 
 
                r1, r2, r3 = st.columns(3) 
 
                best_confidence = selected_rules[ 
                    "confidence" 
                ].max() 
 
                best_lift = selected_rules[ 
                    "lift" 
                ].max() 
 
                rule_count = len( 
                    selected_rules 
                ) 
 
                with r1: 
 
                    st.metric( 
                        "📌 Rules Found", 
                        rule_count 
                    ) 
 
                with r2: 
 
                    st.metric( 
                        "🎯 Best Confidence", 
                        f"{best_confidence:.2%}" 
                    ) 
 
                with r3: 
 
                    st.metric( 
                        "🚀 Best Lift", 
                        f"{best_lift:.2f}" 
                    ) 
 
                # Top recommendations 
 
                st.html(""" 
                <div class="section-title"> 
                    🏆 Top Recommendations 
                </div> 
                """) 
 
                top_rules = selected_rules.head(10) 
 
                recommendation_number = 1 
 
                for _, rule in top_rules.iterrows(): 
 
                    antecedents = list( 
                        rule["antecedents"] 
                    ) 
 
                    consequents = list( 
                        rule["consequents"] 
                    ) 
 
                    recommended_products = [ 
                        product 
                        for product in consequents 
                        if product != selected_product 
                    ] 
 
                    if not recommended_products: 
 
                        recommended_products = [ 
                            product 
                            for product in antecedents 
                            if product != selected_product 
                        ] 
 
                    if not recommended_products: 
                        continue 
 
                    recommended_product = ( 
                        recommended_products[0] 
                    ) 
 
                    support_value = rule["support"] 
                    confidence_value = rule["confidence"] 
                    lift_value = rule["lift"] 
 
                    st.html(f""" 
                    <div class="recommendation-card"> 
 
                        <div class="recommendation-title"> 
                            #{recommendation_number} 
                            🛍️ {recommended_product} 
                        </div> 
 
                        <div class="recommendation-metric"> 
                            <strong>Support:</strong> 
                            {support_value:.2%} 
                        </div> 
 
                        <div class="recommendation-metric"> 
                            <strong>Confidence:</strong> 
                            {confidence_value:.2%} 
                        </div> 
 
                        <div class="recommendation-metric"> 
                            <strong>Lift:</strong> 
                            {lift_value:.2f} 
                        </div> 
 
                    </div> 
                    """) 
 
                    recommendation_number += 1 
 
                # Rules table 
 
                st.html(""" 
                <div class="section-title"> 
                    📈 Association Rules 
                </div> 
                """) 
 
                display_rules = selected_rules.copy() 
 
                display_rules["Antecedent"] = ( 
                    display_rules["antecedents"] 
                    .apply( 
                        lambda x: 
                        ", ".join(list(x)) 
                    ) 
                ) 
 
                display_rules[ 
                    "Recommended Product" 
                ] = ( 
                    display_rules["consequents"] 
                    .apply( 
                        lambda x: 
                        ", ".join(list(x)) 
                    ) 
                ) 
 
                display_rules["Support"] = ( 
                    display_rules["support"] 
                    .map( 
                        lambda x: 
                        f"{x:.2%}" 
                    ) 
                ) 
 
                display_rules["Confidence"] = ( 
                    display_rules["confidence"] 
                    .map( 
                        lambda x: 
                        f"{x:.2%}" 
                    ) 
                ) 
 
                display_rules["Lift"] = ( 
                    display_rules["lift"] 
                    .map( 
                        lambda x: 
                        f"{x:.2f}" 
                    ) 
                ) 
 
                display_rules = display_rules[ 
                    [ 
                        "Antecedent", 
                        "Recommended Product", 
                        "Support", 
                        "Confidence", 
                        "Lift" 
                    ] 
                ].head(10) 
 
                st.dataframe( 
                    display_rules, 
                    width="stretch", 
                    hide_index=True 
                ) 
                # ============================================================ 
# DEEP LEARNING PAGE 
# ============================================================ 
 
elif page == "🧠 Deep Learning": 
 
    st.html(""" 
    <div class="hero"> 
        <h1>🧠 Deep Learning Prediction</h1> 
        <p> 
            Neural Network based transaction value classification 
            with performance graphs and confusion matrix. 
        </p> 
    </div> 
    """) 
 
    # -------------------------------------------------------- 
    # PREPARE DL DATA 
    # -------------------------------------------------------- 
 
    dl_result = prepare_dl_data(cleaned_df) 
 
    if dl_result is None: 
 
        st.error( 
            "Quantity and UnitPrice columns are required " 
            "for Deep Learning prediction." 
        ) 
 
        st.stop() 
 
    X_dl, y_dl, median_dl = dl_result 
 
    st.html(""" 
    <div class="section-title"> 
        📌 Deep Learning Model 
    </div> 
    """) 
 
    st.html(f""" 
    <div class="white-card"> 
 
        <h2>Feedforward Neural Network</h2> 
 
        <p> 
            The Deep Learning model uses a Feedforward Neural Network 
            to classify transactions into Normal-Value and High-Value 
            categories. 
        </p> 
 
        <p> 
            <b>Input Features:</b> 
            Quantity and Unit Price 
        </p> 
 
        <p> 
            <b>Target:</b> 
            High-Value Transaction 
        </p> 
 
        <p> 
            <b>Classification Threshold:</b> 
            {median_dl:.2f} 
        </p> 
 
    </div> 
    """) 
 
    # -------------------------------------------------------- 
    # MODEL ARCHITECTURE 
    # -------------------------------------------------------- 
 
    st.html(""" 
    <div class="section-title"> 
        🏗️ Neural Network Architecture 
    </div> 
    """) 
 
    a1, a2, a3, a4 = st.columns(4) 
 
    with a1: 
        st.metric("Input Layer", "2 Features") 
 
    with a2: 
        st.metric("Hidden Layer 1", "64 Neurons") 
 
    with a3: 
        st.metric("Hidden Layer 2", "32 Neurons") 
 
    with a4: 
        st.metric("Output Layer", "1 Neuron") 
 
    st.code(""" 
Input Layer 
     ↓ 
Dense Layer (64 neurons, ReLU) 
     ↓ 
Dense Layer (32 neurons, ReLU) 
     ↓ 
Output Layer (1 neuron, Sigmoid) 
     ↓ 
Normal / High-Value 
""") 
 
    # -------------------------------------------------------- 
    # TRAIN DL MODEL 
    # -------------------------------------------------------- 
 
    @st.cache_resource 
    def train_deep_learning_model(X, y): 
 
        X_train, X_test, y_train, y_test = train_test_split( 
            X, 
            y, 
            test_size=0.20, 
            random_state=42, 
            stratify=y 
        ) 
 
        scaler = StandardScaler() 
 
        X_train_scaled = scaler.fit_transform(X_train) 
        X_test_scaled = scaler.transform(X_test) 
 
        model = Sequential([ 
            Input(shape=(X_train_scaled.shape[1],)), 
            Dense(64, activation="relu"), 
            Dense(32, activation="relu"), 
            Dense(1, activation="sigmoid") 
        ]) 
 
        model.compile( 
            optimizer="adam", 
            loss="binary_crossentropy", 
            metrics=["accuracy"] 
        ) 
 
        history = model.fit( 
            X_train_scaled, 
            y_train, 
            validation_split=0.20, 
            epochs=15, 
            batch_size=32, 
            verbose=0 
        ) 
 
        probabilities = model.predict( 
            X_test_scaled, 
            verbose=0 
        ).ravel() 
 
        predictions = ( 
            probabilities >= 0.5 
        ).astype(int) 
 
        return ( 
            model, 
            scaler, 
            history, 
            X_test, 
            y_test, 
            predictions, 
            probabilities 
        ) 
 
    # -------------------------------------------------------- 
    # TRAIN BUTTON 
    # -------------------------------------------------------- 
 
    train_dl = st.button( 
        "🚀 Train Deep Learning Model", 
        type="primary", 
        width="stretch" 
    ) 
 
    if train_dl: 
 
        with st.spinner( 
            "Training Deep Learning Neural Network..." 
        ): 
 
            ( 
                dl_model, 
                dl_scaler, 
                dl_history, 
                X_test_dl, 
                y_test_dl, 
                dl_predictions, 
                dl_probabilities 
            ) = train_deep_learning_model( 
                X_dl, 
                y_dl 
            ) 
 
        st.success( 
            "✅ Deep Learning model trained successfully!" 
        ) 
 
        # ---------------------------------------------------- 
        # DL PERFORMANCE 
        # ---------------------------------------------------- 
 
        dl_accuracy = accuracy_score( 
            y_test_dl, 
            dl_predictions 
        ) 
 
        dl_precision = precision_score( 
            y_test_dl, 
            dl_predictions, 
            zero_division=0 
        ) 
 
        dl_recall = recall_score( 
            y_test_dl, 
            dl_predictions, 
            zero_division=0 
        ) 
 
        dl_f1 = f1_score( 
            y_test_dl, 
            dl_predictions, 
            zero_division=0 
        ) 
 
        st.html(""" 
        <div class="section-title"> 
            📊 Deep Learning Performance 
        </div> 
        """) 
 
        p1, p2, p3, p4 = st.columns(4) 
 
        with p1: 
            st.metric( 
                "Accuracy", 
                f"{dl_accuracy:.2%}" 
            ) 
 
        with p2: 
            st.metric( 
                "Precision", 
                f"{dl_precision:.2%}" 
            ) 
 
        with p3: 
            st.metric( 
                "Recall", 
                f"{dl_recall:.2%}" 
            ) 
 
        with p4: 
            st.metric( 
                "F1 Score", 
                f"{dl_f1:.2%}" 
            ) 
 
        # ---------------------------------------------------- 
        # TRAINING GRAPH 
        # ---------------------------------------------------- 
 
        st.html(""" 
        <div class="section-title"> 
            📈 Training Performance Graph 
        </div> 
        """) 
 
        history_df = pd.DataFrame({ 
            "Epoch": range( 
                1, 
                len( 
                    dl_history.history["accuracy"] 
                ) + 1 
            ), 
 
            "Training Accuracy": 
                dl_history.history["accuracy"], 
 
            "Validation Accuracy": 
                dl_history.history["val_accuracy"] 
        }) 
 
        st.line_chart( 
            history_df.set_index("Epoch") 
        ) 
 
        # ---------------------------------------------------- 
        # LOSS GRAPH 
        # ---------------------------------------------------- 
 
        st.html(""" 
        <div class="section-title"> 
            📉 Training & Validation Loss 
        </div> 
        """) 
 
        loss_df = pd.DataFrame({ 
 
            "Epoch": range( 
                1, 
                len( 
                    dl_history.history["loss"] 
                ) + 1 
            ), 
 
            "Training Loss": 
                dl_history.history["loss"], 
 
            "Validation Loss": 
                dl_history.history["val_loss"] 
        }) 
 
        st.line_chart( 
            loss_df.set_index("Epoch") 
        ) 
 
        # ---------------------------------------------------- 
        # CONFUSION MATRIX 
        # ---------------------------------------------------- 
 
        st.html(""" 
        <div class="section-title"> 
            🔍 Deep Learning Confusion Matrix 
        </div> 
        """) 
 
        dl_cm = confusion_matrix( 
            y_test_dl, 
            dl_predictions 
        ) 
 
        cm_df = pd.DataFrame( 
            dl_cm, 
            index=[ 
                "Actual Normal", 
                "Actual High-Value" 
            ], 
            columns=[ 
                "Predicted Normal", 
                "Predicted High-Value" 
            ] 
        ) 
 
        st.dataframe( 
            cm_df, 
            width="stretch" 
        ) 
 
        # ---------------------------------------------------- 
        # CONFUSION MATRIX GRAPH 
        # ---------------------------------------------------- 
 
        st.bar_chart( 
            cm_df 
        ) 
 
        # ---------------------------------------------------- 
        # PREDICTION INPUT 
        # ---------------------------------------------------- 
 
        st.html(""" 
        <div class="section-title"> 
            🔮 DL Transaction Prediction 
        </div> 
        """) 
 
        st.html(""" 
        <div class="white-card"> 
 
            <h2>Enter Transaction Details</h2> 
 
            <p> 
                Enter Quantity and Unit Price to predict whether 
                the transaction is Normal-Value or High-Value. 
            </p> 
 
        </div> 
        """) 
 
        input_col1, input_col2 = st.columns(2) 
 
        with input_col1: 
 
            input_quantity = st.number_input( 
                "🛒 Quantity", 
                min_value=1.0, 
                value=2.0, 
                step=1.0 
            ) 
 
        with input_col2: 
 
            input_unit_price = st.number_input( 
                "💰 Unit Price", 
                min_value=0.01, 
                value=10.0, 
                step=0.50 
            ) 
 
        transaction_value = ( 
            input_quantity * 
            input_unit_price 
        ) 
 
        st.info( 
            f"Transaction Value = " 
            f"{input_quantity:.2f} × " 
            f"{input_unit_price:.2f} = " 
            f"₹{transaction_value:.2f}" 
        ) 
 
        predict_dl = st.button( 
            "🔮 Predict Transaction", 
            type="primary", 
            width="stretch" 
        ) 
 
        if predict_dl: 
 
            input_data = np.array([ 
                [ 
                    input_quantity, 
                    input_unit_price 
                ] 
            ]) 
 
            input_scaled = dl_scaler.transform( 
                input_data 
            ) 
 
            probability = dl_model.predict( 
                input_scaled, 
                verbose=0 
            )[0][0] 
 
            prediction = ( 
                1 
                if probability >= 0.5 
                else 0 
            ) 
 
            # ----------------------------------------------- 
            # RESULT 
            # ----------------------------------------------- 
 
            st.html(""" 
            <div class="section-title"> 
                🎯 Prediction Result 
            </div> 
            """) 
 
            if prediction == 1: 
 
                st.success( 
                    "🟢 HIGH-VALUE TRANSACTION" 
                ) 
 
                st.metric( 
                    "High-Value Probability", 
                    f"{probability:.2%}" 
                ) 
 
            else: 
 
                st.info( 
                    "🔵 NORMAL-VALUE TRANSACTION" 
                ) 
 
                st.metric( 
                    "High-Value Probability", 
                    f"{probability:.2%}" 
                ) 
 
            # Probability bar 
 
            probability_df = pd.DataFrame({ 
                "Probability": [ 
                    probability 
                ] 
            }) 
 
            st.progress( 
                float(probability) 
            ) 
 
            st.write( 
                f"**Prediction Probability:** " 
                f"{probability:.2%}" 
            ) 
 
            st.write( 
                f"**Transaction Value:** " 
                f"₹{transaction_value:.2f}" 
            ) 
 
            st.write( 
                f"**Classification Threshold:** " 
                f"₹{median_dl:.2f}" 
            ) 
 
        # ---------------------------------------------------- 
        # MODEL EXPLANATION 
        # ---------------------------------------------------- 
 
        st.html(""" 
        <div class="section-title"> 
            📚 Deep Learning Explanation 
        </div> 
 
        <div class="white-card"> 
 
            <h2>Feedforward Neural Network</h2> 
 
            <p> 
                A Feedforward Neural Network is used to classify 
                transactions based on Quantity and Unit Price. 
            </p> 
 
            <br> 
 
            <h2>ReLU Activation</h2> 
 
            <p> 
                ReLU is used in the hidden layers to learn 
                non-linear relationships in the transaction data. 
            </p> 
 
            <br> 
 
            <h2>Sigmoid Activation</h2> 
 
            <p> 
                The output layer uses Sigmoid activation to 
                produce a probability between 0 and 1. 
            </p> 
 
            <br> 
 
            <h2>Optimizer</h2> 
 
            <p> 
                Adam optimizer is used for efficient neural 
                network training. 
            </p> 
 
            <br> 
 
            <h2>Loss Function</h2> 
 
            <p> 
                Binary Cross Entropy is used because this is 
                a binary classification problem. 
            </p> 
 
        </div> 
        """) 
                 
 
# ============================================================ 
# MACHINE LEARNING PAGE 
# ============================================================ 
 
elif page == "🤖 ML Prediction": 
 
    st.html(""" 
    <div class="section-title"> 
        🤖 Machine Learning Prediction 
    </div> 
    """) 
 
    st.html(""" 
    <div class="white-card"> 
 
        <h2> 
            Transaction Value Classification 
        </h2> 
 
        <p> 
            This Machine Learning module classifies transactions 
            into High-Value and Normal-Value categories. 
        </p> 
 
        <p> 
            Two Machine Learning algorithms are used: 
            <b>Logistic Regression</b> and 
            <b>Random Forest</b>. 
        </p> 
 
    </div> 
    """) 
 
    # Prepare ML data 
 
    ml_result = prepare_ml_data( 
        cleaned_df 
    ) 
 
    if ml_result is None: 
 
        st.error( 
            "Required Quantity and UnitPrice columns " 
            "were not found or there are not enough valid records." 
        ) 
 
        st.stop() 
 
    X, y, median_value = ml_result 
 
    st.info( 
        f"Transactions with value ≥ {median_value:.2f} " 
        "are classified as High-Value transactions." 
    ) 
 
    # Summary 
 
    m1, m2, m3 = st.columns(3) 
 
    with m1: 
 
        st.metric( 
            "ML Records", 
            f"{len(X):,}" 
        ) 
 
    with m2: 
 
        st.metric( 
            "Features", 
            X.shape[1] 
        ) 
 
    with m3: 
 
        st.metric( 
            "Classes", 
            2 
        ) 
 
    st.html(""" 
    <div class="section-title"> 
        🧠 Train Machine Learning Models 
    </div> 
    """) 
 
    train_button = st.button( 
        "🚀 Train & Compare ML Models", 
        type="primary", 
        width="stretch" 
    ) 
 
    if train_button: 
 
        with st.spinner( 
            "Training Logistic Regression and Random Forest..." 
        ): 
 
            ( 
                results, 
                logistic_model, 
                random_forest_model, 
                scaler, 
                X_test, 
                y_test, 
                logistic_pred, 
                rf_pred 
            ) = train_ml_models( 
                X, 
                y 
            ) 
 
        st.success( 
            "Machine Learning models trained successfully!" 
        ) 
 
        # Model comparison 
 
        st.html(""" 
        <div class="section-title"> 
            📊 Model Performance Comparison 
        </div> 
        """) 
 
        display_results = results.copy() 
 
        for col in [ 
            "Accuracy", 
            "Precision", 
            "Recall", 
            "F1 Score" 
        ]: 
 
            display_results[col] = ( 
                display_results[col] 
                .map( 
                    lambda x: 
                    f"{x:.2%}" 
                ) 
            ) 
 
        st.dataframe( 
            display_results, 
            width="stretch", 
            hide_index=True 
        ) 
 
        # Best model 
 
        best_model = results.loc[ 
            results["Accuracy"].idxmax(), 
            "Model" 
        ] 
 
        best_accuracy = results[ 
            "Accuracy" 
        ].max() 
 
        st.success( 
            f"🏆 Best Model: {best_model} " 
            f"with {best_accuracy:.2%} accuracy." 
        ) 
 
        # Confusion matrices 
 
        st.html(""" 
        <div class="section-title"> 
            🔍 Confusion Matrix 
        </div> 
        """) 
 
        cm1, cm2 = st.columns(2) 
 
        with cm1: 
 
            st.subheader( 
                "Logistic Regression" 
            ) 
 
            logistic_cm = confusion_matrix( 
                y_test, 
                logistic_pred 
            ) 
 
            logistic_cm_df = pd.DataFrame( 
                logistic_cm, 
                index=[ 
                    "Actual Normal", 
                    "Actual High-Value" 
                ], 
                columns=[ 
                    "Predicted Normal", 
                    "Predicted High-Value" 
                ] 
            ) 
 
            st.dataframe( 
                logistic_cm_df, 
                width="stretch" 
            ) 
 
        with cm2: 
 
            st.subheader( 
                "Random Forest" 
            ) 
 
            rf_cm = confusion_matrix( 
                y_test, 
                rf_pred 
            ) 
 
            rf_cm_df = pd.DataFrame( 
                rf_cm, 
                index=[ 
                    "Actual Normal", 
                    "Actual High-Value" 
                ], 
                columns=[ 
                    "Predicted Normal", 
                    "Predicted High-Value" 
                ] 
            ) 
 
            st.dataframe( 
                rf_cm_df, 
                width="stretch" 
            ) 
 
        # Model explanation 
 
        st.html(""" 
        <div class="section-title"> 
            📚 Model Explanation 
        </div> 
 
        <div class="white-card"> 
 
            <h2> 
                Logistic Regression 
            </h2> 
 
            <p> 
                Logistic Regression is a classification algorithm 
                used to predict whether a transaction belongs to 
                the High-Value or Normal-Value class. 
            </p> 
 
            <br> 
 
            <h2> 
                Random Forest 
            </h2> 
 
            <p> 
                Random Forest is an ensemble classification algorithm 
                that combines multiple decision trees to improve 
                prediction performance. 
            </p> 
 
            <br> 
 
            <h2> 
                Features Used 
            </h2> 
 
            <p> 
                • Quantity<br> 
                • Unit Price 
            </p> 
 
        </div> 
        """) 
 
 
# ============================================================ 
# DATASET PAGE 
# ============================================================ 
 
elif page == "📄 Dataset": 
 
    st.html(""" 
    <div class="section-title"> 
        📄 Dataset Information 
    </div> 
    """) 
 
    st.html(f""" 
    <div class="white-card"> 
 
        <h2> 
            Online Retail Dataset 
        </h2> 
 
        <p> 
            This dataset contains customer transaction records 
            used for Market Basket Analysis. 
        </p> 
 
        <p> 
            <b>Total Rows:</b> {total_rows:,} 
        </p> 
 
        <p> 
            <b>Total Columns:</b> {total_columns} 
        </p> 
 
        <p> 
            <b>Unique Products:</b> {unique_products:,} 
        </p> 
 
    </div> 
    """) 
 
    st.subheader( 
        "📋 Dataset Preview" 
    ) 
 
    st.dataframe( 
        df.head(20), 
        width="stretch", 
        hide_index=True 
    ) 
 
    st.subheader( 
        "📊 Dataset Columns" 
    ) 
 
    column_info = pd.DataFrame({ 
 
        "Column Name": df.columns, 
 
        "Data Type": [ 
            str(df[col].dtype) 
            for col in df.columns 
        ], 
 
        "Missing Values": [ 
            int( 
                df[col].isnull().sum() 
            ) 
            for col in df.columns 
        ] 
    }) 
 
    st.dataframe( 
        column_info, 
        width="stretch", 
        hide_index=True 
    ) 
 
 
# ============================================================ 
# ABOUT PROJECT 
# ============================================================ 
 
elif page == "🎓 About Project": 
 
    st.html(""" 
    <div class="section-title"> 
        🎓 About This Project 
    </div> 
 
    <div class="white-card"> 
 
        <h2> 
            Market Basket Analysis Using Association Rule Mining 
        </h2> 
 
        <p> 
            This project analyzes customer transaction data to 
            discover relationships between products purchased together. 
        </p> 
 
        <br> 
 
        <p> 
            <b>Association Rule Mining Algorithm:</b> 
            Apriori 
        </p> 
 
        <br> 
 
        <p> 
            <b>Association Rule Metrics:</b> 
            Support, Confidence and Lift 
        </p> 
 
        <br> 
 
        <p> 
            <b>Machine Learning Algorithms:</b> 
            Logistic Regression and Random Forest 
        </p> 
        <br> 
 
<p> 
    <b>Deep Learning Algorithm:</b> 
    Feedforward Neural Network (FNN) 
</p> 
 
<br> 
 
<p> 
    <b>Deep Learning Framework:</b> 
     TensorFlow / Keras
</p> 
 
<br> 
 
<p> 
    <b>Deep Learning Features:</b> 
    Quantity and Unit Price 
</p> 
<p> 
    <b>Deep Learning Output:</b> 
    Normal-Value or High-Value Transaction 
</p> 
 
        <br> 
 
        <p> 
            <b>Programming Language:</b> 
            Python 
        </p> 
 
        <br> 
 
        <p> 
            <b>Web Framework:</b> 
            Streamlit 
        </p> 
 
        <br> 
 
        <p> 
           <b>Project Objective:</b> 
    Identify frequently purchased product combinations, 
    generate product recommendations, classify transaction 
    values using Machine Learning, and predict transaction 
    value categories using a Deep Learning Neural Network. 
 
    </div> 
    """) 
 
 
# ============================================================ 
# FOOTER 
# ============================================================ 
 
st.html(""" 
<div class="footer"> 
    🛒 Market Basket Analysis | 
    College Project | 
    Built with Python & Streamlit 
</div> 
""")      
