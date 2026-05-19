import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Churn Prediction", layout="wide")

# ---------------------- UI ----------------------
st.markdown("""
<style>
.main { background-color: #0f172a; }
h1, h2, h3 { color: #e2e8f0; }
.stMetric {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:#6366f1;'>💡 Customer Churn Prediction System</h1>
<p style='text-align:center; color:gray;'>End-to-End ML Pipeline (Train → Predict → Insights)</p>
<hr>
""", unsafe_allow_html=True)

# ---------------------- FUNCTIONS ----------------------

def clean_data(df):
    df.columns = df.columns.str.strip().str.replace(" ", "")

    for col in df.columns:
        if 'id' in col.lower():
            df = df.drop(col, axis=1)

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    return df.drop_duplicates()


# 🔥 FIXED: NO ROW DELETION HERE
def prepare_input(df, model_columns):
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip().str.replace(" ", "")

    df_encoded = pd.get_dummies(df_clean)
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

    return df_final


def retention_strategy(score):
    if score > 70:
        return "🔥 High Risk"
    elif score > 40:
        return "⚠️ Medium Risk"
    return "✅ Low Risk"


def detect_target_column(df):
    for col in df.columns:
        if col.lower() == "churn":
            return col
    return None


# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_assets():
    try:
        return joblib.load('churn_model.pkl'), joblib.load('model_columns.pkl')
    except:
        return None, None

model, model_columns = load_assets()

# ---------------------- SIDEBAR ----------------------
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio("", ["Train Model", "Predict", "Clean Data", "Insights"])

# ---------------------- GLOBAL DATA ----------------------
st.sidebar.markdown("### 📂 Upload Dataset (One-Time)")
file = st.sidebar.file_uploader("Upload CSV/XLSX", type=['csv','xlsx'])

if file:
    df_global = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    st.session_state['data'] = df_global
    st.sidebar.success("✅ Dataset Loaded")

df = st.session_state.get('data', None)


# ====================== TRAIN ======================
if page == "Train Model":
    st.subheader("Train Model")

    if df is not None:
        st.dataframe(df.head())

        df_cleaned = clean_data(df)

        target_col = detect_target_column(df_cleaned)

        if target_col is None:
            st.error("❌ No churn column found")
            st.stop()

        st.write("Target column detected:", target_col)

        df_cleaned[target_col] = df_cleaned[target_col].astype(str).str.strip().str.lower()
        df_cleaned[target_col] = df_cleaned[target_col].map({
            'yes': 1, 'no': 0,
            '1': 1, '0': 0
        })

        if df_cleaned[target_col].isnull().any():
            st.error("❌ Invalid values in churn column")
            st.write("Values:", df[target_col].unique())
            st.stop()

        X = pd.get_dummies(df_cleaned.drop(target_col, axis=1))
        y = df_cleaned[target_col]

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)

        st.success("Model trained successfully")
        st.metric("Accuracy", f"{acc:.2f}")

        from sklearn.metrics import confusion_matrix, classification_report
        y_pred = model.predict(X_test)
        st.markdown("### 📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.markdown("### 📊 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        joblib.dump(model, "churn_model.pkl")
        joblib.dump(X.columns, "model_columns.pkl")

        st.success("Model Saved")

    else:
        st.warning("Upload dataset first")


# ====================== PREDICT ======================
elif page == "Predict":
    st.subheader("📈 Churn Prediction")

    if df is not None and model is not None:
        df_final = prepare_input(df, model_columns)

        preds = model.predict(df_final)
        probs = model.predict_proba(df_final)[:, 1]

        # ✅ SAFE ASSIGNMENT
        df_result = df.copy()

        df_result['Prediction'] = ["Yes" if p == 1 else "No" for p in preds]
        df_result['Risk (%)'] = (probs * 100).round(2)
        df_result['Strategy'] = df_result['Risk (%)'].apply(retention_strategy)

        st.markdown("## 📊 Summary")

        total = len(df_result)
        churned = (df_result['Prediction'] == 'Yes').sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Churn", churned)
        c3.metric("Avg Risk", f"{df_result['Risk (%)'].mean():.2f}%")

        df_result['Segment'] = pd.cut(df_result['Risk (%)'], [0,30,70,100], labels=["Low","Medium","High"])

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.pie(df_result, names='Segment'))
        col2.plotly_chart(px.histogram(df_result, x='Risk (%)'))

        st.markdown("## 🚨 Top Risk Customers")
        st.dataframe(df_result.sort_values(by='Risk (%)', ascending=False).head(10))

    elif model is None:
        st.warning("⚠️ Train model first")

    else:
        st.warning("⚠️ Upload dataset first")


# ====================== CLEAN DATA ======================
elif page == "Clean Data":
    st.subheader("Data Cleaning model is in building stage")

    if df is not None:
        col1, col2 = st.columns(2)
        col1.dataframe(df.head())
        col2.dataframe(clean_data(df).head())
    else:
        st.warning("Upload dataset first")


# ====================== INSIGHTS ======================
elif page == "Insights":
    st.subheader("📊 Data Insights")

    if df is not None:
        st.dataframe(df.head())

        cat_cols = df.select_dtypes(include='object').columns.tolist()

        if not cat_cols:
            st.warning("No categorical columns")
            st.stop()

        feature = st.selectbox("Select Feature", cat_cols)

        if any(col.lower() == 'churn' for col in df.columns):
            churn_col = [col for col in df.columns if col.lower() == 'churn'][0]
            st.plotly_chart(px.histogram(df, x=feature, color=churn_col))
        else:
            st.plotly_chart(px.histogram(df, x=feature))
    else:
        st.warning("Upload dataset first")