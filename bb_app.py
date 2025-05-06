###############################################################################
# BANGLADESH BANK ML APPLICATION
# 
# A comprehensive machine learning application for Bangladesh Bank that addresses
# various analytical needs:
# - Financial Fraud Detection
# - Remittance Inflow Forecasting
# - Credit Risk Modeling and NPL Management
# - Economic Forecasting and Stress Testing
# - AML/CFT Pattern Analysis
#
# This application provides an intuitive interface for bank analysts to leverage
# machine learning techniques without requiring programming expertise.
###############################################################################

###############################################################################
# SECTION 1: LIBRARY IMPORTS AND INITIAL SETUP
###############################################################################

# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datetime
import pathlib
import warnings
import os
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ML models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier, IsolationForest
)
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, OneClassSVM

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm

###############################################################################
# SECTION 2: APPLICATION CONFIG AND SETUP
###############################################################################

def setup_app():
    """Configures the Streamlit application and sets initial parameters."""
    # Configure page settings
    st.set_page_config(
        page_title="Bangladesh Bank ML Application",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Application modes/use cases
    MODES = {
        "financial_fraud": "Financial Fraud Detection",
        "remittance_forecast": "Remittance Inflow Forecasting",
        "credit_risk": "Credit Risk Modeling and NPL Management",
        "economic_forecast": "Economic Forecasting and Stress Testing",
        "aml_cft": "AML/CFT Pattern Analysis"
    }
    
    return MODES

###############################################################################
# SECTION 3: SESSION STATE MANAGEMENT
###############################################################################

def initialize_session_state():
    """Initialize and manage all session state variables for the application."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'categorical_features' not in st.session_state:
        st.session_state.categorical_features = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'best_model_name' not in st.session_state:
        st.session_state.best_model_name = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = "regression"  # Default model type

###############################################################################
# SECTION 4: AUTHENTICATION SYSTEM
###############################################################################

def check_password():
    """
    Provides user authentication functionality.
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    def password_entered():
        """Validates entered username and password."""
        if st.session_state["username"] in st.secrets.get("credentials", {}):
            if st.session_state["password"] == st.secrets["credentials"][st.session_state["username"]]:
                st.session_state.authenticated = True
                st.session_state.current_user = st.session_state["username"]
                # Remove password from session state for security
                del st.session_state["password"]
                return True
            else:
                st.session_state.authenticated = False
                st.error("Password incorrect")
                return False
        else:
            st.session_state.authenticated = False
            st.error("User not found")
            return False

    # Display login form if not authenticated
    if not st.session_state.authenticated:
        # Show logo and bank name
        st.image("https://www.bb.org.bd/aboutus/logo/header_logo.png", width=200)
        st.title("Bangladesh Bank ML Application")
        st.caption("Secure Authentication Required")
        
        # Create login form with username and password fields
        with st.form("login_form"):
            st.text_input("Username", key="username", 
                         help="Enter your Bangladesh Bank username")
            st.text_input("Password", type="password", key="password",
                         help="Enter your password")
            submit = st.form_submit_button("Login")
            
            # Check credentials when form is submitted
            if submit:
                password_entered()

    # Return current authentication status
    return st.session_state.authenticated

###############################################################################
# SECTION A: FINANCIAL FRAUD DETECTION MODULE
###############################################################################

def fraud_detection_app():
    """
    Provides a module for detecting financial fraud using machine learning.
    The module supports both supervised learning with labeled data and
    unsupervised anomaly detection approaches.
    """
    st.header("Financial Fraud Detection")
    st.info("Upload transaction data to train a fraud detection model. "
            "This will help identify suspicious transactions based on historical patterns.")
    
    # Special options for fraud detection
    col1, col2 = st.columns(2)
    
    with col1:
        detection_approach = st.radio(
            "Choose detection approach:",
            ["Supervised Learning (Labeled Data)", "Anomaly Detection (Unlabeled Data)"]
        )
    
    with col2:
        alert_threshold = st.slider(
            "Alert Threshold (Precision-Recall tradeoff)",
            min_value=0.5,
            max_value=0.99,
            value=0.85,
            help="Higher values reduce false positives but may miss some fraud"
        )
    
    # File uploader for transaction data
    uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")
    
    if uploaded_file is not None:
        # Read the transaction data
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Display basic data info
            st.write("Transaction Data Overview:")
            st.dataframe(df.head())
            
            # Show data statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("Data Shape:", df.shape)
            with col2:
                st.write("Column Types:", df.dtypes.astype(str).value_counts())
            
            # Feature selection for fraud detection
            st.subheader("Feature Selection")
            
            # Depending on the approach, select target or not
            if detection_approach == "Supervised Learning (Labeled Data)":
                target_col = st.selectbox(
                    "Select fraud label column (1=fraud, 0=normal):",
                    df.columns
                )
                st.session_state.target = target_col
                st.session_state.model_type = "classification"
                
                # Check class balance
                if target_col in df.columns:
                    fraud_count = df[target_col].sum()
                    total_count = len(df)
                    fraud_percentage = (fraud_count / total_count) * 100
                    
                    st.write(f"Fraud transactions: {fraud_count} ({fraud_percentage:.2f}%)")
                    st.write(f"Normal transactions: {total_count - fraud_count} ({100-fraud_percentage:.2f}%)")
                    
                    # Warning for imbalanced dataset
                    if fraud_percentage < 5:
                        st.warning("Highly imbalanced dataset detected. Consider using specialized techniques.")
            else:
                # For anomaly detection, no target column is needed
                st.info("Anomaly detection will identify unusual transactions without labeled data")
                st.session_state.model_type = "anomaly_detection"
            
            # Select transaction ID column
            id_column = st.selectbox(
                "Select transaction ID column:",
                df.columns
            )
            st.session_state.id_column = id_column
            
            # Select features for fraud detection model
            exclude_cols = [id_column]
            if detection_approach == "Supervised Learning (Labeled Data)":
                exclude_cols.append(target_col)
                
            available_features = [col for col in df.columns if col not in exclude_cols]
            selected_features = st.multiselect(
                "Select features for fraud detection model:",
                available_features,
                default=available_features
            )
            
            st.session_state.features = selected_features
            
            # Identify categorical features automatically
            categorical_features = [f for f in selected_features if df[f].dtype == 'object' or df[f].dtype.name == 'category']
            st.session_state.categorical_features = categorical_features
            
            if categorical_features:
                st.write("Categorical features detected:", categorical_features)
            
            # Train fraud detection model button
            if st.button("Train Fraud Detection Model"):
                with st.spinner("Training model..."):
                    if detection_approach == "Supervised Learning (Labeled Data)":
                        # Train supervised fraud detection model
                        train_supervised_fraud_model(df, selected_features, target_col, id_column)
                    else:
                        # Train anomaly detection model
                        train_anomaly_detection_model(df, selected_features, id_column, alert_threshold)
        
        except Exception as e:
            st.error(f"Error processing transaction data: {str(e)}")
            st.info("Please check your data format and try again.")

def train_supervised_fraud_model(df, features, target, id_col):
    """
    Trains a supervised fraud detection model using labeled data.
    
    Args:
        df: DataFrame containing transaction data
        features: List of features to use for prediction
        target: Target column with fraud labels
        id_col: Column containing transaction IDs
    """
    # Prepare the dataset
    X = df[features].copy()
    y = df[target].copy()
    
    # Identify categorical features
    categorical_features = [f for f in features if df[f].dtype == 'object' or df[f].dtype.name == 'category']
    numerical_features = [f for f in features if f not in categorical_features]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Define models to evaluate
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    # Train and evaluate models
    results = []
    
    for name, model in models.items():
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': auc,
            'Pipeline': pipeline
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Pipeline'} for r in results])
    
    # Display results
    st.subheader("Fraud Detection Model Evaluation")
    st.dataframe(results_df)
    
    # Find best model based on AUC
    best_model_idx = results_df['AUC'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_pipeline = [r['Pipeline'] for r in results if r['Model'] == best_model_name][0]
    
    # Store best model in session state
    st.session_state.best_model = best_pipeline
    st.session_state.best_model_name = best_model_name
    
    # Display confusion matrix
    y_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    st.subheader(f"Confusion Matrix - {best_model_name}")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    ax.set_xticklabels(['Normal', 'Fraud'])
    ax.set_yticklabels(['Normal', 'Fraud'])
    st.pyplot(fig)
    
    # Display feature importance if applicable
    if hasattr(best_pipeline[-1], 'feature_importances_'):
        st.subheader("Feature Importance")
        
        # Extract feature names after preprocessing
        preprocessor = best_pipeline.named_steps['preprocessor']
        model = best_pipeline.named_steps['model']
        
        # For numericals: keep the same names
        # For categoricals: get one-hot encoded feature names
        feature_names = []
        
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                # Numerical features keep their names
                feature_names.extend(cols)
            elif name == 'cat':
                # Get one-hot encoded feature names for categorical features
                # This requires first getting the categories from the trained OneHotEncoder
                onehot = transformer.named_steps['onehot']
                for i, col in enumerate(cols):
                    categories = onehot.categories_[i]
                    feature_names.extend([f"{col}_{cat}" for cat in categories])
        
        # Get feature importances from the model
        importances = model.feature_importances_
        
        # Create a DataFrame for feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],  # Ensure matching lengths
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax)
        ax.set_title('Top 15 Features for Fraud Detection')
        st.pyplot(fig)
    
    # Create fraud scoring functionality
    st.subheader("Fraud Risk Scoring")
    st.info("Use the trained model to score new transactions for fraud risk")
    
    # Sample of transactions to score
    st.write("Sample transactions from test data:")
    sample_df = X_test.head(5).copy()
    sample_df[id_col] = df[id_col].iloc[:5].values  # Add ID column
    st.dataframe(sample_df)
    
    # Score the sample transactions
    if st.button("Score Sample Transactions"):
        with st.spinner("Scoring transactions..."):
            # Predict probabilities
            risk_scores = best_pipeline.predict_proba(sample_df[features])[:, 1]
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Transaction ID': sample_df[id_col],
                'Fraud Risk Score': risk_scores,
                'Risk Level': ['High' if score > 0.75 else 'Medium' if score > 0.5 else 'Low' for score in risk_scores]
            })
            
            # Display results
            st.write("Fraud Risk Assessment:")
            st.dataframe(results.style.background_gradient(subset=['Fraud Risk Score'], cmap='YlOrRd'))
            
            # Plot risk scores
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=results['Transaction ID'].astype(str), y=results['Fraud Risk Score'], ax=ax)
            ax.set_title('Fraud Risk Scores for Sample Transactions')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.axhline(y=0.75, color='red', linestyle='--', label='High Risk Threshold')
            ax.axhline(y=0.5, color='orange', linestyle='--', label='Medium Risk Threshold')
            ax.legend()
            st.pyplot(fig)

def train_anomaly_detection_model(df, features, id_col, threshold=0.85):
    """
    Trains an anomaly detection model for identifying unusual transactions.
    
    Args:
        df: DataFrame containing transaction data
        features: List of features to use for anomaly detection
        id_col: Column containing transaction IDs
        threshold: Percentile threshold for anomaly detection (0.85 = top 15% most unusual)
    """
    # Prepare the dataset
    X = df[features].copy()
    
    # Identify categorical features
    categorical_features = [f for f in features if df[f].dtype == 'object' or df[f].dtype.name == 'category']
    numerical_features = [f for f in features if f not in categorical_features]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)
    
    # Train multiple anomaly detection models
    # 1. Isolation Forest
    iso_forest = IsolationForest(contamination=1.0-threshold, random_state=42)
    iso_forest.fit(X_processed)
    
    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=1.0-threshold, novelty=True)
    lof.fit(X_processed)
    
    # 3. One-Class SVM (if dataset is not too large)
    if X_processed.shape[0] < 10000:  # Only use for smaller datasets
        ocsvm = OneClassSVM(nu=1.0-threshold, kernel='rbf', gamma='scale')
        ocsvm.fit(X_processed)
        svm_used = True
    else:
        ocsvm = None
        svm_used = False
    
    # Score anomalies with each model
    # For Isolation Forest: lower scores are more anomalous
    iso_scores = -iso_forest.score_samples(X_processed)
    
    # For LOF: negative of the decision function gives anomaly score
    lof_scores = -lof.decision_function(X_processed)
    
    # For One-Class SVM (if used)
    if svm_used:
        svm_scores = -ocsvm.decision_function(X_processed)
    else:
        svm_scores = np.zeros_like(iso_scores)
    
    # Create ensemble score (average of normalized scores)
    # First, normalize each score set
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
    
    if svm_used:
        svm_scores_norm = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())
        ensemble_scores = (iso_scores_norm + lof_scores_norm + svm_scores_norm) / 3
    else:
        ensemble_scores = (iso_scores_norm + lof_scores_norm) / 2
    
    # Determine anomalies based on threshold
    anomaly_threshold = np.percentile(ensemble_scores, threshold * 100)
    is_anomaly = ensemble_scores > anomaly_threshold
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Transaction ID': df[id_col],
        'Anomaly Score': ensemble_scores,
        'Is Anomaly': is_anomaly
    })
    
    # Store models in session state
    st.session_state.best_model = {
        'preprocessor': preprocessor,
        'isolation_forest': iso_forest,
        'lof': lof,
        'ocsvm': ocsvm if svm_used else None,
        'threshold': anomaly_threshold
    }
    st.session_state.best_model_name = "Ensemble Anomaly Detection"
    
    # Display results
    st.subheader("Anomaly Detection Results")
    st.write(f"Found {is_anomaly.sum()} potential anomalies ({is_anomaly.mean()*100:.2f}% of transactions)")
    
    # Display top anomalies
    st.subheader("Top Anomalous Transactions")
    top_anomalies = results.sort_values('Anomaly Score', ascending=False).head(10)
    st.dataframe(top_anomalies)
    
    # Plot anomaly scores distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(ensemble_scores, bins=50, ax=ax)
    ax.axvline(x=anomaly_threshold, color='red', linestyle='--', 
               label=f'Threshold ({threshold*100}%)')
    ax.set_title('Distribution of Anomaly Scores')
    ax.set_xlabel('Anomaly Score')
    ax.legend()
    st.pyplot(fig)
    
    # Create feature contribution visualization for top anomaly
    if len(top_anomalies) > 0:
        st.subheader("Feature Contribution to Anomaly")
        
        # Get top anomaly
        top_anomaly_id = top_anomalies.iloc[0]['Transaction ID']
        top_anomaly_idx = df[df[id_col] == top_anomaly_id].index[0]
        
        # Calculate feature contributions
        # For this, we'll use the difference from mean, normalized by std
        feature_contribs = {}
        
        for feature in numerical_features:
            value = df.iloc[top_anomaly_idx][feature]
            mean = df[feature].mean()
            std = df[feature].std()
            if std > 0:  # Avoid division by zero
                z_score = abs(value - mean) / std
                feature_contribs[feature] = z_score
        
        # For categorical features, we'll use frequency-based anomaly score
        for feature in categorical_features:
            value = df.iloc[top_anomaly_idx][feature]
            # Calculate frequency of this value
            frequency = df[df[feature] == value].shape[0] / df.shape[0]
            # Rarer values are more anomalous
            feature_contribs[feature] = 1 - frequency
        
        # Plot feature contributions
        contrib_df = pd.DataFrame({
            'Feature': list(feature_contribs.keys()),
            'Contribution': list(feature_contribs.values())
        }).sort_values('Contribution', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Contribution', y='Feature', data=contrib_df.head(15), ax=ax)
        ax.set_title(f'Feature Contributions to Anomaly (Transaction {top_anomaly_id})')
        st.pyplot(fig)
    
    # Provide download link for full results
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Anomaly Detection Results (CSV)",
        csv,
        "anomaly_detection_results.csv",
        "text/csv",
        key='download-anomalies'
    )

###############################################################################
# SECTION B: REMITTANCE INFLOW FORECASTING MODULE
###############################################################################

def remittance_forecast_app():
    """
    Provides a module for forecasting remittance inflows using time series models.
    This helps in predicting future remittance trends to inform policy decisions.
    """
    st.header("Remittance Inflow Forecasting")
    st.info("Upload time series data to forecast future remittance inflows. "
           "The application will train time series models to predict future trends.")
    
    # Special options for time series forecasting
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_horizon = st.slider(
            "Forecast Horizon (periods ahead)",
            min_value=1,
            max_value=12,
            value=3,
            help="Number of future periods to forecast"
        )
    
    with col2:
        seasonality = st.checkbox(
            "Account for Seasonality",
            value=True,
            help="Enable seasonal components in forecasting models"
        )
    
    # File uploader for time series data
    uploaded_file = st.file_uploader("Upload remittance data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the remittance data
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Display data overview
            st.subheader("Remittance Data Overview")
            st.dataframe(df.head())
            
            # Date column selection
            date_col = st.selectbox(
                "Select date/time column:",
                df.columns
            )
            
            # Check if the column can be converted to datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                st.success(f"Successfully converted {date_col} to datetime")
            except:
                st.error(f"Could not convert {date_col} to datetime. Please select a valid date column.")
                return
            
            # Target column selection (remittance amount)
            target_col = st.selectbox(
                "Select remittance amount column:",
                [col for col in df.columns if col != date_col]
            )
            st.session_state.target = target_col
            
            # Additional features for forecasting
            external_features = st.multiselect(
                "Select additional features for forecasting (optional):",
                [col for col in df.columns if col not in [date_col, target_col]]
            )
            
            # Set model type to time_series
            st.session_state.model_type = "time_series"
            
            # Prepare time series data
            df = df.sort_values(date_col)
            
            # Display time series plot
            st.subheader("Remittance Time Series")
            fig = px.line(df, x=date_col, y=target_col, title=f"{target_col} over Time")
            st.plotly_chart(fig)
            
            # Check for stationarity
            if len(df) >= 10:  # Need sufficient data
                from statsmodels.tsa.stattools import adfuller
                
                # Run Augmented Dickey-Fuller test
                result = adfuller(df[target_col].dropna())
                
                st.subheader("Stationarity Check (Augmented Dickey-Fuller Test)")
                st.write(f"ADF Statistic: {result[0]:.4f}")
                st.write(f"p-value: {result[1]:.4f}")
                
                # Interpret the result
                if result[1] <= 0.05:
                    st.success("Time series is stationary (p-value <= 0.05)")
                    is_stationary = True
                else:
                    st.warning("Time series is not stationary (p-value > 0.05)")
                    st.info("Non-stationary data may need differencing for better forecasts.")
                    is_stationary = False
            
            # Train forecasting models button
            if st.button("Train Forecasting Models"):
                with st.spinner("Training time series models..."):
                    # Train time series forecasting models
                    train_time_series_models(
                        df, 
                        date_col, 
                        target_col, 
                        external_features, 
                        forecast_horizon, 
                        seasonality
                    )
        
        except Exception as e:
            st.error(f"Error processing remittance data: {str(e)}")
            st.info("Please check your data format and try again.")

def train_time_series_models(df, date_col, target_col, external_features, forecast_horizon, use_seasonality):
    """
    Trains time series models for forecasting remittance inflows.
    
    Args:
        df: DataFrame containing time series data
        date_col: Column containing date information
        target_col: Column containing remittance amounts
        external_features: Additional features to include in models
        forecast_horizon: Number of periods to forecast
        use_seasonality: Whether to include seasonal components
    """
    # Prepare the time series data
    df = df.sort_values(date_col)
    
    # Set the date as index for time series analysis
    df_ts = df.set_index(date_col)
    
    # Extract the target series
    y = df_ts[target_col]
    
    # Determine the frequency of the data
    freq = pd.infer.freq(df_ts.index)
    if freq is None:
        # Try to infer frequency if not automatically detected
        min_interval = min(df_ts.index[1:] - df_ts.index[:-1])
        
        if min_interval.days == 1:
            freq = 'D'  # Daily
        elif 28 <= min_interval.days <= 31:
            freq = 'M'  # Monthly
        elif 89 <= min_interval.days <= 92:
            freq = 'Q'  # Quarterly
        elif 350 <= min_interval.days <= 370:
            freq = 'A'  # Annual
        else:
            freq = 'D'  # Default to daily if can't determine
    
    st.write(f"Time series frequency: {freq}")
    
    # Determine seasonal periods based on frequency
    if freq == 'D':
        seasonal_periods = 7  # Weekly seasonality for daily data
    elif freq == 'M':
        seasonal_periods = 12  # Monthly seasonality
    elif freq == 'Q':
        seasonal_periods = 4  # Quarterly seasonality
    else:
        seasonal_periods = 1  # No seasonality for other frequencies
    
    # Determine train/test split
    train_size = int(len(y) * 0.8)
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Calculate the number of forecasts in the test set
    n_forecasts = min(len(y_test), forecast_horizon)
    
    # Initialize results storage
    results = []
    forecasts = {}
    
    # 1. Simple Moving Average
    st.subheader("Simple Moving Average")
    
    try:
        # Calculate moving average with window size = seasonal_periods
        window_size = seasonal_periods if seasonal_periods > 1 else 3
        ma = y_train.rolling(window=window_size).mean()
        
        # Forecast for test period is the last available MA value
        ma_forecast = [ma.iloc[-1]] * n_forecasts
        
        # Calculate MAE and RMSE
        ma_mae = mean_absolute_error(y_test[:n_forecasts], ma_forecast)
        ma_rmse = np.sqrt(mean_squared_error(y_test[:n_forecasts], ma_forecast))
        
        # Store results
        results.append({
            'Model': 'Moving Average',
            'MAE': ma_mae,
            'RMSE': ma_rmse
        })
        
        forecasts['Moving Average'] = ma_forecast
        
        st.write(f"Moving Average (window={window_size})")
        st.write(f"MAE: {ma_mae:.4f}, RMSE: {ma_rmse:.4f}")
    except Exception as e:
        st.error(f"Error training Moving Average model: {str(e)}")
    
    # 2. Exponential Smoothing
    st.subheader("Exponential Smoothing")
    
    try:
        # Define and fit exponential smoothing model
        if use_seasonality and seasonal_periods > 1:
            # Triple Exponential Smoothing (Holt-Winters)
            exp_model = ExponentialSmoothing(
                y_train, 
                seasonal_periods=seasonal_periods,
                seasonal='add',
                trend='add'
            )
        else:
            # Double Exponential Smoothing (Holt's method)
            exp_model = ExponentialSmoothing(
                y_train,
                trend='add',
                seasonal=None
            )
        
        exp_fit = exp_model.fit()
        
        # Forecast
        exp_forecast = exp_fit.forecast(n_forecasts)
        
        # Calculate MAE and RMSE
        exp_mae = mean_absolute_error(y_test[:n_forecasts], exp_forecast)
        exp_rmse = np.sqrt(mean_squared_error(y_test[:n_forecasts], exp_forecast))
        
        # Store results
        results.append({
            'Model': 'Exponential Smoothing',
            'MAE': exp_mae,
            'RMSE': exp_rmse
        })
        
        forecasts['Exponential Smoothing'] = exp_forecast
        
        st.write("Exponential Smoothing")
        st.write(f"MAE: {exp_mae:.4f}, RMSE: {exp_rmse:.4f}")
    except Exception as e:
        st.error(f"Error training Exponential Smoothing model: {str(e)}")
    
    # 3. ARIMA/SARIMA
    st.subheader("ARIMA/SARIMA")
    
    try:
        # Define and fit ARIMA model
        if use_seasonality and seasonal_periods > 1:
            # SARIMA with automatically selected parameters
            from pmdarima import auto_arima
            
            arima_model = auto_arima(
                y_train,
                seasonal=True,
                m=seasonal_periods,
                suppress_warnings=True,
                stepwise=True,
                max_order=None
            )
            
            model_name = "SARIMA"
        else:
            # ARIMA with automatically selected parameters
            from pmdarima import auto_arima
            
            arima_model = auto_arima(
                y_train,
                seasonal=False,
                suppress_warnings=True,
                stepwise=True
            )
            
            model_name = "ARIMA"
        
        # Get model order
        order = arima_model.order
        seasonal_order = arima_model.seasonal_order if hasattr(arima_model, 'seasonal_order') else None
        
        st.write(f"{model_name} Order: {order}")
        if seasonal_order:
            st.write(f"Seasonal Order: {seasonal_order}")
        
        # Forecast
        arima_forecast = arima_model.predict(n_periods=n_forecasts)
        
        # Calculate MAE and RMSE
        arima_mae = mean_absolute_error(y_test[:n_forecasts], arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(y_test[:n_forecasts], arima_forecast))
        
        # Store results
        results.append({
            'Model': model_name,
            'MAE': arima_mae,
            'RMSE': arima_rmse
        })
        
        forecasts[model_name] = arima_forecast
        
        st.write(f"{model_name}")
        st.write(f"MAE: {arima_mae:.4f}, RMSE: {arima_rmse:.4f}")
    except Exception as e:
        st.error(f"Error training ARIMA/SARIMA model: {str(e)}")
    
    # 4. Prophet (if available)
    st.subheader("Prophet (Facebook)")
    
    try:
        # Check if Prophet is installed
        import importlib.util
        prophet_spec = importlib.util.find_spec("prophet")
        
        if prophet_spec is not None:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df_ts.index,
                'y': df_ts[target_col]
            })
            
            # Add external regressors if available
            if external_features:
                for feature in external_features:
                    prophet_df[feature] = df_ts[feature]
            
            # Split into train/test
            prophet_train = prophet_df.iloc[:train_size]
            prophet_test = prophet_df.iloc[train_size:train_size+n_forecasts]
            
            # Create and fit Prophet model
            prophet_model = Prophet(
                yearly_seasonality=use_seasonality,
                weekly_seasonality=use_seasonality,
                daily_seasonality=False
            )
            
            # Add regressors if available
            if external_features:
                for feature in external_features:
                    prophet_model.add_regressor(feature)
            
            # Fit the model
            prophet_model.fit(prophet_train)
            
            # Create future dataframe for prediction
            future = prophet_model.make_future_dataframe(
                periods=n_forecasts,
                freq=freq
            )
            
            # Add regressors for future periods
            if external_features:
                for feature in external_features:
                    future[feature] = pd.concat([prophet_df[feature], 
                                              pd.Series([prophet_df[feature].mean()] * (n_forecasts - len(prophet_test)))])
            
            # Make prediction
            prophet_forecast = prophet_model.predict(future)
            
            # Extract forecasted values for test period
            prophet_pred = prophet_forecast.iloc[train_size:train_size+n_forecasts]['yhat'].values
            
            # Calculate MAE and RMSE
            prophet_mae = mean_absolute_error(y_test[:n_forecasts], prophet_pred)
            prophet_rmse = np.sqrt(mean_squared_error(y_test[:n_forecasts], prophet_pred))
            
            # Store results
            results.append({
                'Model': 'Prophet',
                'MAE': prophet_mae,
                'RMSE': prophet_rmse
            })
            
            forecasts['Prophet'] = prophet_pred
            
            st.write("Prophet")
            st.write(f"MAE: {prophet_mae:.4f}, RMSE: {prophet_rmse:.4f}")
            
            # Generate component plots
            fig1 = prophet_model.plot_components(prophet_forecast)
            st.write("Prophet Components")
            st.pyplot(fig1)
        else:
            st.warning("Prophet is not installed. Skipping Prophet model.")
    except Exception as e:
        st.error(f"Error training Prophet model: {str(e)}")
    
    # Compare all models
    results_df = pd.DataFrame(results)
    
    st.subheader("Model Comparison")
    st.dataframe(results_df)
    
    # Find the best model based on MAE
    best_model_idx = results_df['MAE'].idxmin()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    
    st.success(f"Best model: {best_model_name} (MAE: {results_df.loc[best_model_idx, 'MAE']:.4f})")
    
    # Plot actual vs predicted
    st.subheader("Forecast Comparison")
    
    # Create forecast visualization
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=df_ts.index,
        y=df_ts[target_col],
        mode='lines',
        name='Actual',
        line=dict(color='black')
    ))
    
    # Add predictions for each model
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    color_idx = 0
    
    for model_name, forecast in forecasts.items():
        # Create forecast dates (continuing from training data)
        forecast_dates = pd.date_range(
            start=df_ts.index[train_size],
            periods=len(forecast),
            freq=freq
        )
        
        # Add to plot
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines',
            name=f'{model_name} Forecast',
            line=dict(color=colors[color_idx % len(colors)])
        ))
        
        color_idx += 1
    
    # Update layout
    fig.update_layout(
        title='Remittance Forecasts by Model',
        xaxis_title='Date',
        yaxis_title=target_col,
        legend=dict(x=0, y=1.1, orientation='h'),
        template='plotly_white'
    )
    
    st.plotly_chart(fig)
    
    # Generate future forecasts with best model
    st.subheader("Future Forecast")
    st.write(f"Generating {forecast_horizon}-period forecast using {best_model_name}")
    
    # Get the full dataset for training final model
    y_full = df_ts[target_col]
    
    # Train final model on full dataset and forecast future
    if best_model_name == 'Moving Average':
        # For moving average, use the last window average
        window_size = seasonal_periods if seasonal_periods > 1 else 3
        ma_full = y_full.rolling(window=window_size).mean()
        future_forecast = [ma_full.iloc[-1]] * forecast_horizon
        
    elif best_model_name == 'Exponential Smoothing':
        # Exponential smoothing on full dataset
        if use_seasonality and seasonal_periods > 1:
            exp_model_full = ExponentialSmoothing(
                y_full, 
                seasonal_periods=seasonal_periods,
                seasonal='add',
                trend='add'
            )
        else:
            exp_model_full = ExponentialSmoothing(
                y_full,
                trend='add',
                seasonal=None
            )
        
        exp_fit_full = exp_model_full.fit()
        future_forecast = exp_fit_full.forecast(forecast_horizon)
        
    elif best_model_name in ['ARIMA', 'SARIMA']:
        # Refit ARIMA/SARIMA on full dataset
        from pmdarima import auto_arima
        
        if best_model_name == 'SARIMA':
            arima_model_full = auto_arima(
                y_full,
                seasonal=True,
                m=seasonal_periods,
                suppress_warnings=True,
                stepwise=True,
                max_order=None
            )
        else:
            arima_model_full = auto_arima(
                y_full,
                seasonal=False,
                suppress_warnings=True,
                stepwise=True
            )
        
        future_forecast = arima_model_full.predict(n_periods=forecast_horizon)
        
    elif best_model_name == 'Prophet':
        # Prophet on full dataset
        from prophet import Prophet
        
        # Prepare data
        prophet_df_full = pd.DataFrame({
            'ds': df_ts.index,
            'y': df_ts[target_col]
        })
        
        # Add external regressors if available
        if external_features:
            for feature in external_features:
                prophet_df_full[feature] = df_ts[feature]
        
        # Create and fit Prophet model
        prophet_model_full = Prophet(
            yearly_seasonality=use_seasonality,
            weekly_seasonality=use_seasonality,
            daily_seasonality=False
        )
        
        # Add regressors if available
        if external_features:
            for feature in external_features:
                prophet_model_full.add_regressor(feature)
        
        # Fit model
        prophet_model_full.fit(prophet_df_full)
        
        # Create future dataframe
        future_full = prophet_model_full.make_future_dataframe(
            periods=forecast_horizon,
            freq=freq
        )
        
        # Add regressor values for future (using mean of historical values)
        if external_features:
            for feature in external_features:
                future_full[feature] = list(prophet_df_full[feature]) + \
                                     [prophet_df_full[feature].mean()] * forecast_horizon
        
        # Forecast
        future_forecast_df = prophet_model_full.predict(future_full)
        future_forecast = future_forecast_df.iloc[-forecast_horizon:]['yhat'].values
    
    else:
        # Default to simple average forecast
        future_forecast = [y_full.mean()] * forecast_horizon
    
    # Create future dates
    last_date = df_ts.index[-1]
    future_dates = pd.date_range(
        start=pd.Timestamp(last_date) + pd.DateOffset(**{f"{freq.lower()}s": 1}),
        periods=forecast_horizon,
        freq=freq
    )
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_forecast
    })
    
    # Display forecast table
    st.write("Future Remittance Forecast:")
    st.dataframe(forecast_df)
    
    # Plot the forecast
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df_ts.index,
        y=df_ts[target_col],
        mode='lines',
        name='Historical Data',
        line=dict(color='black')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_forecast,
        mode='lines',
        name='Forecast',
        line=dict(color='blue', dash='dash')
    ))
    
    # Add prediction intervals (±10% and ±20% as a simple approximation)
    # In a real implementation, use proper prediction intervals from the models
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(np.array(future_forecast) * 1.1) + list(np.array(future_forecast) * 0.9)[::-1],
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)',
        line=dict(color='rgba(0, 0, 255, 0)'),
        name='90% Confidence'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(np.array(future_forecast) * 1.2) + list(np.array(future_forecast) * 0.8)[::-1],
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)',
        line=dict(color='rgba(0, 0, 255, 0)'),
        name='80% Confidence'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{forecast_horizon}-Period Remittance Forecast',
        xaxis_title='Date',
        yaxis_title=target_col,
        legend=dict(x=0, y=1.1, orientation='h'),
        template='plotly_white'
    )
    
    st.plotly_chart(fig)
    
    # Provide download link for forecast
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Forecast (CSV)",
        csv,
        "remittance_forecast.csv",
        "text/csv",
        key='download-forecast'
    )
    
    # Save the best model to session state
    if best_model_name == 'Moving Average':
        best_model = {
            'type': 'Moving Average',
            'window_size': window_size,
            'last_value': ma_full.iloc[-1]
        }
    elif best_model_name == 'Exponential Smoothing':
        best_model = {
            'type': 'Exponential Smoothing',
            'model': exp_fit_full
        }
    elif best_model_name in ['ARIMA', 'SARIMA']:
        best_model = {
            'type': best_model_name,
            'model': arima_model_full
        }
    elif best_model_name == 'Prophet':
        best_model = {
            'type': 'Prophet',
            'model': prophet_model_full
        }
    
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_model_name

###############################################################################
# SECTION C: CREDIT RISK MODELING AND NPL MANAGEMENT MODULE
###############################################################################

def credit_risk_app():
    """
    Provides credit risk modeling and NPL management functionality.
    This module helps identify loans likely to default and enables interventions.
    """
    st.header("Credit Risk Modeling and NPL Management")
    st.info("Upload loan portfolio data to train a credit risk model. "
           "This will help identify loans likely to default and prioritize interventions.")
    
    # File uploader for loan data
    uploaded_file = st.file_uploader("Upload loan portfolio data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the loan data
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Display data overview
            st.subheader("Loan Portfolio Overview")
            st.dataframe(df.head())
            
            # Loan ID column selection
            id_column = st.selectbox(
                "Select loan ID column:",
                df.columns
            )
            st.session_state.id_column = id_column
            
            # Target column selection (default status)
            target_col = st.selectbox(
                "Select default status column (1=default, 0=performing):",
                [col for col in df.columns if col != id_column]
            )
            st.session_state.target = target_col
            st.session_state.model_type = "classification"
            
            # Check class balance
            if target_col in df.columns:
                default_count = df[target_col].sum()
                total_count = len(df)
                default_percentage = (default_count / total_count) * 100
                
                st.write(f"Defaulted loans: {default_count} ({default_percentage:.2f}%)")
                st.write(f"Performing loans: {total_count - default_count} ({100-default_percentage:.2f}%)")
                
                # Warning for imbalanced dataset
                if default_percentage < 10:
                    st.warning("Imbalanced dataset detected. Consider using specialized techniques.")
            
            # Feature selection
            exclude_cols = [id_column, target_col]
            available_features = [col for col in df.columns if col not in exclude_cols]
            
            # Group features for easier selection
            numeric_features = [col for col in available_features if pd.api.types.is_numeric_dtype(df[col])]
            categorical_features = [col for col in available_features if not pd.api.types.is_numeric_dtype(df[col])]
            
            # Create expandable sections for feature selection
            with st.expander("Select Features for Credit Risk Model", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Numerical Features")
                    selected_numeric = []
                    for feature in numeric_features:
                        if st.checkbox(feature, value=True, key=f"num_{feature}"):
                            selected_numeric.append(feature)
                
                with col2:
                    st.write("Categorical Features")
                    selected_categorical = []
                    for feature in categorical_features:
                        if st.checkbox(feature, value=True, key=f"cat_{feature}"):
                            selected_categorical.append(feature)
            
            # Combine selected features
            selected_features = selected_numeric + selected_categorical
            
            if not selected_features:
                st.warning("Please select at least one feature to continue.")
            else:
                st.session_state.features = selected_features
                st.session_state.categorical_features = selected_categorical
                
                # Model training section
                st.subheader("Credit Risk Model Training")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Model options
                    class_weight = st.checkbox(
                        "Apply class weights for imbalanced data", 
                        value=True,
                        help="Gives more weight to minority class (defaulted loans)"
                    )
                    
                    threshold = st.slider(
                        "Prediction Threshold", 
                        min_value=0.1, 
                        max_value=0.9, 
                        value=0.5,
                        help="Lower values increase sensitivity to defaults"
                    )
                
                with col2:
                    # Additional options
                    cv_folds = st.slider(
                        "Cross-validation folds", 
                        min_value=3, 
                        max_value=10, 
                        value=5,
                        help="Number of folds for cross-validation"
                    )
                    
                    use_smote = st.checkbox(
                        "Use SMOTE for Imbalanced Data", 
                        value=False,
                        help="Synthetic Minority Over-sampling Technique"
                    )
                
                # Display feature exploration
                st.subheader("Feature Exploration")
                
                # Select feature to explore
                explore_feature = st.selectbox(
                    "Select feature to explore:",
                    selected_features
                )
                
                # Create visualization based on feature type
                if explore_feature in selected_numeric:
                    # For numeric features, show distribution by default status
                    fig = px.histogram(
                        df, 
                        x=explore_feature, 
                        color=target_col,
                        marginal="box",
                        labels={target_col: "Default Status"},
                        title=f"Distribution of {explore_feature} by Default Status"
                    )
                    st.plotly_chart(fig)
                    
                    # Calculate statistics
                    stats = df.groupby(target_col)[explore_feature].agg(['mean', 'median', 'std']).reset_index()
                    st.write(f"Statistics for {explore_feature}:")
                    st.dataframe(stats)
                else:
                    # For categorical features, show count by default status
                    fig = px.histogram(
                        df, 
                        x=explore_feature, 
                        color=target_col,
                        barmode="group",
                        labels={target_col: "Default Status"},
                        title=f"Distribution of {explore_feature} by Default Status"
                    )
                    st.plotly_chart(fig)
                    
                    # Calculate default rates by category
                    default_rates = df.groupby(explore_feature)[target_col].mean().reset_index()
                    default_rates.columns = [explore_feature, 'Default Rate']
                    default_rates['Default Rate'] = default_rates['Default Rate'] * 100
                    
                    st.write(f"Default Rates by {explore_feature}:")
                    st.dataframe(default_rates)
                
                # Train model button
                if st.button("Train Credit Risk Model"):
                    # Train the credit risk model
                    with st.spinner("Training credit risk model..."):
                        train_credit_risk_model(
                            df,
                            selected_features,
                            target_col,
                            id_column,
                            class_weight,
                            threshold,
                            cv_folds,
                            use_smote
                        )
        
        except Exception as e:
            st.error(f"Error processing loan data: {str(e)}")
            st.info("Please check your data format and try again.")

def train_credit_risk_model(df, features, target, id_col, use_class_weights, threshold, cv_folds, use_smote):
    """
    Trains a credit risk model to predict loan defaults.
    
    Args:
        df: DataFrame containing loan data
        features: List of features to use for prediction
        target: Target column with default status
        id_col: Column containing loan IDs
        use_class_weights: Whether to apply class weights for imbalanced data
        threshold: Prediction threshold for classification
        cv_folds: Number of cross-validation folds
        use_smote: Whether to use SMOTE for imbalanced data
    """
    # Prepare the dataset
    X = df[features].copy()
    y = df[target].copy()
    
    # Identify categorical features
    categorical_features = [f for f in features if df[f].dtype == 'object' or df[f].dtype.name == 'category']
    numerical_features = [f for f in features if f not in categorical_features]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Apply SMOTE if requested
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            
            # First preprocess the data
            X_train_processed = preprocessor.fit_transform(X_train)
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            
            # Check the new class distribution
            class_counts = np.bincount(y_train_resampled)
            st.write(f"Class distribution after SMOTE: 0: {class_counts[0]}, 1: {class_counts[1]}")
            
            # Create a pipeline that includes preprocessing and SMOTE
            pipeline_with_smote = True
        except ImportError:
            st.warning("SMOTE is not installed. Skipping SMOTE.")
            use_smote = False
            pipeline_with_smote = False
    else:
        pipeline_with_smote = False
    
    # Define models to evaluate
    models = {
        "Random Forest": RandomForestClassifier(
            random_state=42, 
            class_weight='balanced' if use_class_weights else None
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                random_state=42,
                class_weight='balanced' if use_class_weights else None,
                

max_iter=1000
            ))
        ])
    }
    
    # Evaluate models using cross-validation
    results = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        try:
            # Update progress
            progress_bar.progress((i) / len(models))
            
            # Create full pipeline with preprocessing
            if pipeline_with_smote:
                # If using SMOTE, train directly on resampled data
                pipeline = model
                pipeline.fit(X_train_resampled, y_train_resampled)
                
                # Get predictions using a custom threshold
                y_proba = pipeline.predict_proba(preprocessor.transform(X_test))[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
            else:
                # Create a pipeline with preprocessing
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                # Perform cross-validation
                cv_results = cross_val_predict(
                    pipeline, 
                    X_train, 
                    y_train, 
                    cv=cv_folds, 
                    method='predict_proba'
                )
                
                # Train the full model
                pipeline.fit(X_train, y_train)
                
                # Get predictions using a custom threshold
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            # Store results
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'AUC': auc,
                'Pipeline': pipeline
            })
            
            st.write(f"✅ Evaluated {name}")
        except Exception as e:
            st.error(f"❌ Error evaluating {name}: {str(e)}")
    
    # Clear progress bar
    progress_bar.empty()
    
    # Create results DataFrame (exclude pipeline)
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Pipeline'} for r in results])
    
    # Display results
    st.subheader("Credit Risk Model Evaluation")
    st.dataframe(results_df)
    
    # Find best model (based on AUC)
    if not results_df.empty:
        best_model_idx = results_df['AUC'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'Model']
        best_pipeline = [r['Pipeline'] for r in results if r['Model'] == best_model_name][0]
        
        # Store best model in session state
        st.session_state.best_model = {
            'pipeline': best_pipeline,
            'threshold': threshold
        }
        st.session_state.best_model_name = best_model_name
        
        st.success(f"Best model: {best_model_name} (AUC: {results_df.loc[best_model_idx, 'AUC']:.4f})")
        
        # Display ROC curve
        st.subheader("ROC Curve")
        
        # Get y_proba
        if pipeline_with_smote:
            final_proba = best_pipeline.predict_proba(preprocessor.transform(X_test))[:, 1]
        else:
            final_proba = best_pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, final_proba)
        
        # Plot ROC curve
        fig = px.line(
            x=fpr, y=tpr,
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            title=f'ROC Curve for {best_model_name} (AUC = {results_df.loc[best_model_idx, "AUC"]:.4f})'
        )
        
        # Add diagonal line (random classifier)
        fig.add_shape(
            type='line',
            line=dict(dash='dash', color='gray'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        # Add threshold point
        threshold_point = np.argmin(np.abs(np.array(_) - threshold))
        fig.add_trace(
            go.Scatter(
                x=[fpr[threshold_point]], 
                y=[tpr[threshold_point]],
                mode='markers',
                marker=dict(size=10, color='red'),
                name=f'Threshold = {threshold}'
            )
        )
        
        st.plotly_chart(fig)
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        
        # Get predictions
        y_pred = (final_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create heatmap
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual"),
            x=['Non-Default', 'Default'],
            y=['Non-Default', 'Default'],
            text_auto=True,
            color_continuous_scale='Blues',
            title=f"Confusion Matrix - {best_model_name}"
        )
        
        st.plotly_chart(fig)
        
        # Display classification report
        st.subheader("Classification Report")
        
        # Generate report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        # Display
        st.dataframe(report_df)
        
        # Create loan risk scoring app
        st.header("Loan Risk Scoring Tool")
        st.info("Use the trained model to score loans for default risk")
        
        # Sample loans for demonstration
        sample_loans = X_test.head(5).copy()
        sample_loans[id_col] = df[id_col].iloc[:5].values
        
        # Display sample loans
        st.write("Sample loans from test data:")
        st.dataframe(sample_loans)
        
        # Score sample loans button
        if st.button("Score Sample Loans"):
            with st.spinner("Scoring loans..."):
                # Get predictions
                if pipeline_with_smote:
                    sample_proba = best_pipeline.predict_proba(
                        preprocessor.transform(sample_loans[features])
                    )[:, 1]
                else:
                    sample_proba = best_pipeline.predict_proba(
                        sample_loans[features]
                    )[:, 1]
                
                # Create results
                risk_results = pd.DataFrame({
                    'Loan ID': sample_loans[id_col],
                    'Default Risk': sample_proba,
                    'Risk Category': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in sample_proba]
                })
                
                # Display results
                st.write("Loan Default Risk Assessment:")
                st.dataframe(risk_results.style.background_gradient(
                    subset=['Default Risk'],
                    cmap='YlOrRd'
                ))
                
                # Create risk visualization
                fig = px.bar(
                    risk_results,
                    x='Loan ID',
                    y='Default Risk',
                    color='Risk Category',
                    color_discrete_map={
                        'Low': 'green',
                        'Medium': 'orange',
                        'High': 'red'
                    },
                    title='Default Risk by Loan'
                )
                
                # Add threshold line
                fig.add_shape(
                    type='line',
                    x0=-0.5,
                    x1=len(risk_results)-0.5,
                    y0=threshold,
                    y1=threshold,
                    line=dict(
                        color='black',
                        width=2,
                        dash='dash'
                    ),
                    name='Threshold'
                )
                
                # Add annotation for threshold
                fig.add_annotation(
                    x=len(risk_results)-1,
                    y=threshold,
                    text=f'Threshold: {threshold}',
                    showarrow=False,
                    yshift=10
                )
                
                st.plotly_chart(fig)
                
                # Provide download option
                csv = risk_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Risk Assessment (CSV)",
                    csv,
                    "loan_risk_assessment.csv",
                    "text/csv",
                    key='download-risk'
                )
        
        # Add a loan scoring form
        st.subheader("Score a New Loan")
        
        # Create a form for inputting loan features
        with st.form("loan_scoring_form"):
            # Create input fields for each feature
            loan_data = {}
            
            # Group features into columns
            cols = st.columns(3)
            col_idx = 0
            
            for feature in features:
                with cols[col_idx % 3]:
                    if feature in categorical_features:
                        # For categorical features, create a dropdown
                        unique_values = df[feature].dropna().unique()
                        loan_data[feature] = st.selectbox(
                            f"{feature}:",
                            unique_values,
                            key=f"new_{feature}"
                        )
                    else:
                        # For numerical features, create a number input
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        default_val = float(df[feature].median())
                        
                        loan_data[feature] = st.number_input(
                            f"{feature}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            key=f"new_{feature}"
                        )
                
                col_idx += 1
            
            # Add submit button
            submit_button = st.form_submit_button("Calculate Default Risk")
        
        # Process form submission
        if submit_button:
            with st.spinner("Calculating risk..."):
                # Create DataFrame from input
                new_loan = pd.DataFrame([loan_data])
                
                # Make prediction
                if pipeline_with_smote:
                    new_proba = best_pipeline.predict_proba(
                        preprocessor.transform(new_loan)
                    )[0, 1]
                else:
                    new_proba = best_pipeline.predict_proba(new_loan)[0, 1]
                
                # Display result
                st.subheader("Risk Assessment Result")
                
                # Create risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=new_proba,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Default Risk"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "green"},
                            {'range': [0.3, 0.7], 'color': "orange"},
                            {'range': [0.7, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold
                        }
                    }
                ))
                
                st.plotly_chart(fig)
                
                # Add text interpretation
                if new_proba >= 0.7:
                    st.error(f"High Risk Loan (Risk Score: {new_proba:.2%})")
                    st.write("This loan has a high risk of default. Consider additional collateral or guarantees.")
                elif new_proba >= 0.3:
                    st.warning(f"Medium Risk Loan (Risk Score: {new_proba:.2%})")
                    st.write("This loan has a moderate risk of default. Monitor this loan more closely.")
                else:
                    st.success(f"Low Risk Loan (Risk Score: {new_proba:.2%})")
                    st.write("This loan has a low risk of default. Standard monitoring should be sufficient.")
    else:
        st.error("No models were successfully evaluated. Please check your data and try again.")

###############################################################################
# SECTION D: ECONOMIC FORECASTING AND STRESS TESTING MODULE
###############################################################################

def economic_forecast_app():
    """
    Provides a module for economic forecasting and bank stress testing.
    This helps forecast key economic indicators and assess bank resilience.
    """
    st.header("Economic Forecasting and Stress Testing")
    st.info("Upload economic data to build forecasting models and conduct stress tests "
            "on the banking sector.")
    
    # Create tabs for forecasting and stress testing
    tabs = st.tabs(["Economic Forecasting", "Bank Stress Testing"])
    
    # Economic Forecasting Tab
    with tabs[0]:
        st.subheader("Economic Indicator Forecasting")
        st.write("Upload historical economic data to forecast key indicators like "
                "inflation, GDP, exchange rates, etc.")
        
        # Indicator selection
        indicator = st.selectbox(
            "Select economic indicator to forecast:",
            ["Inflation Rate", "GDP Growth", "Exchange Rate", "Interest Rate", "Other"]
        )
        
        if indicator == "Other":
            custom_indicator = st.text_input("Enter custom indicator name:")
            if custom_indicator:
                indicator = custom_indicator
        
        # File uploader for economic data
        uploaded_file = st.file_uploader(
            f"Upload {indicator} historical data (CSV)",
            type="csv",
            key="economic_data"
        )
        
        if uploaded_file is not None:
            try:
                # Read the data
                df = pd.read_csv(uploaded_file)
                
                # Display data overview
                st.write("Data Overview:")
                st.dataframe(df.head())
                
                # Let user select date column
                date_col = st.selectbox(
                    "Select date column:",
                    df.columns
                )
                
                # Convert to datetime
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    st.success(f"Successfully converted {date_col} to datetime")
                except:
                    st.error(f"Could not convert {date_col} to datetime. Please select a valid date column.")
                    st.stop()
                
                # Select target column (indicator value)
                target_col = st.selectbox(
                    f"Select column containing {indicator} values:",
                    [col for col in df.columns if col != date_col]
                )
                
                # Select external features (optional)
                external_features = st.multiselect(
                    "Select additional features for forecasting (optional):",
                    [col for col in df.columns if col not in [date_col, target_col]]
                )
                
                # Forecast parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    forecast_horizon = st.slider(
                        "Forecast Horizon (periods ahead)",
                        min_value=1,
                        max_value=12,
                        value=4
                    )
                
                with col2:
                    use_seasonality = st.checkbox(
                        "Account for Seasonality",
                        value=True
                    )
                
                # Time series visualization
                st.subheader(f"Historical {indicator} Trend")
                
                # Create time series plot
                fig = px.line(
                    df,
                    x=date_col,
                    y=target_col,
                    title=f"Historical {indicator} Values"
                )
                
                st.plotly_chart(fig)
                
                # Run forecast button
                if st.button("Generate Forecast"):
                    with st.spinner(f"Forecasting {indicator}..."):
                        # This would call the same function as remittance forecasting
                        # with appropriate modifications for the economic indicator
                        train_time_series_models(
                            df, 
                            date_col, 
                            target_col, 
                            external_features, 
                            forecast_horizon,
                            use_seasonality
                        )
            
            except Exception as e:
                st.error(f"Error processing economic data: {str(e)}")
                st.info("Please check your data format and try again.")
    
    # Bank Stress Testing Tab
    with tabs[1]:
        st.subheader("Banking Sector Stress Testing")
        st.write("Upload bank data to conduct stress tests under various economic scenarios.")
        
        # Stress test parameters
        col1, col2 = st.columns(2)
        
        with col1:
            stress_scenario = st.selectbox(
                "Select stress scenario:",
                ["Moderate Recession", "Severe Recession", "Currency Crisis", 
                 "Interest Rate Shock", "Custom Scenario"]
            )
        
        with col2:
            test_metric = st.selectbox(
                "Select metric to stress test:",
                ["Capital Adequacy Ratio", "NPL Ratio", "Liquidity Coverage Ratio"]
            )
        
        # File uploader for bank data
        uploaded_bank_file = st.file_uploader(
            "Upload bank financial data (CSV)",
            type="csv",
            key="bank_data"
        )
        
        if uploaded_bank_file is not None:
            try:
                # Read the bank data
                bank_df = pd.read_csv(uploaded_bank_file)
                
                # Display data
                st.write("Bank Data Overview:")
                st.dataframe(bank_df.head())
                
                # Bank identifier column
                bank_id_col = st.selectbox(
                    "Select bank identifier column:",
                    bank_df.columns
                )
                
                # Select relevant financial metrics
                financial_metrics = st.multiselect(
                    "Select financial metrics for stress testing:",
                    [col for col in bank_df.columns if col != bank_id_col],
                    default=[col for col in bank_df.columns if col != bank_id_col][:5]  # Default to first 5 columns
                )
                
                # Custom scenario parameters
                if stress_scenario == "Custom Scenario":
                    st.subheader("Custom Stress Scenario Parameters")
                    
                    # GDP shock
                    gdp_shock = st.slider(
                        "GDP Growth Shock",
                        min_value=-15.0,
                        max_value=0.0,
                        value=-5.0,
                        format="%.1f%%"
                    )
                    
                    # Interest rate shock
                    interest_shock = st.slider(
                        "Interest Rate Shock",
                        min_value=0.0,
                        max_value=10.0,
                        value=2.0,
                        format="%.1f percentage points"
                    )
                    
                    # Exchange rate shock
                    fx_shock = st.slider(
                        "Exchange Rate Depreciation",
                        min_value=0.0,
                        max_value=50.0,
                        value=15.0,
                        format="%.1f%%"
                    )
                    
                    # Unemployment shock
                    unemployment_shock = st.slider(
                        "Unemployment Rate Increase",
                        min_value=0.0,
                        max_value=10.0,
                        value=3.0,
                        format="%.1f percentage points"
                    )
                
                # Run stress test button
                if st.button("Run Stress Test"):
                    with st.spinner("Running stress test simulations..."):
                        # Call the stress testing function
                        run_stress_test(
                            bank_df,
                            bank_id_col,
                            financial_metrics,
                            stress_scenario,
                            test_metric
                        )
            
            except Exception as e:
                st.error(f"Error processing bank data: {str(e)}")
                st.info("Please check your data format and try again.")

def run_stress_test(bank_df, bank_id_col, metrics, scenario, test_metric):
    """
    Runs a stress test simulation on bank data under various economic scenarios.
    
    Args:
        bank_df: DataFrame containing bank financial data
        bank_id_col: Column identifying banks
        metrics: Financial metrics to include in stress test
        scenario: Economic scenario to simulate
        test_metric: Primary metric to stress test
    """
    # Define scenario parameters
    scenario_params = {
        "Moderate Recession": {
            "gdp_shock": -3.0,  # GDP decline of 3%
            "interest_shock": 1.5,  # Interest rate increase of 1.5 percentage points
            "fx_shock": 10.0,  # Currency depreciation of 10%
            "unemployment_shock": 2.0,  # Unemployment increase of 2 percentage points
            "severity": "moderate"
        },
        "Severe Recession": {
            "gdp_shock": -8.0,
            "interest_shock": 3.0,
            "fx_shock": 25.0,
            "unemployment_shock": 5.0,
            "severity": "severe"
        },
        "Currency Crisis": {
            "gdp_shock": -2.0,
            "interest_shock": 5.0,
            "fx_shock": 40.0,
            "unemployment_shock": 1.5,
            "severity": "severe"
        },
        "Interest Rate Shock": {
            "gdp_shock": -1.0,
            "interest_shock": 4.0,
            "fx_shock": 5.0,
            "unemployment_shock": 1.0,
            "severity": "moderate"
        },
        "Custom Scenario": {
            # These are placeholders - actual values come from UI sliders
            "gdp_shock": -5.0,
            "interest_shock": 2.0,
            "fx_shock": 15.0,
            "unemployment_shock": 3.0,
            "severity": "custom"
        }
    }
    
    # Get parameters for the selected scenario
    params = scenario_params[scenario]
    
    # Display scenario details
    st.subheader(f"{scenario} - Stress Test Parameters")
    
    # Create parameter display
    param_df = pd.DataFrame({
        "Parameter": ["GDP Growth", "Interest Rate", "Exchange Rate", "Unemployment Rate"],
        "Shock": [f"{params['gdp_shock']}%", 
                 f"+{params['interest_shock']} pp", 
                 f"{params['fx_shock']}% depr.", 
                 f"+{params['unemployment_shock']} pp"]
    })
    
    st.table(param_df)
    
    # Create a simple stress testing model
    # In a real implementation, this would be a more sophisticated model
    # based on historical relationships between economic factors and bank metrics
    
    # For demonstration, we'll use a simplified approach
    
    # Define impact factors based on scenario severity
    if params["severity"] == "severe":
        impact_factors = {
            "Capital Adequacy Ratio": -0.25,  # Severe impact on capital
            "NPL Ratio": 1.5,  # Large increase in NPLs
            "Liquidity Coverage Ratio": -0.3  # Significant liquidity reduction
        }
    elif params["severity"] == "moderate":
        impact_factors = {
            "Capital Adequacy Ratio": -0.15,
            "NPL Ratio": 0.8,
            "Liquidity Coverage Ratio": -0.2
        }
    else:  # custom
        impact_factors = {
            "Capital Adequacy Ratio": -0.2,
            "NPL Ratio": 1.0,
            "Liquidity Coverage Ratio": -0.25
        }
    
    # Check if test metric exists in bank data
    if test_metric not in bank_df.columns:
        st.error(f"{test_metric} not found in the bank data. Please check your data.")
        return
    
    # Create stressed dataframe
    stressed_df = bank_df.copy()
    
    # Apply stress impacts
    # This is a simplified model where:
    # - GDP shock affects all metrics
    # - Interest rate shock mainly affects NPLs and liquidity
    # - FX shock mainly affects capital for banks with foreign exposure
    # - Unemployment mainly affects NPLs
    
    # Calculate combined impact
    gdp_impact = abs(params["gdp_shock"]) / 10.0  # Normalize impact
    interest_impact = params["interest_shock"] / 5.0
    fx_impact = params["fx_shock"] / 50.0
    unemployment_impact = params["unemployment_shock"] / 5.0
    
    # Apply stress to the test metric
    if test_metric == "Capital Adequacy Ratio":
        # CAR decreases under stress
        factor = impact_factors[test_metric]
        stressed_df[test_metric] = bank_df[test_metric] * (1 + factor * (
            0.5 * gdp_impact + 0.2 * interest_impact + 0.2 * fx_impact + 0.1 * unemployment_impact
        ))
    elif test_metric == "NPL Ratio":
        # NPL increases under stress
        factor = impact_factors[test_metric]
        stressed_df[test_metric] = bank_df[test_metric] * (1 + factor * (
            0.3 * gdp_impact + 0.3 * interest_impact + 0.1 * fx_impact + 0.3 * unemployment_impact
        ))
    elif test_metric == "Liquidity Coverage Ratio":
        # LCR decreases under stress
        factor = impact_factors[test_metric]
        stressed_df[test_metric] = bank_df[test_metric] * (1 + factor * (
            0.3 * gdp_impact + 0.4 * interest_impact + 0.2 * fx_impact + 0.1 * unemployment_impact
        ))
    
    # Enforce reasonable limits (prevent negative values)
    if test_metric in ["Capital Adequacy Ratio", "Liquidity Coverage Ratio"]:
        stressed_df[test_metric] = stressed_df[test_metric].clip(lower=0)
    
    # Display results
    st.subheader("Stress Test Results")
    
    # Create a comparison DataFrame
    results_df = pd.DataFrame({
        "Bank": bank_df[bank_id_col],
        f"Current {test_metric}": bank_df[test_metric],
        f"Stressed {test_metric}": stressed_df[test_metric],
        "Change": stressed_df[test_metric] - bank_df[test_metric],
        "Percent Change": ((stressed_df[test_metric] - bank_df[test_metric]) / bank_df[test_metric]) * 100
    })
    
    # Display table with formatting
    st.dataframe(results_df.style.format({
        f"Current {test_metric}": "{:.2f}",
        f"Stressed {test_metric}": "{:.2f}",
        "Change": "{:.2f}",
        "Percent Change": "{:.2f}%"
    }).background_gradient(
        cmap='RdYlGn',
        subset=["Percent Change"]
    ))
    
    # Create a bar chart comparing current vs stressed values
    fig = go.Figure()
    
    # Add current values
    fig.add_trace(go.Bar(
        x=results_df["Bank"],
        y=results_df[f"Current {test_metric}"],
        name="Current",
        marker_color='blue'
    ))
    
    # Add stressed values
    fig.add_trace(go.Bar(
        x=results_df["Bank"],
        y=results_df[f"Stressed {test_metric}"],
        name=f"Stressed ({scenario})",
        marker_color='red'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Impact of {scenario} on {test_metric}",
        xaxis_title="Bank",
        yaxis_title=test_metric,
        barmode='group'
    )
    
    st.plotly_chart(fig)
    
    # Add regulatory threshold line if applicable
    if test_metric == "Capital Adequacy Ratio":
        # Add regulatory minimum visualization
        min_car = 10.0  # Example: Bangladesh Bank's minimum CAR requirement
        
        # Create a new chart
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=results_df["Bank"],
            y=results_df[f"Stressed {test_metric}"],
            name=f"Stressed {test_metric}",
            marker_color=['red' if val < min_car else 'green' for val in results_df[f"Stressed {test_metric}"]]
        ))
        
        # Add threshold line
        fig2.add_shape(
            type='line',
            x0=-0.5,
            x1=len(results_df)-0.5,
            y0=min_car,
            y1=min_car,
            line=dict(
                color='red',
                width=2,
                dash='dash'
            )
        )
        
        # Add annotation
        fig2.add_annotation(
            x=len(results_df)-1,
            y=min_car,
            text=f'Regulatory Minimum: {min_car}%',
            showarrow=False,
            yshift=10
        )
        
        # Update layout
        fig2.update_layout(
            title=f"Banks Below Regulatory Minimum {test_metric} Under Stress",
            xaxis_title="Bank",
            yaxis_title=test_metric
        )
        
        st.plotly_chart(fig2)
        
        # Identify vulnerable banks
        vulnerable_banks = results_df[results_df[f"Stressed {test_metric}"] < min_car]
        
        if not vulnerable_banks.empty:
            st.warning(f"{len(vulnerable_banks)} banks would fall below regulatory minimum {test_metric} in this scenario.")
            st.dataframe(vulnerable_banks)
        else:
            st.success(f"All banks would maintain minimum {test_metric} requirements in this scenario.")
    
    # Summary recommendations
    st.subheader("Stress Test Recommendations")
    
    # Calculate overall impact
    avg_impact = results_df["Percent Change"].mean()
    
    # Generate recommendations based on scenario and impact
    if avg_impact < -20 or (test_metric == "NPL Ratio" and avg_impact > 20):
        st.error("Severe Impact Detected")
        recommendations = [
            "Consider increasing capital buffers for vulnerable banks",
            "Review and strengthen liquidity facilities",
            "Develop contingency plans for worst-case scenarios",
            "Enhance supervisory monitoring frequency"
        ]
    elif avg_impact < -10 or (test_metric == "NPL Ratio" and avg_impact > 10):
        st.warning("Moderate Impact Detected")
        recommendations = [
            "Monitor vulnerable banks more closely", 
            "Recommend increased provisioning for affected institutions",
            "Review stress testing methodology regularly",
            "Conduct targeted inspections of high-risk portfolios"
        ]
    else:
        st.success("Limited Impact Detected")
        recommendations = [
            "Maintain regular monitoring activities",
            "Consider including more severe scenarios in future tests",
            "Share results with banks to encourage risk management",
            "Review individual bank exposures to key risk factors"
        ]
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Provide download option
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Stress Test Results (CSV)",
        csv,
        "stress_test_results.csv",
        "text/csv",
        key='download-stress-test'
    )

###############################################################################
# SECTION E: AML/CFT PATTERN ANALYSIS MODULE
###############################################################################

def aml_cft_app():
    """
    Provides AML/CFT pattern analysis and suspicious transaction monitoring.
    This module helps identify patterns indicating illicit activity.
    """
    st.header("AML/CFT Pattern Analysis")
    st.info("Upload transaction data to identify patterns that may indicate "
            "money laundering or financing of terrorism.")
    
    # Risk score threshold
    risk_threshold = st.slider(
        "Risk Score Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.7,
        help="Transactions with risk scores above this threshold will be flagged"
    )
    
    # File uploader for transaction data
    uploaded_file = st.file_uploader(
        "Upload transaction data (CSV)",
        type="csv",
        key="aml_data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the data
            df = pd.read_csv(uploaded_file)
            
            # Display data overview
            st.write("Transaction Data Overview:")
            st.dataframe(df.head())
            
            # Basic data info
            st.write(f"Total transactions: {len(df)}")
            
            # Let user select key columns
            col1, col2 = st.columns(2)
            
            with col1:
                transaction_id = st.selectbox(
                    "Select transaction ID column:",
                    df.columns
                )
                
                amount_col = st.selectbox(
                    "Select transaction amount column:",
                    [col for col in df.columns if col != transaction_id]
                )
            
            with col2:
                date_col = st.selectbox(
                    "Select transaction date column:",
                    [col for col in df.columns if col not in [transaction_id, amount_col]]
                )
                
                # Try to convert to datetime
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    st.success(f"Successfully converted {date_col} to datetime")
                except:
                    st.warning(f"Could not convert {date_col} to datetime. Proceeding with original format.")
            
            # Select additional feature columns for AML analysis
            feature_cols = st.multiselect(
                "Select additional columns for AML analysis:",
                [col for col in df.columns if col not in [transaction_id, amount_col, date_col]],
                default=[col for col in df.columns if col not in [transaction_id, amount_col, date_col]][:5]  # Default to first 5 columns
            )
            
            # Check if SAR column exists (for supervised learning)
            sar_col = None
            potential_sar_cols = [
                col for col in df.columns if any(term in col.lower() 
                for term in ["sar", "suspicious", "flag", "alert", "risk"])
            ]
            
            if potential_sar_cols:
                sar_col = st.selectbox(
                    "Select column indicating suspicious activity (if available):",
                    ["None"] + potential_sar_cols,
                    index=0
                )
                
                if sar_col == "None":
                    sar_col = None
            
            # AML analysis approach
            if sar_col:
                st.info(f"Found potential SAR column: {sar_col}. Will use supervised learning approach.")
                approach = "supervised"
            else:
                st.info("No SAR column found. Will use unsupervised anomaly detection.")
                approach = "unsupervised"
            
            # Run AML analysis button
            if st.button("Run AML Analysis"):
                with st.spinner("Analyzing transactions for suspicious patterns..."):
                    # Call AML analysis function
                    run_aml_analysis(
                        df,
                        transaction_id,
                        amount_col,
                        date_col,
                        feature_cols,
                        sar_col,
                        approach,
                        risk_threshold
                    )
        
        except Exception as e:
            st.error(f"Error processing transaction data: {str(e)}")
            st.info("Please check your data format and try again.")

def run_aml_analysis(df, id_col, amount_col, date_col, feature_cols, sar_col, approach, threshold):
    """
    Runs AML analysis on transaction data to identify suspicious activities.
    
    Args:
        df: DataFrame containing transaction data
        id_col: Column containing transaction IDs
        amount_col: Column containing transaction amounts
        date_col: Column containing transaction dates
        feature_cols: Additional feature columns
        sar_col: Column indicating suspicious activity (if available)
        approach: 'supervised' or 'unsupervised'
        threshold: Risk score threshold for flagging transactions
    """
    # Display analysis parameters
    st.subheader("AML Analysis Parameters")
    st.write(f"Approach: {approach}")
    st.write(f"Risk threshold: {threshold}")
    
    # Create a copy of the data
    analysis_df = df.copy()
    
    # Calculate basic AML risk indicators
    st.subheader("AML Risk Indicators")
    
    # 1. Large transaction indicator
    # Calculate 95th percentile of transaction amounts
    large_threshold = np.percentile(analysis_df[amount_col], 95)
    analysis_df['is_large'] = (analysis_df[amount_col] > large_threshold).astype(int)
    
    # 2. Round amount indicator (e.g., exactly 10000)
    analysis_df['is_round'] = ((analysis_df[amount_col] % 1000) == 0).astype(int)
    
    # 3. Just-below-threshold indicator (e.g., 9900, 9950 - potentially avoiding reporting threshold)
    analysis_df['is_below_threshold'] = (
        (analysis_df[amount_col] > 9000) & 
        (analysis_df[amount_col] < 10000)
    ).astype(int)
    
    # 4. Unusual hour indicator (if timestamp available)
    try:
        if pd.api.types.is_datetime64_any_dtype(analysis_df[date_col]):
            analysis_df['hour'] = analysis_df[date_col].dt.hour
            analysis_df['is_unusual_hour'] = (
                (analysis_df['hour'] >= 23) | 
                (analysis_df['hour'] <= 4)
            ).astype(int)
        else:
            analysis_df['is_unusual_hour'] = 0
    except:
        analysis_df['is_unusual_hour'] = 0
    
    # 5. Velocity indicators (if multiple transactions per entity)
    if 'sender' in feature_cols or 'customer_id' in feature_cols:
        entity_col = 'sender' if 'sender' in feature_cols else 'customer_id'
        
        # Sort by entity and date
        if pd.api.types.is_datetime64_any_dtype(analysis_df[date_col]):
            analysis_df = analysis_df.sort_values([entity_col, date_col])
            
            # Calculate time difference between consecutive transactions
            analysis_df['time_diff'] = analysis_df.groupby(entity_col)[date_col].diff()
            
            # Convert to hours if possible
            try:
                analysis_df['time_diff_hours'] = analysis_df['time_diff'].dt.total_seconds() / 3600
                
                # Flag rapid sequence (multiple transactions within 24 hours)
                analysis_df['is_rapid_sequence'] = (
                    (analysis_df['time_diff_hours'] <= 24) & 
                    (analysis_df['time_diff_hours'].notna())
                ).astype(int)
            except:
                analysis_df['is_rapid_sequence'] = 0
        else:
            analysis_df['is_rapid_sequence'] = 0
    else:
        analysis_df['is_rapid_sequence'] = 0
    
    # Display risk indicators
    indicators_df = analysis_df[[id_col, amount_col, 'is_large', 'is_round', 
                              'is_below_threshold', 'is_unusual_hour', 
                              'is_rapid_sequence']].head(10)
    
    st.write("Sample Risk Indicators:")
    st.dataframe(indicators_df)
    
    # Run analysis based on approach
    if approach == 'supervised' and sar_col:
        # Supervised approach - train a model using known suspicious activities
        run_supervised_aml(analysis_df, id_col, amount_col, date_col, 
                          feature_cols, sar_col, threshold)
    else:
        # Unsupervised approach - use anomaly detection
        run_unsupervised_aml(analysis_df, id_col, amount_col, date_col, 
                            feature_cols, threshold)

def run_supervised_aml(df, id_col, amount_col, date_col, feature_cols, sar_col, threshold):
    """
    Runs supervised AML analysis using known suspicious activities.
    
    Args:
        df: DataFrame with risk indicators
        id_col: Transaction ID column
        amount_col: Transaction amount column
        date_col: Transaction date column
        feature_cols: Additional feature columns
        sar_col: Column indicating suspicious activity
        threshold: Risk score threshold
    """
    # Prepare features for model
    risk_indicators = ['is_large', 'is_round', 'is_below_threshold', 
                      'is_unusual_hour', 'is_rapid_sequence']
    
    # Combine with selected features
    model_features = risk_indicators + feature_cols
    
    # Filter to only include columns that exist
    model_features = [f for f in model_features if f in df.columns]
    
    # Handle categorical features
    categorical_features = [f for f in model_features 
                          if df[f].dtype == 'object' or df[f].dtype.name == 'category']
    
    # Prepare data for model
    X = df[model_features].copy()
    y = df[sar_col].copy()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), [f for f in model_features if f not in categorical_features]),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train a Random Forest model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Display model performance
    st.subheader("AML Model Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")
    
    with col2:
        st.metric("Precision", f"{precision:.2f}")
    
    with col3:
        st.metric("Recall", f"{recall:.2f}")
    
    with col4:
        st.metric("F1 Score", f"{f1:.2f}")
    
    with col5:
        st.metric("AUC", f"{auc:.2f}")
    
    # Apply model to full dataset
    full_proba = model.predict_proba(df[model_features])[:, 1]
    
    # Add risk scores to original data
    results_df = df[[id_col, amount_col, date_col] + 
                    [f for f in feature_cols if f in df.columns]].copy()
    results_df['risk_score'] = full_proba
    results_df['is_suspicious'] = (full_proba >= threshold).astype(int)
    
    # Display results
    st.subheader("Suspicious Activity Detection Results")
    
    # Calculate summary statistics
    total_suspicious = results_df['is_suspicious'].sum()
    suspicious_percentage = (total_suspicious / len(results_df)) * 100
    
    st.write(f"Detected {total_suspicious} suspicious transactions " 
            f"({suspicious_percentage:.2f}% of all transactions)")
    
    # Display top suspicious transactions
    st.subheader("Top High-Risk Transactions")
    
    # Sort by risk score
    suspicious_df = results_df.sort_values('risk_score', ascending=False).head(20)
    
    # Display with highlighting
    st.dataframe(suspicious_df.style.background_gradient(
        subset=['risk_score'],
        cmap='YlOrRd'
    ))
    
    # Provide download option
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Risk Assessment Results (CSV)",
        csv,
        "aml_risk_assessment.csv",
        "text/csv",
        key='download-aml'
    )
    
    # Create risk distribution visualization
    st.subheader("Risk Score Distribution")
    
    fig = px.histogram(
        results_df,
        x='risk_score',
        nbins=50,
        title='Distribution of AML Risk Scores'
    )
    
    # Add threshold line
    fig.add_shape(
        type='line',
        x0=threshold,
        x1=threshold,
        y0=0,
        y1=results_df['risk_score'].value_counts().max(),
        line=dict(
            color='red',
            width=2,
            dash='dash'
        )
    )
    
    # Add annotation
    fig.add_annotation(
        x=threshold,
        y=results_df['risk_score'].value_counts().max()/2,
        text=f'Threshold: {threshold}',
        showarrow=True,
        arrowhead=1
    )
    
    st.plotly_chart(fig)
    
    # Save model to session state
    st.session_state.best_model = model
    st.session_state.best_model_name = "AML Detection Model"

def run_unsupervised_aml(df, id_col, amount_col, date_col, feature_cols, threshold):
    """
    Runs unsupervised AML analysis using anomaly detection.
    
    Args:
        df: DataFrame with risk indicators
        id_col: Transaction ID column
        amount_col: Transaction amount column
        date_col: Transaction date column
        feature_cols: Additional feature columns
        threshold: Anomaly score threshold
    """
    # Prepare features for model
    risk_indicators = ['is_large', 'is_round', 'is_below_threshold', 
                      'is_unusual_hour', 'is_rapid_sequence']
    
    # Combine with selected features
    model_features = risk_indicators + feature_cols
    
    # Filter to only include columns that exist
    model_features = [f for f in model_features if f in df.columns]
    
    # Handle categorical features
    categorical_features = [f for f in model_features 
                          if df[f].dtype == 'object' or df[f].dtype.name == 'category']
    
    # Prepare data for model
    X = df[model_features].copy()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), [f for f in model_features if f not in categorical_features]),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)
    
    # Train multiple anomaly detection models
    # 1. Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_processed)
    
    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    lof.fit(X_processed)
    
    # 3. One-Class SVM (if dataset is not too large)
    if X_processed.shape[0] < 10000:  # Only use for smaller datasets
        ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
        ocsvm.fit(X_processed)
        svm_used = True
    else:
        ocsvm = None
        svm_used = False
    
    # Score anomalies with each model
    # For Isolation Forest: lower scores are more anomalous
    iso_scores = -iso_forest.score_samples(X_processed)
    
    # For LOF: negative of the decision function gives anomaly score
    lof_scores = -lof.decision_function(X_processed)
    
    # For One-Class SVM (if used)
    if svm_used:
        svm_scores = -ocsvm.decision_function(X_processed)
    else:
        svm_scores = np.zeros_like(iso_scores)
    
    # Create ensemble score (average of normalized scores)
    # First, normalize each score set
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
    
    if svm_used:
        svm_scores_norm = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())
        ensemble_scores = (iso_scores_norm + lof_scores_norm + svm_scores_norm) / 3
    else:
        ensemble_scores = (iso_scores_norm + lof_scores_norm) / 2
    
    # Determine anomalies based on threshold
    anomaly_threshold = np.percentile(ensemble_scores, 100 * (1 - threshold))
    is_anomaly = ensemble_scores > anomaly_threshold
    
    # Add scores to original data
    results_df = df[[id_col, amount_col, date_col] + 
                    [f for f in feature_cols if f in df.columns]].copy()
    results_df['anomaly_score'] = ensemble_scores
    results_df['is_suspicious'] = is_anomaly.astype(int)
    
    # Display results
    st.subheader("Suspicious Activity Detection Results")
    
    # Calculate summary statistics
    total_suspicious = results_df['is_suspicious'].sum()
    suspicious_percentage = (total_suspicious / len(results_df)) * 100
    
    st.write(f"Detected {total_suspicious} suspicious transactions " 
            f"({suspicious_percentage:.2f}% of all transactions)")
    
    # Display top suspicious transactions
    st.subheader("Top Anomalous Transactions")
    
    # Sort by anomaly score
    suspicious_df = results_df.sort_values('anomaly_score', ascending=False).head(20)
    
    # Display with highlighting
    st.dataframe(suspicious_df.style.background_gradient(
        subset=['anomaly_score'],
        cmap='YlOrRd'
    ))
    
    # Provide download option
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Anomaly Detection Results (CSV)",
        csv,
        "aml_anomaly_detection.csv",
        "text/csv",
        key='download-anomaly'
    )
    
    # Create anomaly score visualization
    st.subheader("Anomaly Score Distribution")
    
    fig = px.histogram(
        results_df,
        x='anomaly_score',
        nbins=50,
        title='Distribution of AML Anomaly Scores'
    )
    
    # Add threshold line
    fig.add_shape(
        type='line',
        x0=anomaly_threshold,
        x1=anomaly_threshold,
        y0=0,
        y1=results_df['anomaly_score'].value_counts().max(),
        line=dict(
            color='red',
            width=2,
            dash='dash'
        )
    )
    
    # Add annotation
    fig.add_annotation(
        x=anomaly_threshold,
        y=results_df['anomaly_score'].value_counts().max()/2,
        text=f'Threshold: {anomaly_threshold:.2f}',
        showarrow=True,
        arrowhead=1
    )
    
    st.plotly_chart(fig)
    
    # Feature contribution analysis
    st.subheader("Feature Contribution to Anomaly Scores")
    
    # Take top 10 anomalous transactions
    top_anomalies = results_df.nlargest(10, 'anomaly_score')
    
    # Create box plots for numerical features
    numerical_features = [f for f in model_features 
                        if f not in categorical_features and f in df.columns]
    
    for feature in numerical_features[:5]:  # Limit to first 5 features
        fig = px.box(
            df,
            x='is_suspicious',
            y=feature,
            color='is_suspicious',
            points='all',
            title=f'Distribution of {feature} by Suspicious Flag'
        )
        
        st.plotly_chart(fig)
    
    # Save model components to session state
    st.session_state.best_model = {
        'preprocessor': preprocessor,
        'iso_forest': iso_forest,
        'lof': lof,
        'ocsvm': ocsvm if svm_used else None,
        'threshold': anomaly_threshold
    }
    st.session_state.best_model_name = "AML Anomaly Detection Model"

###############################################################################
# SECTION Z: MAIN APPLICATION LAYOUT AND ROUTING
###############################################################################

def main():
    """
    Main function for the Bangladesh Bank ML Application.
    Handles application layout, user authentication, and module routing.
    """
    # Set up the application
    MODES = setup_app()
    
    # Initialize session state
    initialize_session_state()
    
    # Check authentication
    if check_password():
        # Show the main application interface
        st.sidebar.title("Bangladesh Bank ML Application")
        
        # Add module selection
        selected_mode = st.sidebar.selectbox(
            "Select Module",
            list(MODES.values())
        )
        
        # Store current mode in session state
        st.session_state.current_mode = selected_mode
        
        # Display current mode
        st.sidebar.success(f"Current module: {selected_mode}")
        
        # Display logout button
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.experimental_rerun()
        
        # Add information about current session
        st.sidebar.info(
            f"User: {st.session_state.current_user}\n\n"
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}"
        )
        
        # Show application help
        with st.sidebar.expander("Help"):
            st.write("""
            This application provides various machine learning modules for 
            Bangladesh Bank's analytical needs. Each module has specific 
            functionality for addressing different challenges.
            
            - Upload data in CSV format
            - Configure model parameters
            - Train and evaluate models
            - Make predictions on new data
            """)
        
        # Route to appropriate module based on selection
        if selected_mode == MODES["financial_fraud"]:
            fraud_detection_app()
        elif selected_mode == MODES["remittance_forecast"]:
            remittance_forecast_app()
        elif selected_mode == MODES["credit_risk"]:
            credit_risk_app()
        elif selected_mode == MODES["economic_forecast"]:
            economic_forecast_app()
        elif selected_mode == MODES["aml_cft"]:
            aml_cft_app()

# Run the application
if __name__ == "__main__":
    main()
