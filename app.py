import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI ML Deep Learning Dashboard", 
    layout="wide",
    page_icon="🤖"
)

# Professional color palette
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE"]

# Set seaborn theme
sns.set_theme(style="whitegrid", palette="husl")

# -------------------------
# CSS STYLING
# -------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_data(uploaded_file):
    """Load CSV data with error handling"""
    try:
        data = pd.read_csv(uploaded_file)
        return data, None
    except Exception as e:
        return None, str(e)

def preprocess_data(data, target_column, feature_columns):
    """Preprocess data for ML models"""
    try:
        # Create copies
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
        # Encode categorical features
        le_dict = {}
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        
        # Encode target if needed
        le_y = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X, y, X_scaled, le_y, scaler
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None, None, None, None

def train_models(X_train, y_train, target_type):
    """Train multiple ML models"""
    models = {}
    
    try:
        # Random Forest
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        
        # Gradient Boosting
        models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100, random_state=42
        )
        
        # Logistic Regression
        models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, random_state=42
        )
        
        # Train all models
        for name, model in models.items():
            model.fit(X_train, y_train)
        
        return models
    except Exception as e:
        st.error(f"Model training error: {e}")
        return {}

def evaluate_models(models, X_test, y_test, target_type):
    """Evaluate all models and return results"""
    results = {}
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if target_type == "binary" else None
            
            # Calculate accuracy
            accuracy = (y_pred == y_test).mean()
            
            # Calculate AUC for binary
            auc_score = None
            if target_type == "binary" and y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = auc(fpr, tpr)
            
            results[name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_prob
            }
        except Exception as e:
            st.error(f"Error evaluating {name}: {e}")
    
    return results

# -------------------------
# VISUALIZATION FUNCTIONS
# -------------------------
def create_pie_chart(data, title, ax):
    """Create a beautiful pie chart"""
    try:
        counts = pd.Series(data).value_counts()
        if len(counts) > len(colors):
            colors_extended = colors * (len(counts) // len(colors) + 1)
        else:
            colors_extended = colors[:len(counts)]
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=counts.index, 
            autopct='%1.1f%%', 
            colors=colors_extended,
            startangle=90,
            explode=[0.02] * len(counts),
            shadow=True
        )
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        # Style the text
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_heatmap(data, title, ax):
    """Create a correlation heatmap"""
    try:
        if data.empty or len(data.columns) < 2:
            ax.text(0.5, 0.5, "Need at least 2 numeric columns\nfor correlation heatmap", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
            return
        
        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr, 
            mask=mask,
            cmap="RdYlBu_r", 
            ax=ax, 
            annot=True, 
            fmt='.2f',
            annot_kws={'size': 6},
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True,
            linewidths=0.5
        )
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.tick_params(axis='both', labelsize=6)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_histogram(data, column, title, ax):
    """Create a histogram with KDE"""
    try:
        if data.empty or column not in data.columns:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        sns.histplot(
            data[column], 
            kde=True, 
            color=np.random.choice(colors), 
            ax=ax, 
            bins=30,
            edgecolor='white',
            alpha=0.7
        )
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(column, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_confusion_matrix(y_true, y_pred, title, ax):
    """Create a confusion matrix heatmap"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        # Create annotations with percentages
        cm_percent = cm.astype('float') / cm.sum() * 100
        annot = np.array([f'{v}\n({p:.1f}%)' for v, p in zip(cm.flatten(), cm_percent.flatten())])
        annot = annot.reshape(cm.shape)
        
        sns.heatmap(
            cm, 
            annot=annot, 
            fmt='', 
            cmap="Blues", 
            ax=ax,
            xticklabels=[f'Pred {i}' for i in range(cm.shape[1])],
            yticklabels=[f'Actual {i}' for i in range(cm.shape[0])],
            cbar_kws={'shrink': 0.8}
        )
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.tick_params(axis='both', labelsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_roc_curve(y_true, y_prob, title, ax):
    """Create ROC curve"""
    try:
        if len(np.unique(y_true)) != 2:
            ax.text(0.5, 0.5, "ROC curve requires\nbinary classification", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color="#4ECDC4", lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
               label='Random Classifier')
        
        ax.fill_between(fpr, tpr, alpha=0.3, color="#4ECDC4")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_bar_chart(data, column, title, ax):
    """Create a bar chart for categorical data"""
    try:
        if data.empty or column not in data.columns:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        counts = data[column].value_counts().head(10)
        bars = ax.bar(range(len(counts)), counts.values, 
                     color=colors[:len(counts)], edgecolor='white', linewidth=0.5)
        
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=8, fontweight='bold')
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_box_plot(data, column, title, ax):
    """Create a box plot"""
    try:
        if data.empty or column not in data.columns:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        bp = ax.boxplot(data[column].dropna(), patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor(np.random.choice(colors))
            patch.set_alpha(0.7)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(column, fontsize=10)
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_violin_plot(data, column, title, ax):
    """Create a violin plot"""
    try:
        if data.empty or column not in data.columns:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        sns.violinplot(x=data[column].dropna(), color=np.random.choice(colors), ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(column, fontsize=10)
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_scatter_plot(data, x_col, y_col, title, ax):
    """Create a scatter plot"""
    try:
        if data.empty or x_col not in data.columns or y_col not in data.columns:
            ax.text(0.5, 0.5, "Invalid columns selected", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        ax.scatter(data[x_col], data[y_col], c=np.random.choice(colors), 
                  alpha=0.6, edgecolors='white', s=50)
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_pairplot_data(data, columns, title, ax):
    """Create a pair plot representation"""
    try:
        if len(columns) < 2 or data.empty:
            ax.text(0.5, 0.5, "Need at least 2 numeric columns\nfor pair plot", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
            return
        
        # Create a mini pairplot using scatter matrix approach
        subset = data[columns[:3]].dropna()  # Take first 3 columns
        
        for i, col in enumerate(subset.columns):
            for j, col2 in enumerate(subset.columns):
                if i != j:
                    ax_sub = ax if i == 0 and j == 1 else None
                    if ax_sub:
                        ax.scatter(subset[col2], subset[col], alpha=0.5, 
                                  c=np.random.choice(colors), s=20)
                        ax.set_xlabel(col2)
                        ax.set_ylabel(col)
                        ax.set_title(f"{col} vs {col2}")
                        break
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

# -------------------------
# MAIN APPLICATION
# -------------------------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🤖 AI + ML + Deep Learning Dashboard
        </h1>
        <p style="color: white; text-align: center; margin: 10px 0 0 0;">
            Upload your CSV file and explore powerful visualizations and ML predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.info("Upload your CSV file to get started!")
    
    # File Uploader
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV File", type=["csv"])
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Process uploaded file
    if uploaded_file is not None:
        # Load data
        data, error = load_data(uploaded_file)
        
        if error:
            st.error(f"Error loading file: {error}")
        else:
            st.session_state.data_loaded = True
            st.session_state.data = data
            
            # Success message
            st.sidebar.success(f"✅ File uploaded successfully!")
            st.sidebar.info(f"📊 Dataset: {data.shape[0]} rows × {data.shape[1]} columns")
            
            # Show data on dashboard
            st.markdown("## 📋 Uploaded CSV Data")
            
            # Data preview tabs
            tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Statistics", "🔍 Data Info"])
            
            with tab1:
                st.dataframe(data.head(20), use_container_width=True)
                
                # Download button
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Original CSV",
                    data=csv,
                    file_name='uploaded_data.csv',
                    mime='text/csv',
                )
            
            with tab2:
                st.write("### Descriptive Statistics")
                st.dataframe(data.describe(), use_container_width=True)
            
            with tab3:
                st.write("### Data Types and Missing Values")
                info_df = pd.DataFrame({
                    'Column': data.columns,
                    'Data Type': data.dtypes.astype(str),
                    'Missing Values': data.isnull().sum(),
                    'Missing %': (data.isnull().sum() / len(data) * 100).round(2),
                    'Unique Values': data.nunique()
                })
                st.dataframe(info_df, use_container_width=True)
            
            # Column selection
            st.markdown("## 🎯 Configuration")
            
            col_config1, col_config2 = st.columns(2)
            
            with col_config1:
                target_column = st.selectbox(
                    "🎯 Select Target Column (Label)", 
                    data.columns,
                    index=len(data.columns)-1 if len(data.columns) > 0 else 0
                )
            
            with col_config2:
                feature_columns = st.multiselect(
                    "📊 Select Feature Columns",
                    [col for col in data.columns if col != target_column],
                    default=[col for col in data.columns if col != target_column]
                )
            
            if not feature_columns:
                st.error("⚠️ Please select at least one feature column!")
                return
            
            # Preprocess data
            X, y, X_scaled, le_y, scaler = preprocess_data(data, target_column, feature_columns)
            
            if X is not None:
                # Detect classification type
                target_type = "binary" if len(np.unique(y)) == 2 else "multiclass"
                st.sidebar.info(f"**Target Type:** {target_type.capitalize()}")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train models
                with st.spinner('Training ML models...'):
                    models = train_models(X_train, y_train, target_type)
                
                # Evaluate models
                results = evaluate_models(models, X_test, y_test, target_type)
                
                # Show model comparison
                st.markdown("## 🏆 Model Performance Comparison")
                
                if results:
                    # Create comparison dataframe
                    comparison_data = []
                    for name, res in results.items():
                        row = {'Model': name, 'Accuracy': res['accuracy']}
                        if res['auc'] is not None:
                            row['AUC'] = res['auc']
                        comparison_data.append(row)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Best model
                    best_model_name = max(results.keys(), 
                                         key=lambda x: results[x]['accuracy'])
                    st.success(f"🏆 Best Model: **{best_model_name}** with {results[best_model_name]['accuracy']:.2%} accuracy")
                
                # ==========================
                # ROW 1 — DATA INSIGHTS
                # ==========================
                st.markdown("## 📊 Row 1 — Data Insights")
                
                row1_col1, row1_col2, row1_col3 = st.columns(3)
                
                with row1_col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    create_pie_chart(y, "Target Distribution", ax1)
                    st.pyplot(fig1, use_container_width=True)
                
                with row1_col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    numeric_data = data.select_dtypes(include=[np.number])
                    numeric_features = [col for col in feature_columns if col in numeric_data.columns]
                    create_heatmap(data[numeric_features] if numeric_features else numeric_data, 
                                  "Feature Correlation Heatmap", ax2)
                    st.pyplot(fig2, use_container_width=True)
                
                with row1_col3:
                    fig3, ax3 = plt.subplots(figsize=(6, 5))
                    first_numeric = numeric_data.columns[0] if not numeric_data.empty else None
                    if first_numeric:
                        create_histogram(data, first_numeric, f"Distribution of {first_numeric}", ax3)
                    else:
                        ax3.text(0.5, 0.5, "No numeric features\navailable", ha='center', va='center')
                        ax3.axis('off')
                    st.pyplot(fig3, use_container_width=True)
                
                # ==========================
                # ROW 2 — MODEL PREDICTIONS
                # ==========================
                st.markdown("## 🤖 Row 2 — Model Predictions")
                
                # Select model for visualization
                selected_model_name = st.selectbox("Select Model for Visualization", list(results.keys()))
                selected_result = results[selected_model_name]
                
                row2_col1, row2_col2, row2_col3 = st.columns(3)
                
                with row2_col1:
                    fig4, ax4 = plt.subplots(figsize=(6, 5))
                    create_confusion_matrix(y_test, selected_result['predictions'], 
                                           f"Confusion Matrix ({selected_model_name})", ax4)
                    st.pyplot(fig4, use_container_width=True)
                
                with row2_col2:
                    fig5, ax5 = plt.subplots(figsize=(6, 5))
                    if target_type == "binary":
                        create_roc_curve(y_test, selected_result['probabilities'], 
                                        f"ROC Curve ({selected_model_name})", ax5)
                    else:
                        ax5.text(0.5, 0.5, "ROC Curve not available\nfor multiclass classification", 
                                ha='center', va='center', fontsize=12)
                        ax5.axis('off')
                    st.pyplot(fig5, use_container_width=True)
                
                with row2_col3:
                    fig6, ax6 = plt.subplots(figsize=(6, 5))
                    if target_type == "binary":
                        sns.histplot(selected_result['probabilities'], bins=20, 
                                    color=np.random.choice(colors), ax=ax6, kde=True)
                        ax6.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                                   label='Threshold (0.5)')
                        ax6.set_xlabel('Prediction Probability', fontsize=10)
                        ax6.set_ylabel('Count', fontsize=10)
                        ax6.set_title(f"Prediction Distribution ({selected_model_name})", 
                                     fontsize=12, fontweight='bold')
                        ax6.legend()
                    else:
                        unique, counts = np.unique(selected_result['predictions'], return_counts=True)
                        ax6.bar(unique.astype(str), counts, color=colors[:len(unique)])
                        ax6.set_xlabel('Class', fontsize=10)
                        ax6.set_ylabel('Count', fontsize=10)
                        ax6.set_title(f"Class Distribution ({selected_model_name})", 
                                     fontsize=12, fontweight='bold')
                    st.pyplot(fig6, use_container_width=True)
                
                # ==========================
                # ROW 3 — ADDITIONAL METRICS
                # ==========================
                st.markdown("## 📈 Row 3 — Additional Insights")
                
                row3_col1, row3_col2, row3_col3 = st.columns(3)
                
                with row3_col1:
                    st.markdown("### 📊 Key Metrics")
                    
                    if target_type == "binary":
                        positive_rate = (np.sum(y) / len(y)) * 100
                        st.metric("Positive Class Rate", f"{positive_rate:.2f}%")
                    else:
                        st.metric("Number of Classes", len(np.unique(y)))
                    
                    st.metric("Total Records", len(data))
                    st.metric("Features Used", len(feature_columns))
                    st.metric("Training Samples", len(X_train))
                    st.metric("Test Samples", len(X_test))
                
                with row3_col2:
                    st.markdown("### 📉 Class Distribution")
                    
                    if target_type == "binary":
                        class_counts = pd.Series(y).value_counts().sort_index()
                        st.bar_chart(class_counts)
                    else:
                        class_counts = pd.Series(y).value_counts().sort_index()
                        st.bar_chart(class_counts)
                    
                    st.write("Class counts:")
                    st.write(class_counts)
                
                with row3_col3:
                    st.markdown("### 📊 Categorical Analysis")
                    
                    categorical_data = data.select_dtypes(exclude=[np.number])
                    if not categorical_data.empty:
                        cat_col = st.selectbox("Select Categorical Column", categorical_data.columns)
                        
                        fig9, ax9 = plt.subplots(figsize=(6, 5))
                        create_bar_chart(data, cat_col, f"Top Categories: {cat_col}", ax9)
                        st.pyplot(fig9, use_container_width=True)
                    else:
                        st.info("No categorical columns found in the dataset")
                
                # ==========================
                # ROW 4 — FEATURE ANALYSIS
                # ==========================
                st.markdown("## 📉 Row 4 — Feature Analysis")
                
                # Select feature for analysis
                numeric_features = list(data.select_dtypes(include=[np.number]).columns)
                
                if numeric_features:
                    col_analysis1, col_analysis2 = st.columns(2)
                    
                    with col_analysis1:
                        selected_feature = st.selectbox("Select Feature for Analysis", numeric_features)
                        
                        fig10, ax10 = plt.subplots(figsize=(7, 5))
                        create_box_plot(data, selected_feature, f"Box Plot: {selected_feature}", ax10)
                        st.pyplot(fig10, use_container_width=True)
                    
                    with col_analysis2:
                        fig11, ax11 = plt.subplots(figsize=(7, 5))
                        create_violin_plot(data, selected_feature, f"Violin Plot: {selected_feature}", ax11)
                        st.pyplot(fig11, use_container_width=True)
                    
                    # Scatter plot
                    if len(numeric_features) >= 2:
                        col_scatter1, col_scatter2 = st.columns(2)
                        
                        with col_scatter1:
                            x_feature = st.selectbox("X-axis Feature", numeric_features, index=0)
                        
                        with col_scatter2:
                            y_feature = st.selectbox("Y-axis Feature", numeric_features, index=1 if len(numeric_features) > 1 else 0)
                        
                        fig12, ax12 = plt.subplots(figsize=(7, 5))
                        create_scatter_plot(data, x_feature, y_feature, 
                                           f"Scatter Plot: {x_feature} vs {y_feature}", ax12)
                        st.pyplot(fig12, use_container_width=True)
                
                # ==========================
                # ROW 5 — EXPORT RESULTS
                # ==========================
                st.markdown("## 💾 Export Predictions")
                
                # Create results dataframe
                results_df = data.copy()
                
                # Add predictions from best model
                best_predictions = results[best_model_name]['predictions']
                best_probabilities = results[best_model_name]['probabilities']
                
                if target_type == "binary":
                    results_df['Predicted_Probability'] = best_probabilities
                    results_df['Predicted_Class'] = best_predictions
                else:
                    for i in range(best_probabilities.shape[1]):
                        results_df[f'Class_{i}_Probability'] = best_probabilities[:, i]
                    results_df['Predicted_Class'] = best_predictions
                
                # Convert back to original labels if label encoder exists
                if le_y:
                    results_df['Predicted_Class'] = le_y.inverse_transform(results_df['Predicted_Class'])
                
                # Show preview
                st.write("### Predictions Preview")
                st.dataframe(results_df.head(20), use_container_width=True)
                
                # Download button
                csv_results = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_results,
                    file_name='predictions_results.csv',
                    mime='text/csv',
                )
                
                # Model comparison chart
                st.write("### Model Accuracy Comparison")
                fig13, ax13 = plt.subplots(figsize=(10, 5))
                
                model_names = list(results.keys())
                accuracies = [results[name]['accuracy'] for name in model_names]
                aucs = [results[name]['auc'] for name in model_names]
                
                x = np.arange(len(model_names))
                width = 0.35
                
                bars1 = ax13.bar(x - width/2, accuracies, width, label='Accuracy', color='#4ECDC4')
                bars2 = ax13.bar(x + width/2, aucs, width, label='AUC', color='#FF6B6B')
                
                ax13.set_xlabel('Model', fontsize=12)
                ax13.set_ylabel('Score', fontsize=12)
                ax13.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
                ax13.set_xticks(x)
                ax13.set_xticklabels(model_names, rotation=45, ha='right')
                ax13.legend()
                ax13.set_ylim([0, 1.1])
                
                # Add value labels
                for bar in bars1:
                    height = bar.get_height()
                    ax13.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                for bar in bars2:
                    if not np.isnan(bar.get_height()):
                        height = bar.get_height()
                        ax13.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                st.pyplot(fig13, use_container_width=True)
    
    else:
        # Show welcome message when no file is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px 0;">
            <h2 style="color: white;">👋 Welcome to AI ML Dashboard!</h2>
            <p style="color: white; font-size: 18px;">
                Please upload a CSV file from the sidebar to get started.
            </p>
            <div style="color: white; margin-top: 30px;">
                <h4>📋 Supported Features:</h4>
                <p>• Upload any CSV file</p>
                <p>• Automatic data preprocessing</p>
                <p>• Multiple ML model training</p>
                <p>• Beautiful visualizations</p>
                <p>• Export predictions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data structure
        st.markdown("### 📊 Expected CSV Format")
        st.info("Your CSV file should have:")
        st.write("- One column for target variable (label)")
        st.write("- One or more feature columns")
        st.write("- Headers in the first row")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45, 50, 55, 60],
            'Income': [50000, 60000, 75000, 80000, 90000, 100000, 110000, 120000],
            'CreditScore': [650, 700, 720, 750, 780, 800, 820, 850],
            'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego'],
            'Churn': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
        })
        
        st.markdown("### 📋 Sample Data Format")
        st.dataframe(sample_data, use_container_width=True)

# Run the application
if __name__ == "__main__":
    main()
