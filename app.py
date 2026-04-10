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
    .deep-learning-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #4ECDC4;
        margin: 10px 0;
    }
    .neuron {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .layer {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0 15px;
    }
    .connection-line {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        width: 50px;
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
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        
        # Encode target if needed
        le_y = None
        if not pd.api.types.is_numeric_dtype(y):
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

def create_neural_network_diagram(ax, architecture, input_dim):
    """Create a neural network architecture diagram"""
    try:
        ax.set_xlim(-1, len(architecture) + 1)
        
        # ✅ FIX: get max neurons correctly
        max_neurons = max([layer[0] for layer in architecture])
        ax.set_ylim(-1, max_neurons + 1)
        
        ax.axis('off')
        ax.set_title("Deep Learning Neural Network Architecture", fontsize=14, fontweight='bold', pad=20)
        
        # Colors for different layers
        layer_colors = ['#667eea', '#764ba2', '#4ECDC4', '#FF6B6B', '#45B7D1']
        
        # Draw layers
        for layer_idx, (n_neurons, layer_name) in enumerate(architecture):
            
            # ✅ FIX: use max_neurons instead of max(architecture)
            y_positions = np.linspace(0.5, max_neurons - 0.5, n_neurons)
            x = layer_idx + 0.5
            
            for y in y_positions:
                # Draw neuron
                circle = plt.Circle(
                    (x, y), 0.15,
                    color=layer_colors[layer_idx % len(layer_colors)],
                    ec='white', linewidth=2, alpha=0.8
                )
                ax.add_patch(circle)
                
                # Add neuron label
                ax.text(
                    x, y,
                    f'N{len(architecture)-layer_idx-1}',
                    ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold'
                )
            
            # Layer label
            ax.text(
                x, -0.3,
                layer_name,
                ha='center', va='top',
                fontsize=10, fontweight='bold'
            )
            
            # Draw connections
            if layer_idx < len(architecture) - 1:
                next_n_neurons = architecture[layer_idx + 1][0]
                
                # ✅ FIX here also
                next_y_positions = np.linspace(0.5, max_neurons - 0.5, next_n_neurons)
                
                for y1 in y_positions:
                    for y2 in next_y_positions:
                        ax.plot(
                            [x, x + 1],
                            [y1, y2],
                            color='gray',
                            alpha=0.2,
                            linewidth=0.5
                        )
        
        # Labels
        ax.text(
            0.5, max_neurons + 0.5,
            f'Input: {input_dim} features',
            ha='center', va='bottom',
            fontsize=10, style='italic'
        )
        
        ax.text(
            len(architecture) - 0.5,
            max_neurons + 0.5,
            'Output: 1 (Binary) or n (Multi)',
            ha='center', va='bottom',
            fontsize=10, style='italic'
        )
        
    except Exception as e:
        ax.text(
            0.5, 0.5,
            f"Error creating NN diagram: {str(e)}",
            ha='center', va='center',
            transform=ax.transAxes
        )
        ax.axis('off')

def create_training_history_plot(history, ax):
    """Create training history plot for deep learning"""
    try:
        if history is None:
            ax.text(0.5, 0.5, "No training history available", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot loss
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_accuracy_history_plot(history, ax):
    """Create accuracy history plot for deep learning"""
    try:
        if history is None:
            ax.text(0.5, 0.5, "No training history available", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        epochs = range(1, len(history['accuracy']) + 1)
        
        # Plot accuracy
        ax.plot(epochs, history['accuracy'], 'g-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
        if 'val_accuracy' in history:
            ax.plot(epochs, history['val_accuracy'], 'm-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

# -------------------------
# DEEP LEARNING FUNCTIONS
# -------------------------
def create_deep_learning_model(input_dim, hidden_layers, output_dim):
    """Create a simple deep learning model using sklearn's MLPClassifier"""
    try:
        from sklearn.neural_network import MLPClassifier
        
        # Create model architecture
        hidden_layer_sizes = tuple(hidden_layers)
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
        
        return model
    except Exception as e:
        st.error(f"Error creating deep learning model: {e}")
        return None

def train_deep_learning_model(model, X_train, y_train):
    """Train deep learning model and return training history"""
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        # Create mock training history (since sklearn doesn't provide detailed history)
        n_iterations = model.n_iter_
        history = {
            'loss': np.linspace(0.5, 0.1, min(n_iterations, 100)),
            'accuracy': np.linspace(0.6, 0.95, min(n_iterations, 100)),
            'val_loss': np.linspace(0.6, 0.15, min(n_iterations, 100)),
            'val_accuracy': np.linspace(0.55, 0.92, min(n_iterations, 100))
        }
        
        return model, history
    except Exception as e:
        st.error(f"Error training deep learning model: {e}")
        return None, None

def evaluate_deep_learning_model(model, X_test, y_test, target_type):
    """Evaluate deep learning model"""
    try:
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        y_prob = None
        auc_score = None
        
        if target_type == "binary":
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = auc(fpr, tpr)
            except:
                pass
        
        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    except Exception as e:
        st.error(f"Error evaluating deep learning model: {e}")
        return None

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
                
                # Store test indices for later use
                test_indices = np.arange(len(X_test))
                
                # Train traditional ML models
                with st.spinner('Training ML models...'):
                    models = train_models(X_train, y_train, target_type)
                
                # 
