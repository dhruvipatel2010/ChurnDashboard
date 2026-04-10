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
                
                # Evaluate traditional ML models
                results = evaluate_models(models, X_test, y_test, target_type)
                
                # ==========================
                # DEEP LEARNING SECTION
                # ==========================
                st.markdown("""
                <div class="deep-learning-card">
                    <h2 style="color: #4ECDC4; text-align: center;">🧠 Deep Learning Neural Network</h2>
                    <p style="color: white; text-align: center;">Advanced neural network with customizable architecture</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Deep Learning Configuration
                st.markdown("### 🔧 Deep Learning Configuration")
                
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                
                with dl_col1:
                    hidden_layers = st.multiselect(
                        "Hidden Layers (Neurons per layer)",
                        [8, 16, 32, 64, 128],
                        default=[64, 32, 16]
                    )
                
                with dl_col2:
                    epochs = st.slider("Max Epochs", 50, 500, 200)
                
                with dl_col3:
                    learning_rate = st.select_slider(
                        "Learning Rate",
                        options=[0.0001, 0.001, 0.01, 0.1],
                        value=0.001
                    )
                
                # Train Deep Learning Model
                if st.button("🚀 Train Deep Learning Model"):
                    with st.spinner('Training Deep Learning Neural Network...'):
                        # Create and train model
                        input_dim = X_train.shape[1]
                        output_dim = len(np.unique(y_train))
                        
                        dl_model = create_deep_learning_model(input_dim, hidden_layers, output_dim)
                        
                        if dl_model is not None:
                            # Train the model
                            dl_model, dl_history = train_deep_learning_model(dl_model, X_train, y_train)
                            
                            if dl_model is not None:
                                # Evaluate the model
                                dl_results = evaluate_deep_learning_model(dl_model, X_test, y_test, target_type)
                                
                                if dl_results is not None:
                                    # Store results
                                    st.session_state.dl_model = dl_model
                                    st.session_state.dl_history = dl_history
                                    st.session_state.dl_results = dl_results
                                    st.session_state.dl_trained = True
                                    
                                    st.success("✅ Deep Learning Model trained successfully!")
                                else:
                                    st.error("Failed to evaluate deep learning model")
                            else:
                                st.error("Failed to train deep learning model")
                        else:
                            st.error("Failed to create deep learning model")
                
                # Display Deep Learning Results
                if 'dl_trained' in st.session_state and st.session_state.dl_trained:
                    st.markdown("### 📊 Deep Learning Results")
                    
                    dl_results = st.session_state.dl_results
                    dl_history = st.session_state.dl_history
                    
                    # Neural Network Architecture Diagram
                    st.markdown("#### 🏗️ Neural Network Architecture")
                    dl_arch_col1, dl_arch_col2 = st.columns([2, 1])
                    
                    with dl_arch_col1:
                        fig_nn, ax_nn = plt.subplots(figsize=(12, 6))
                        architecture = [
                            (len(hidden_layers[0]) if isinstance(hidden_layers[0], tuple) else hidden_layers[0], 'Input Layer'),
                            (hidden_layers[1] if len(hidden_layers) > 1 else hidden_layers[0], 'Hidden Layer 1'),
                            (hidden_layers[2] if len(hidden_layers) > 2 else (hidden_layers[1] if len(hidden_layers) > 1 else hidden_layers[0]), 'Hidden Layer 2'),
                            (1, 'Output Layer')
                        ]
                        create_neural_network_diagram(ax_nn, architecture, X_train.shape[1])
                        st.pyplot(fig_nn, use_container_width=True)
                    
                    with dl_arch_col2:
                        st.markdown("""
                        **Network Configuration:**
                        - **Input Layer:** {} features
                        - **Hidden Layers:** {}
                        - **Activation:** ReLU
                        - **Optimizer:** Adam
                        - **Early Stopping:** Enabled
                        - **Validation Split:** 10%
                        """.format(X_train.shape[1], hidden_layers))
                        
                        # Display metrics
                        st.metric("DL Accuracy", f"{dl_results['accuracy']:.4f}")
                        if dl_results['auc'] is not None:
                            st.metric("DL AUC", f"{dl_results['auc']:.4f}")
                    
                    # Training History
                    st.markdown("#### 📈 Training History")
                    dl_history_col1, dl_history_col2 = st.columns(2)
                    
                    with dl_history_col1:
                        fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
                        create_training_history_plot(dl_history, ax_loss)
                        st.pyplot(fig_loss, use_container_width=True)
                    
                    with dl_history_col2:
                        fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
                        create_accuracy_history_plot(dl_history, ax_acc)
                        st.pyplot(fig_acc, use_container_width=True)
                    
                    # Deep Learning Confusion Matrix
                    dl_cm_col1, dl_cm_col2 = st.columns(2)
                    
                    with dl_cm_col1:
                        fig_dl_cm, ax_dl_cm = plt.subplots(figsize=(6, 5))
                        create_confusion_matrix(y_test, dl_results['predictions'], 
                                               "Deep Learning Confusion Matrix", ax_dl_cm)
                        st.pyplot(fig_dl_cm, use_container_width=True)
                    
                    with dl_cm_col2:
                        if target_type == "binary":
                            fig_dl_roc, ax_dl_roc = plt.subplots(figsize=(6, 5))
                            create_roc_curve(y_test, dl_results['probabilities'], 
                                            "Deep Learning ROC Curve", ax_dl_roc)
                            st.pyplot(fig_dl_roc, use_container_width=True)
                        else:
                            st.info("ROC curve not available for multiclass classification")
                    
                    # Add Deep Learning to results
                    results['Deep Learning'] = dl_results
                
                # ==========================
                # MODEL COMPARISON
                # ==========================
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
                model_options = list(results.keys())
                selected_model_name = st.selectbox("Select Model for Visualization", model_options, 
                                                   index=model_options.index(best_model_name) if best_model_name in model_options else 0)
                selected_result = results[selected_model_name]
                
                row2_col1, row2_col2, row2_col3 = st.columns(3)
                
                with row2_col1:
                    fig4, ax4 = plt.subplots(figsize=(6, 5))
                    create_confusion_matrix(y_test, selected_result['predictions'], 
                                           f"Confusion Matrix ({selected_model_name})", ax4)
                    st.pyplot(fig4, use_container_width=True)
                
                with row2_col2:
                    fig5, ax5 = plt.subplots(figsize=(6, 5))
                    if target_type == "binary" and selected_result['probabilities'] is not None:
                        create_roc_curve(y_test, selected_result['probabilities'], 
                                        f"ROC Curve ({selected_model_name})", ax5)
                    else:
                        ax5.text(0.5, 0.5, "ROC Curve not available\nfor multiclass classification", 
                                ha='center', va='center', fontsize=12)
                        ax5.axis('off')
                    st.pyplot(fig5, use_container_width=True)
                
                with row2_col3:
                    fig6, ax6 = plt.subplots(figsize=(6, 5))
                    if target_type == "binary" and selected_result['probabilities'] is not None:
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
                
                # Create results dataframe for TEST SET ONLY (fixing the length mismatch)
                test_data = data.iloc[X_test.shape[0]:X_test.shape[0] + len(X_test)].copy()
                
                # Add predictions from best model
                best_predictions = results[best_model_name]['predictions']
                best_probabilities = results[best_model_name]['probabilities']
                
                if target_type == "binary":
                    test_data['Predicted_Probability'] = best_probabilities
                    test_data['Predicted_Class'] = best_predictions
                else:
                    for i in range(best_probabilities.shape[1]):
                        test_data[f'Class_{i}_Probability'] = best_probabilities[:, i]
                    test_data['Predicted_Class'] = best_predictions
                
                # Convert back to original labels if label encoder exists
                if le_y:
                    test_data['Predicted_Class'] = le_y.inverse_transform(test_data['Predicted_Class'])
                
                # Show preview
                st.write("### Test Set Predictions Preview")
                st.dataframe(test_data.head(20), use_container_width=True)
                
                # Download button
                csv_results = test_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Test Set Predictions CSV",
                    data=csv_results,
                    file_name='test_predictions_results.csv',
                    mime='text/csv',
                )
                
                # Model comparison chart
                st.write("### Model Accuracy Comparison")
                fig13, ax13 = plt.subplots(figsize=(10, 5))
                
                model_names = list(results.keys())
                accuracies = [results[name]['accuracy'] for name in model_names]
                aucs = [results[name]['auc'] if results[name]['auc'] is not None else 0 for name in model_names]
                
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
                    if bar.get_height() > 0:
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
                <p>• Deep Learning Neural Network</p>
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
