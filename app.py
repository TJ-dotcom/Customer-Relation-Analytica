import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(page_title="Retail Customer Analysis Tool", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data(file, column_mapping):
    try:
        df = pd.read_csv(file)
        # Rename columns based on user-provided mapping
        df.rename(columns=column_mapping, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def preprocess_data(df):
    try:
        df = df.copy()
        
        # Check for required columns
        required_columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'Description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None, None
        
        # Basic cleaning
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        
        # Convert data types
        df['CustomerID'] = df['CustomerID'].astype(int)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Feature engineering
        df['Total_Price'] = df['Quantity'] * df['UnitPrice']
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        
        # Add seasons
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        df['Season'] = df['Month'].apply(get_season)
        
        # Filter valid transactions
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        
        # Clean text data
        def safe_strip(x):
            return x.strip() if isinstance(x, str) else x
        str_col = df.select_dtypes(include=['object']).columns
        df[str_col] = df[str_col].applymap(safe_strip)
        
        def clean_txt(text):
            return re.sub(r'[^\w\s]', '', str(text))
        txt_col = df.select_dtypes(include=['object']).columns
        for column in txt_col:
            try:
                df[column] = df[column].apply(clean_txt)
            except KeyError:
                print(f"Column '{column}' not found in the DataFrame. Skipping.")
        
        # Product categorization
        def product_categorize(description):
            if 'GIFT' in description.upper():
                return 'Gift'
            elif 'SET' in description.upper():
                return 'Set'
            else:
                return 'Regular'
        df['ProductCategory'] = df['Description'].apply(product_categorize)
        
        # Remove outliers
        Q1 = df['Total_Price'].quantile(0.25)
        Q3 = df['Total_Price'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['Total_Price'] >= Q1 - 1.5 * IQR) & (df['Total_Price'] <= Q3 + 1.5 * IQR)]
        
        # Remove non-product entries
        def valid_product(desc):
            invalid_desc = ['damaged', 'sample', 'postage', 'lost', 'found', 'check', 'adjustment']
            return not any(pattern in desc.lower() for pattern in invalid_desc)
        df = df[df['Description'].apply(valid_product)]
        
        # Handle misplaced decimals
        avg_prices = df.groupby('StockCode')['UnitPrice'].median()
        def correct_decimal(price, avg_price):
            if pd.notnull(price) and pd.notnull(avg_price):
                if price > 100 * avg_price:
                    return price / 100
                elif price < avg_price / 100:
                    return price * 100
            return price
        df['UnitPrice'] = df.apply(lambda row: correct_decimal(row['UnitPrice'], avg_prices.get(row['StockCode'], np.nan)), axis=1)
        
        # Customer segmentation
        customer_totals = df.groupby('CustomerID')['Total_Price'].sum()
        def categorize_customer(total_price):
            if total_price < 100:
                return 'Low'
            elif total_price < 1000:
                return 'Medium'
            else:
                return 'High'
        df['CustomerSegment'] = df['CustomerID'].map(customer_totals).apply(categorize_customer)
        
        # Purchase frequency
        purchase_frequency = df.groupby('CustomerID')['InvoiceNo'].nunique()
        df['PurchaseFrequency'] = df['CustomerID'].map(purchase_frequency)
        
        # RFM Analysis
        max_date = df['InvoiceDate'].max()
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
            'InvoiceNo': 'count',  # Frequency
            'Total_Price': 'sum'  # Monetary
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        rfm = rfm.reset_index()
        
        return df, rfm
        
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None

# Main interface
st.title("🛍️ Retail Customer Analysis Tool")
st.write("""
Upload your retail transaction data to gain insights into customer behavior and segmentation.
The tool will help you understand your customers better through RFM analysis and clustering.
""")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file:
    # Ask user to map their columns to expected columns
    column_mapping = {
        'CustomerID': st.text_input("Column for Customer ID", value="CustomerID"),
        'InvoiceDate': st.text_input("Column for Invoice Date", value="InvoiceDate"),
        'Quantity': st.text_input("Column for Quantity", value="Quantity"),
        'UnitPrice': st.text_input("Column for Unit Price", value="UnitPrice"),
        'Description': st.text_input("Column for Description", value="Description")
    }
    
    df = load_data(uploaded_file, column_mapping)
    if df is not None:
        df, rfm = preprocess_data(df)
        
        # Dashboard layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Data Overview")
            st.write("Sample of your data:")
            st.dataframe(df.head())
            
            st.write("Basic Statistics:")
            st.dataframe(df.drop(columns=['InvoiceDate']).describe())
        
        with col2:
            st.subheader("📈 Key Metrics")
            total_customers = len(df['CustomerID'].unique())
            total_transactions = len(df)
            total_revenue = df['Total_Price'].sum()
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("Total Customers", f"{total_customers:,}")
            metrics_col2.metric("Total Transactions", f"{total_transactions:,}")
            metrics_col3.metric("Total Revenue", f"${total_revenue:,.2f}")
        

        
        # Normalize RFM metrics for clustering
        scaler = StandardScaler()
        rfm_normalized = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm['Segment'] = kmeans.fit_predict(rfm_normalized)
        
        # Define a mapping from numeric labels to meaningful names
        cluster_names = {
            0: 'High Value',
            1: 'Loyal Customers',
            2: 'Potential Loyalists',
            3: 'At Risk'
        }
        
        # Apply the mapping to the 'Segment' column
        rfm['Segment'] = rfm['Segment'].map(cluster_names)
        
        # Define a color mapping for the segments
        segment_color_map = {
            'High Value': 'red',
            'Loyal Customers': 'blue',
            'Potential Loyalists': 'green',
            'At Risk': 'orange'
        }
        
        # Map the segment names to colors
        rfm['SegmentColor'] = rfm['Segment'].map(segment_color_map)
        
        # Input fields for prediction
        st.subheader("🔍 Predict Customer Segment")
        recency = st.number_input("Recency (days)", min_value=0, max_value=365, value=30)
        frequency = st.number_input("Frequency (number of purchases)", min_value=0, max_value=8000, value=10)
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, max_value=80000.0, value=500.0)
        
        # Scale the input features
        input_features = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_features)
        
        # Predict the cluster
        predicted_cluster = kmeans.predict(input_scaled)
        predicted_cluster_name = cluster_names[predicted_cluster[0]]
        
        # Visualize the input point on a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the clusters
        scatter = ax.scatter(
            rfm['Recency'], 
            rfm['Frequency'], 
            rfm['Monetary'], 
            c=rfm['SegmentColor'],  # Use the mapped colors
            alpha=0.6, 
            edgecolors='w', 
            s=100
        )
        
        # Plot the input point with a distinct color
        ax.scatter(recency, frequency, monetary, color='purple', label='Input Point', s=200, edgecolors='k')
        
        # Plot cluster centers
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='black', marker='X', s=300, label='Centroids')
        
        # Set labels and title
        ax.set_title('K-Means Clustering: Recency vs Frequency vs Monetary')
        ax.set_xlabel('Recency (days)')
        ax.set_ylabel('Frequency (no of purchases)')
        ax.set_zlabel('Monetary (total spend)')
        
        # Create a custom legend
        custom_legend = [
            plt.Line2D([0], [0], marker='o', color='w', label='High Value', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Loyal Customers', markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Potential Loyalists', markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='At Risk', markerfacecolor='orange', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Input Point', markerfacecolor='purple', markersize=10),
            plt.Line2D([0], [0], marker='X', color='w', label='Centroids', markerfacecolor='black', markersize=10)
        ]
        
        ax.legend(handles=custom_legend, loc='upper right')
        
        st.pyplot(fig)
        
        st.write(f"The predicted customer must be: {predicted_cluster_name}")
        
        # Segment Analysis
        st.subheader("📊 Segment Analysis")
        segment_analysis = rfm.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        
        # Create segment descriptions
        segment_analysis['Description'] = segment_analysis.apply(
            lambda x: f"Customers who spent ${x['Monetary']:,.2f} on average, "
                     f"made {x['Frequency']:.1f} purchases, "
                     f"and last visited {x['Recency']:.0f} days ago",
            axis=1
        )
        
        st.dataframe(segment_analysis)
        
        # Download segmented customer data
        rfm_download = rfm.reset_index()
        st.download_button(
            label="Download Customer Segments",
            data=rfm_download.to_csv(index=False),
            file_name="customer_segments.csv",
            mime="text/csv"
        )
        
        # Customer Behavior Analysis
        st.subheader("📅 Customer Behavior Analysis")
        
        # Time-based analysis
        df['Year-Month'] = df['InvoiceDate'].dt.to_period('M')
        monthly_sales = df.groupby('Year-Month')['Total_Price'].sum().reset_index()
        monthly_sales['Year-Month'] = monthly_sales['Year-Month'].astype(str)
        fig = px.line(monthly_sales, x='Year-Month', y='Total_Price',
                     title='Monthly Sales Trend')
        st.plotly_chart(fig)
        
        # Customer Segmentation Distribution
        st.subheader("📊 Customer Segmentation Distribution")
        segment_counts = rfm['Segment'].value_counts()
        fig = px.pie(values=segment_counts, names=segment_counts.index, title='Customer Segmentation Distribution')
        st.plotly_chart(fig)
        
        # Monthly Revenue Growth
        st.subheader("📈 Monthly Revenue Growth")
        monthly_growth = monthly_sales.copy()
        monthly_growth['Growth'] = monthly_growth['Total_Price'].pct_change().fillna(0) * 100
        fig = px.bar(monthly_growth, x='Year-Month', y='Growth', title='Monthly Revenue Growth (%)')
        st.plotly_chart(fig)
        
        # Top Products
        st.subheader("🏆 Top Products")
        top_products = df.groupby('Description')['Total_Price'].sum().nlargest(10).reset_index()
        fig = px.bar(top_products, x='Total_Price', y='Description', orientation='h', title='Top 10 Products by Sales')
        st.plotly_chart(fig)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        for segment_name in cluster_names.values():
            segment_stats = segment_analysis.loc[segment_name]
            st.write(f"**Segment {segment_name}:**")
            if segment_stats['Recency'] < rfm['Recency'].mean():
                st.write("- These are active customers. Consider cross-selling and loyalty programs.")
            else:
                st.write("- These customers need reactivation. Consider special promotions.")
            if segment_stats['Monetary'] > rfm['Monetary'].mean():
                st.write("- High-value customers. Provide VIP treatment and personalized service.")
        
else:
    st.info("Please upload your retail transaction data (CSV format) to begin analysis.")
    st.write("""
    Your CSV file should include columns for Customer ID, Invoice Date, Quantity, Unit Price, and Description.
    """)