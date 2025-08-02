
# Ứng dụng Streamlit cho pipeline khoa học dữ liệu của ABC Manufacturing
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io
import base64

# Tiêu đề ứng dụng
st.title("Ứng dụng Khoa học Dữ liệu cho ABC Manufacturing")

# Phần tải dữ liệu
st.header("Tải lên dữ liệu sản phẩm (CSV)")
uploaded_file = st.file_uploader("Chọn tệp CSV (tương tự amazon.csv)", type="csv")

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_csv(uploaded_file)
    st.write("5 dòng đầu tiên của dữ liệu:")
    st.write(df.head())

    # Tiền xử lý dữ liệu
    st.header("Tiền xử lý dữ liệu")
    def clean_price(x):
        try:
            return float(x.replace('₹', '').replace(',', ''))
        except:
            return np.nan

    def clean_percentage(x):
        try:
            return float(x.replace('%', ''))
        except:
            return np.nan

    def clean_rating_count(x):
        try:
            return int(x.replace(',', ''))
        except:
            return np.nan

    df['product_name'] = df['product_name'].fillna('Không xác định')
    df['category'] = df['category'].fillna('Không xác định')
    df['about_product'] = df['about_product'].fillna('Không có mô tả')
    df['user_id'] = df['user_id'].fillna('Không xác định')
    df['user_name'] = df['user_name'].fillna('Không xác định')
    df['review_id'] = df['review_id'].fillna('Không xác định')
    df['review_title'] = df['review_title'].fillna('Không có tiêu đề')
    df['review_content'] = df['review_content'].fillna('Không có đánh giá')

    df['discounted_price'] = df['discounted_price'].apply(clean_price)
    df['actual_price'] = df['actual_price'].apply(clean_price)
    df['discount_percentage'] = df['discount_percentage'].apply(clean_percentage)
    df['rating'] = df['rating'].apply(lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else np.nan)
    df['rating_count'] = df['rating_count'].apply(clean_rating_count)

    df['discounted_price'] = df['discounted_price'].fillna(df['discounted_price'].median())
    df['actual_price'] = df['actual_price'].fillna(df['actual_price'].median())
    df['discount_percentage'] = df['discount_percentage'].fillna(df['discount_percentage'].median())
    df['rating'] = df['rating'].fillna(df['rating'].median())
    df['rating_count'] = df['rating_count'].fillna(df['rating_count'].median())

    df['main_category'] = df['category'].apply(lambda x: x.split('|')[0] if '|' in x else 'Không xác định')
    df['sub_category'] = df['category'].apply(lambda x: x.split('|')[-1] if '|' in x else 'Không xác định')
    df['price_difference'] = df['actual_price'] - df['discounted_price']
    df['popularity_score'] = np.log1p(df['rating_count'])
    df['review_length'] = df['review_content'].apply(lambda x: len(x.split()))

    def cap_outliers(series):
        q1 = series.quantile(0.01)
        q99 = series.quantile(0.99)
        return series.clip(q1, q99)

    df['discounted_price'] = cap_outliers(df['discounted_price'])
    df['actual_price'] = cap_outliers(df['actual_price'])
    df['rating_count'] = cap_outliers(df['rating_count'])

    # Lưu dữ liệu đã tiền xử lý
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    st.download_button(
        label="Tải xuống dữ liệu đã tiền xử lý",
        data=output,
        file_name="amazon_preprocessed.csv",
        mime="text/csv"
    )

    # Hiển thị thống kê
    st.header("Thống kê dữ liệu")
    st.write("Thống kê tóm tắt cho cột số:")
    st.write(df[['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']].describe())

    st.write("Phân bố danh mục chính:")
    st.write(df['main_category'].value_counts())

    st.write("Ma trận tương quan:")
    st.write(df[['discounted_price', 'discount_percentage', 'rating', 'rating_count', 'popularity_score']].corr())

    # Trực quan hóa
    st.header("Trực quan hóa dữ liệu")
    sns.set(style="whitegrid")

    # Biểu đồ 1: Điểm đánh giá trung bình theo danh mục
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    avg_rating_by_category = df.groupby('main_category')['rating'].mean().reset_index()
    sns.barplot(x='main_category', y='rating', data=avg_rating_by_category, palette='Blues_d', ax=ax1)
    ax1.set_title('Điểm đánh giá trung bình theo danh mục chính')
    ax1.set_xlabel('Danh mục chính')
    ax1.set_ylabel('Điểm đánh giá trung bình')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

    # Biểu đồ 2: Tỷ lệ chiết khấu so với điểm đánh giá
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='discount_percentage', y='rating', data=df, color='red', alpha=0.6, ax=ax2)
    ax2.set_title('Tỷ lệ chiết khấu so với điểm đánh giá')
    ax2.set_xlabel('Tỷ lệ chiết khấu (%)')
    ax2.set_ylabel('Điểm đánh giá')
    plt.tight_layout()
    st.pyplot(fig2)

    # Huấn luyện mô hình
    st.header("Dự đoán điểm đánh giá")
    features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count',
                'price_difference', 'popularity_score', 'review_length']
    categorical_features = ['main_category']
    target = 'rating'

    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix='cat')
    feature_columns = features + [col for col in df_encoded.columns if col.startswith('cat_')]
    X = df_encoded[feature_columns]
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write("Đánh giá mô hình Hồi quy tuyến tính:")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"R²: {r2:.4f}")

    # Biểu đồ dự đoán
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax3.set_title('So sánh giá trị thực và dự đoán (Hồi quy tuyến tính)')
    ax3.set_xlabel('Điểm đánh giá thực tế')
    ax3.set_ylabel('Điểm đánh giá dự đoán')
    plt.tight_layout()
    st.pyplot(fig3)

    # Hệ số hồi quy
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr_model.coef_
    }).sort_values('coefficient', ascending=False)

    st.write("Top 5 đặc trưng với hệ số cao nhất:")
    st.write(coef_df.head())

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='coefficient', y='feature', data=coef_df, palette='coolwarm', ax=ax4)
    ax4.set_title('Hệ số hồi quy của các đặc trưng (Hồi quy tuyến tính)')
    ax4.set_xlabel('Hệ số')
    ax4.set_ylabel('Đặc trưng')
    plt.tight_layout()
    st.pyplot(fig4)
