import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.stats.power import FTestAnovaPower
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
data = pd.read_csv('ys1a.csv')

data = data.dropna(subset=["ys"])
# Xác định các biến độc lập và phụ thuộc
X = data[['vec', 'deltachi', 'delta', 'deltahmix', 'deltasmix']]
y = data['ys']

# Thông số thiết kế thí nghiệm
alpha = 0.05  # Mức ý nghĩa
power = 0.8   # Power
effect_size = 0.5  # Ước tính kích thước hiệu ứng (medium effect size)

# Ước tính số lần lặp (replication) cần thiết
anova_power = FTestAnovaPower()
sample_size = anova_power.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=3)
replications = int(np.ceil(sample_size))
print(f"Số lần lặp cần thiết: {replications}")

# Thiết lập thí nghiệm với các tỷ lệ kiểm thử
test_sizes = [0.1, 0.2, 0.3]
results = []

# Lặp qua từng tỷ lệ và từng lần lặp
for test_size in test_sizes:
    for i in range(replications):
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        
        # Huấn luyện mô hình
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = model.predict(X_test)
        
        # Tính RMSE và MAPE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Ghi lại kết quả
        results.append({'test_size': test_size, 'replication': i+1, 'rmse': rmse, 'mape': mape})

# Chuyển kết quả sang DataFrame
results_df = pd.DataFrame(results)

# Xuất kết quả ra file CSV ngay sau khi tạo DataFrame
results_df.to_csv('crd_results.csv', index=False)
print("Kết quả thí nghiệm đã được lưu vào file 'crd_results.csv'.")

# Phân tích ANOVA cho RMSE
anova_rmse = f_oneway(
    results_df[results_df['test_size'] == 0.1]['rmse'],
    results_df[results_df['test_size'] == 0.2]['rmse'],
    results_df[results_df['test_size'] == 0.3]['rmse']
)

# Phân tích ANOVA cho MAPE
anova_mape = f_oneway(
    results_df[results_df['test_size'] == 0.1]['mape'],
    results_df[results_df['test_size'] == 0.2]['mape'],
    results_df[results_df['test_size'] == 0.3]['mape']
)

# In kết quả ANOVA
print("Kết quả ANOVA cho RMSE:", anova_rmse)
print("Kết quả ANOVA cho MAPE:", anova_mape)

# Tạo biểu đồ hộp (Boxplot) cho RMSE và MAPE theo các tỷ lệ kiểm thử
plt.figure(figsize=(10, 6))

# Biểu đồ hộp cho RMSE
plt.subplot(1, 2, 1)
sns.boxplot(x='test_size', y='rmse', data=results_df)
plt.title('Biểu đồ hộp RMSE theo tỷ lệ kiểm thử')
plt.xlabel('Tỷ lệ kiểm thử')
plt.ylabel('RMSE')

# Biểu đồ hộp cho MAPE
plt.subplot(1, 2, 2)
sns.boxplot(x='test_size', y='mape', data=results_df)
plt.title('Biểu đồ hộp MAPE theo tỷ lệ kiểm thử')
plt.xlabel('Tỷ lệ kiểm thử')
plt.ylabel('MAPE')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

# Tạo biểu đồ cột (Bar chart) cho RMSE và MAPE trung bình theo các tỷ lệ kiểm thử
mean_rmse = results_df.groupby('test_size')['rmse'].mean()
mean_mape = results_df.groupby('test_size')['mape'].mean()

plt.figure(figsize=(10, 6))
# Biểu đồ cột cho RMSE
plt.subplot(1, 2, 1)
mean_rmse.plot(kind='bar', color='skyblue')
plt.title('Trung bình RMSE theo tỷ lệ kiểm thử')
plt.xlabel('Tỷ lệ kiểm thử')
plt.ylabel('RMSE')

# Biểu đồ cột cho MAPE
plt.subplot(1, 2, 2)
mean_mape.plot(kind='bar', color='lightgreen')
plt.title('Trung bình MAPE theo tỷ lệ kiểm thử')
plt.xlabel('Tỷ lệ kiểm thử')
plt.ylabel('MAPE')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

# Tạo biểu đồ phân tán (Scatter plot) giữa RMSE và MAPE
plt.figure(figsize=(8, 6))
sns.scatterplot(x='rmse', y='mape', data=results_df, hue='test_size', palette='viridis')
plt.title('Biểu đồ phân tán RMSE vs MAPE')
plt.xlabel('RMSE')
plt.ylabel('MAPE')
plt.legend(title='Tỷ lệ kiểm thử')
plt.show()

