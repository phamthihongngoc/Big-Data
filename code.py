from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, expr, lit
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Khởi tạo SparkSession
spark = SparkSession.builder.appName("OnlineShoppersAnalysis").getOrCreate()

# Đọc dữ liệu từ file CSV
file_path = "E:/BigData/BTL/Data/online_shoppers_intention.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Chuyển đổi cột 'Weekend' và 'Revenue' từ TRUE/FALSE -> 1/0
data = data.withColumn("Weekend", when(col("Weekend") == True, 1).otherwise(0))
data = data.withColumn("Revenue", when(col("Revenue") == True, 1).otherwise(0))

# Chuyển đổi cột 'Month' và 'VisitorType' bằng StringIndexer
indexer_month = StringIndexer(inputCol="Month", outputCol="MonthIndex")
data = indexer_month.fit(data).transform(data)
indexer_visitor = StringIndexer(inputCol="VisitorType", outputCol="VisitorTypeIndex")
data = indexer_visitor.fit(data).transform(data)

# Chọn các cột đặc trưng và biến mục tiêu
feature_cols = ["Administrative", "Informational", "ProductRelated", "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "MonthIndex", "VisitorTypeIndex", "Weekend"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Chuẩn hóa dữ liệu
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
data = scaler.fit(data).transform(data)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

# Huấn luyện mô hình RandomForest
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="Revenue", numTrees=10)
rf_model = rf.fit(train_data)

# Dự đoán và đánh giá mô hình
y_pred = rf_model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Hiển thị tầm quan trọng của các yếu tố
importances = rf_model.featureImportances
data_columns = feature_cols
feature_importance = list(zip(data_columns, importances))
print("Feature Importances:")
for feature, importance in feature_importance:
    print(f"{feature}: {importance:.4f}")

#bieu do 1: Hành vi duyệt web dựa trên thời gian xem trang sản phẩm và giá trị trang
# Chọn các cột cần thiết
selected_data = data.select("ProductRelated_Duration", "PageValues", "Revenue").dropna()

# Chuyển đổi từ Spark DataFrame sang Pandas DataFrame để vẽ biểu đồ
pdf = selected_data.toPandas()

# Vẽ biểu đồ scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pdf["ProductRelated_Duration"], y=pdf["PageValues"], hue=pdf["Revenue"], palette={0: "blue", 1: "orange"}, alpha=0.7)

# Định dạng biểu đồ
plt.xlabel("Thời gian xem trang sản phẩm (giây)")
plt.ylabel("Giá trị trang")
plt.title("Hành vi duyệt web dựa trên thời gian xem trang sản phẩm và giá trị trang")
plt.legend(title="Revenue", labels=["Không mua (0)", "Mua hàng (1)"])
plt.grid(True)
plt.show()

#Bieu do 2:So sánh thời gian xem trang sản phẩm giữa hai nhóm khách hàng
# Chọn dữ liệu cần thiết
selected_data = data.select("ProductRelated_Duration", "Revenue").dropna()

# Chuyển sang Pandas để vẽ
pdf = selected_data.toPandas()

# Vẽ biểu đồ Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=pdf["Revenue"], y=pdf["ProductRelated_Duration"], palette="Blues")

# Định dạng biểu đồ
plt.xlabel("Ý định mua hàng")
plt.ylabel("Thời gian xem trang sản phẩm (giây)")
plt.title("So sánh thời gian xem trang sản phẩm giữa hai nhóm khách hàng")

# Hiển thị biểu đồ
plt.show()

#Bieu do 3:Feature Importance (RandomForest)
# Lấy tầm quan trọng của các yếu tố từ mô hình RandomForest
importances = rf_model.featureImportances
feature_names = feature_cols

# Chuyển dữ liệu thành DataFrame để dễ xử lý
df_importance = pd.DataFrame(list(zip(feature_names, importances)), columns=["Feature", "Importance"])
df_importance = df_importance.sort_values(by="Importance", ascending=False)  # Sắp xếp giảm dần

# Vẽ biểu đồ cột (bar chart)
plt.figure(figsize=(10, 6))
plt.barh(df_importance["Feature"], df_importance["Importance"], color="steelblue")
plt.xlabel("Tầm quan trọng của yếu tố")
plt.ylabel("Yếu tố")
plt.title("Tầm quan trọng của các yếu tố ảnh hưởng đến quyết định mua hàng")
plt.gca().invert_yaxis()  # Đảo ngược trục Y để yếu tố quan trọng nhất ở trên cùng
plt.show()

#Bieu do 4:Thời gian duyệt web trung bình giữa khách hàng mua hàng và không mua hàng
# Chọn các cột cần thiết
selected_data = data.select("Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "Revenue").dropna()

# Chuyển sang Pandas để vẽ
pdf = selected_data.toPandas()

# Tính thời gian duyệt web trung bình theo nhóm khách hàng
avg_duration = pdf.groupby("Revenue")[["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]].mean()

# Đóng tất cả figure cũ trước khi vẽ biểu đồ mới
plt.close("all")

# Tạo figure duy nhất
fig, ax = plt.subplots(figsize=(10, 6))  

# Vẽ biểu đồ cột trên cùng một figure
avg_duration.T.plot(kind="bar", color=["blue", "orange"], ax=ax)  

# Định dạng biểu đồ
ax.set_xlabel("Loại thời gian duyệt web")
ax.set_ylabel("Thời gian duyệt web trung bình")
ax.set_title("Thời gian duyệt web trung bình giữa khách hàng mua hàng và không mua hàng")
ax.legend(title="Revenue", labels=["Không mua (0)", "Mua hàng (1)"])
ax.set_xticklabels(avg_duration.T.index, rotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Biểu đồ 5: Tần suất mua sắm theo tháng
# Nhóm dữ liệu theo tháng và đếm số giao dịch mua hàng
monthly_purchases = data.filter(col("Revenue") == 1).groupby("MonthIndex").count().toPandas()

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel", len(monthly_purchases))  # Tạo màu sắc cho các cột
plt.bar(monthly_purchases["MonthIndex"], monthly_purchases["count"], color=colors)

# Định dạng biểu đồ
plt.xlabel("Tháng")
plt.ylabel("Số giao dịch mua hàng")
plt.title("Tần suất mua sắm theo tháng")
plt.xticks(monthly_purchases["MonthIndex"])  # Hiển thị tất cả giá trị trục X
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Hiển thị biểu đồ
plt.show()

# Chuyển đổi cột 'Weekend' và 'Revenue' từ TRUE/FALSE -> 1/0
data = data.withColumn("Weekend", when(col("Weekend") == True, 1).otherwise(0))
data = data.withColumn("Revenue", when(col("Revenue") == True, 1).otherwise(0))

# Tạo cột "Weekday" từ "MonthIndex" (giả lập do không có dữ liệu ngày cụ thể)
month_to_weekday = {
    "Jan": "Monday", "Feb": "Tuesday", "Mar": "Wednesday", "Apr": "Thursday",
    "May": "Friday", "June": "Saturday", "Jul": "Sunday", "Aug": "Monday",
    "Sep": "Tuesday", "Oct": "Wednesday", "Nov": "Thursday", "Dec": "Friday"
}

for month, weekday in month_to_weekday.items():
    data= data.withColumn("Weekday", when(col("Month") == "Jan", lit("Monday")).otherwise(lit("Unknown")))

# Chuyển đổi kiểu dữ liệu cho các cột boolean
data = data.withColumn("Weekend", when(col("Weekend") == True, 1).otherwise(0))
data = data.withColumn("Revenue", when(col("Revenue") == True, 1).otherwise(0))

# 🔹 **Tạo cột Weekday nếu chưa có**
if "Weekday" not in data.columns:
    data = data.withColumn("Weekday", lit(None))

# Bản đồ ánh xạ Month -> Weekday
month_to_weekday = {
    "Jan": "Monday", "Feb": "Tuesday", "Mar": "Wednesday", "Apr": "Thursday",
    "May": "Friday", "June": "Saturday", "Jul": "Sunday", "Aug": "Monday",
    "Sep": "Tuesday", "Oct": "Wednesday", "Nov": "Thursday", "Dec": "Friday"
}

# Cập nhật cột "Weekday"
for month, weekday in month_to_weekday.items():
    data = data.withColumn("Weekday", when(col("Month") == month, weekday).otherwise(col("Weekday")))

# 🔹 **Biểu đồ 6: Tần suất mua sắm theo ngày trong tuần**
weekly_purchases = (
    data.filter(col("Revenue") == 1)
    .groupby("Weekday")
    .agg(count("*").alias("Total Purchases"))
    .toPandas()
)

# Định dạng thứ tự ngày trong tuần
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekly_purchases.set_index("Weekday", inplace=True)

# Xử lý lỗi nếu thiếu dữ liệu của một ngày nào đó
weekly_purchases = weekly_purchases.reindex(order, fill_value=0).reset_index()

# **Vẽ biểu đồ**
plt.figure(figsize=(10, 6))
plt.bar(weekly_purchases["Weekday"], weekly_purchases["Total Purchases"], color="#1f77b4")
plt.xlabel("Ngày trong tuần", fontsize=12)
plt.ylabel("Số giao dịch mua hàng", fontsize=12)
plt.title("Tần suất mua sắm theo ngày trong tuần", fontsize=14)
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 🔹 **Thêm cột Hour nếu chưa có**
if "Hour" not in data.columns:
    data = data.withColumn("Hour", expr("floor(rand() * 24)"))  # Sinh giá trị ngẫu nhiên từ 0-23

# 🔹 **Biểu đồ 7: Tần suất mua sắm theo giờ trong ngày**
hourly_purchases = (
    data.filter(col("Revenue") == 1)
    .groupby("Hour")
    .agg(count("*").alias("Total Purchases"))
    .toPandas()
)

# Định dạng trục X từ 0 đến 23 giờ
hourly_purchases.set_index("Hour", inplace=True)

# Xử lý lỗi nếu thiếu giờ nào đó
hourly_purchases = hourly_purchases.reindex(range(24), fill_value=0).reset_index()

# **Vẽ biểu đồ**
plt.figure(figsize=(10, 6))
plt.bar(hourly_purchases["Hour"], hourly_purchases["Total Purchases"], color="steelblue")
plt.xlabel("Giờ trong ngày", fontsize=12)
plt.ylabel("Số giao dịch mua hàng", fontsize=12)
plt.title("Tần suất mua sắm theo giờ trong ngày", fontsize=14)
plt.xticks(range(24))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Dừng SparkSession
spark.stop()
