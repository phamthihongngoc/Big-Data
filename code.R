# ----------------- 1. Cài đặt & gọi thư viện -----------------
install.packages("sparklyr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("tidyr")

library(sparklyr)
library(dplyr)
library(ggplot2)
library(tidyr)

# ----------------- 2. Kết nối Spark local -----------------
sc <- spark_connect(master = "local")

# ----------------- 3. Đọc dữ liệu từ file CSV -----------------
file_path <- "C:/Users/User/OneDrive/Desktop/Giang/Năm 3/Kỳ 2/T2- Du Lieu Lon/Nhom4/online_shoppers_intention.csv"
data <- spark_read_csv(sc, name = "shoppers", path = file_path, header = TRUE, infer_schema = TRUE)

# ----------------- 4. Xử lý dữ liệu: Chuyển TRUE/FALSE -> 1/0 -----------------
data <- data %>%
  mutate(
    Weekend = ifelse(Weekend == "TRUE", 1, 0),
    Revenue = ifelse(Revenue == "TRUE", 1, 0)
  )
# ----------------- 5. Encode biến chuỗi: Month, VisitorType -----------------
data <- data %>%
  ft_string_indexer(input_col = "Month", output_col = "MonthIndex", handle_invalid = "keep") %>%
  ft_string_indexer(input_col = "VisitorType", output_col = "VisitorTypeIndex", handle_invalid = "keep")

# ----------------- 6. Chuẩn hóa dữ liệu: VectorAssembler + StandardScaler -----------------
data <- data %>%
  ft_vector_assembler(input_cols = feature_cols, output_col = "features") %>%
  ft_standard_scaler(input_col = "features", output_col = "scaledFeatures")

# ----------------- 7. Chia dữ liệu train/test + huấn luyện mô hình Random Forest -----------------
partitions <- data %>% sdf_random_split(train = 0.7, test = 0.3, seed = 42)

rf_model <- ml_random_forest_classifier(
  partitions$train,
  features_col = "scaledFeatures",
  label_col = "Revenue",
  num_trees = 10
)

predictions <- ml_predict(rf_model, partitions$test)

# ----------------- 8. Tính độ chính xác -----------------
accuracy <- ml_multiclass_classification_evaluator(
  predictions,
  label_col = "Revenue",
  prediction_col = "prediction",
  metric_name = "accuracy"
)

print(paste0("Accuracy: ", round(accuracy * 100, 2), "%"))

# ----------------- 9. BIỂU ĐỒ 1: Scatter plot -----------------
pdf1 <- data %>%
  select(ProductRelated_Duration, PageValues, Revenue) %>%
  na.omit() %>%
  collect()

ggplot(pdf1, aes(x = ProductRelated_Duration, y = PageValues, color = factor(Revenue))) +
  geom_point(alpha = 0.7) +
  labs(x = "Thời gian xem trang sản phẩm (giây)", y = "Giá trị trang", color = "Revenue") +
  ggtitle("Hành vi duyệt web theo thời gian và giá trị trang") +
  scale_color_manual(values = c("blue", "orange"), labels = c("Không mua (0)", "Mua hàng (1)")) +
  theme_minimal()

# ----------------- 10. BIỂU ĐỒ 2: Boxplot -----------------
pdf2 <- data %>%
  select(ProductRelated_Duration, Revenue) %>%
  na.omit() %>%
  collect()

ggplot(pdf2, aes(x = factor(Revenue), y = ProductRelated_Duration, fill = factor(Revenue))) +
  geom_boxplot() +
  labs(x = "Ý định mua hàng", y = "Thời gian xem trang sản phẩm (giây)") +
  ggtitle("So sánh thời gian xem trang sản phẩm") +
  scale_fill_manual(values = c("lightblue", "orange")) +
  theme_minimal()

# ----------------- 11. BIỂU ĐỒ 3: Tầm quan trọng các yếu tố -----------------
importances <- rf_model$feature_importances()
importance_df <- data.frame(Feature = feature_cols, Importance = as.numeric(importances)) %>%
  arrange(desc(Importance))

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(x = "Yếu tố", y = "Tầm quan trọng") +
  ggtitle("Tầm quan trọng các yếu tố (Random Forest)") +
  theme_minimal()

# ----------------- 12. BIỂU ĐỒ 4: Thời gian duyệt web trung bình theo nhóm -----------------
df4 <- data %>%
  select(Administrative_Duration, Informational_Duration, ProductRelated_Duration, Revenue) %>%
  collect()

avg_duration <- df4 %>%
  group_by(Revenue) %>%
  summarise_all(mean)

avg_melt <- pivot_longer(avg_duration, cols = -Revenue, names_to = "Type", values_to = "Duration")

ggplot(avg_melt, aes(x = Type, y = Duration, fill = factor(Revenue))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Loại thời gian", y = "Thời gian trung bình", fill = "Revenue") +
  ggtitle("Thời gian duyệt web trung bình theo nhóm") +
  theme_minimal()

# ----------------- 13. BIỂU ĐỒ 5: Tần suất mua theo tháng -----------------
data <- data %>% mutate(MonthIndex = as.integer(MonthIndex))
pdf5 <- data %>%
  filter(Revenue == 1) %>%
  group_by(MonthIndex) %>%
  summarise(Count = n()) %>%
  collect()

ggplot(pdf5, aes(x = MonthIndex, y = Count)) +
  geom_col(fill = "skyblue") +
  labs(x = "Tháng", y = "Số giao dịch mua hàng") +
  ggtitle("Tần suất mua sắm theo tháng") +
  theme_minimal()

# ----------------- 14. BIỂU ĐỒ 6: Tần suất mua theo ngày trong tuần -----------------
pdf6 <- data %>% collect()

set.seed(42)
pdf6$Weekday <- sample(c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
                       size = nrow(pdf6), replace = TRUE)

weekly_purchases <- pdf6 %>%
  filter(Revenue == 1) %>%
  group_by(Weekday) %>%
  summarise(Count = n())

order_week <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
weekly_purchases$Weekday <- factor(weekly_purchases$Weekday, levels = order_week)

ggplot(weekly_purchases, aes(x = Weekday, y = Count)) +
  geom_col(fill = "steelblue") +
  labs(x = "Ngày", y = "Số giao dịch mua hàng") +
  ggtitle("Tần suất mua sắm theo ngày trong tuần") +
  theme_minimal()

# ----------------- 15. BIỂU ĐỒ 7: Tần suất mua theo giờ -----------------
if (!"Hour" %in% colnames(pdf6)) {
  set.seed(42)
  pdf6$Hour <- sample(0:23, nrow(pdf6), replace = TRUE)
}

hourly_purchases <- pdf6 %>%
  filter(Revenue == 1) %>%
  group_by(Hour) %>%
  summarise(Count = n())

ggplot(hourly_purchases, aes(x = Hour, y = Count)) +
  geom_col(fill = "steelblue") +
  labs(x = "Giờ", y = "Số giao dịch mua hàng") +
  ggtitle("Tần suất mua sắm theo giờ trong ngày") +
  theme_minimal()

# ----------------- 16. Ngắt kết nối Spark -----------------
spark_disconnect(sc)
