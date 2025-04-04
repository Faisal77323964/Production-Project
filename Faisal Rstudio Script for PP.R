install.packages("tidyverse")
install.packages("mice")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("Amelia")
install.packages("FNN")
install.packages("e1071")
install.packages("rpart")
install.packages("randomForest")
install.packages("ROCR")
install.packages("pROC")

# Load necessary libraries
library(tidyverse)  # For data manipulation and visualization
library(mice)       # For handling missing values
library(corrplot)   # For correlation matrix visualization
library(ggplot2)    # For plotting
library(Amelia)     # For visualizing missing values
library(class)
library(FNN)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)
library(pROC)

# Read the dataset
data <- read.csv("Phishing_Legitimate.csv", na.strings = c("", "NA"))

# View basic structure
str(data)
summary(data)

# Check missing values per column
missing_values <- colSums(is.na(data))
print(missing_values)

# Calculate percentage of missing values
missing_percentage <- (missing_values / nrow(data)) * 100
print(missing_percentage)

# Decision to handle missing values
if (max(missing_percentage) > 30) {
  data <- data[, which(missing_percentage <= 30)]  # Drop columns with >30% missing data
  cat("Dropped columns with more than 30% missing values.\n")
} else if (mean(missing_percentage) > 1) {
  data <- complete(mice(data, method = "pmm", m = 5))  # Impute if avg missing >1%
  cat("Missing values imputed using mice().\n")
} else {
  data <- na.omit(data)  # Remove rows with missing values if <1% overall
  cat("Rows with missing values removed.\n")
}

cat("Final dataset dimensions:", dim(data), "\n")

# Boxplot to visualize outliers
boxplot(data, main = "Boxplot for Outlier Detection", col = "lightblue")

# Find outliers using IQR method
Q1 <- apply(data, 2, quantile, 0.25, na.rm = TRUE)
Q3 <- apply(data, 2, quantile, 0.75, na.rm = TRUE)
IQR_values <- Q3 - Q1

# Define outlier limits
lower_bound <- Q1 - 1.5 * IQR_values
upper_bound <- Q3 + 1.5 * IQR_values

outliers <- data < lower_bound | data > upper_bound

cat("Total outliers identified:", sum(outliers, na.rm = TRUE), "\n")

# Calculate correlation matrix
cor_matrix <- cor(data, use = "pairwise.complete.obs")

# Visualize correlation
corrplot(cor_matrix, method = "color", tl.cex = 0.7)

# Identify constant (low variance) variables
low_variance <- apply(data, 2, var, na.rm = TRUE) == 0
data <- data[, !low_variance]

# Check unique values per column
unique_values <- sapply(data, function(x) length(unique(x)))
print(unique_values)

# Remove variables with very few unique values (e.g., identifier columns)
data <- data[, unique_values > 1]
cat("Low-variance and identifier columns removed.\n")

# Min-Max Scaling Function
min_max_scaling <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

# Apply Min-Max Scaling to Numeric Columns
for (col in names(data)) {
  if (is.numeric(data[[col]])) {
    data[[col]] <- min_max_scaling(data[[col]])
  }
}







# Logistic Regression Model

# Data Preparation
data_cleaned$CLASS_LABEL <- as.factor(data_cleaned$CLASS_LABEL)

# Data Partitioning (80% training, 20% testing)
set.seed(123)
train_index <- sample(1:nrow(data_cleaned), 0.8 * nrow(data_cleaned))
train_data <- data_cleaned[train_index, ]
test_data <- data_cleaned[-train_index, ]



# Model Training
logistic_model <- glm(CLASS_LABEL ~ ., data = train_data, family = "binomial")


# Model Testing
pred_logistic <- predict(logistic_model, test_data, type = "response")
pred_logistic_class <- ifelse(pred_logistic > 0.5, 1, 0)



# Model Evaluation
conf_matrix_logistic <- table(Predicted = pred_logistic_class, Actual = test_data$CLASS_LABEL)
print(conf_matrix_logistic)



# Model Evaluation metrics
accuracy_logistic <- sum(diag(conf_matrix_logistic)) / sum(conf_matrix_logistic)
print(paste("Accuracy:", accuracy_logistic))



# ROC Curve
library(pROC)
roc_curve <- roc(test_data$CLASS_LABEL, pred_logistic)
plot(roc_curve)




#KNN

install.packages("FNN")

# Load necessary libraries
if (!require("class")) install.packages("class")
if (!require("ggplot2")) install.packages("ggplot2")
library(class)
library(ggplot2)

# Load the FNN package for knn.reg function
library(FNN)


# Example Data (replace this with your actual dataset)
# For classification, you might have a categorical target (e.g., 'Class')
# For regression, you might have a continuous target (e.g., 'Value')

# Classification Example (binary classification)
set.seed(123)
classification_data <- data.frame(
  feature1 = rnorm(100),
  feature2 = rnorm(100),
  CLASS_LABEL = sample(c("Class1", "Class2"), 100, replace = TRUE)
)




# Split into training and testing data
train_data_class <- classification_data[1:80, ]
test_data_class <- classification_data[81:100, ]




# Data Preparation
train_data_knn_class <- train_data_class[, -which(names(train_data_class) == "CLASS_LABEL")]
test_data_knn_class <- test_data_class[, -which(names(test_data_class) == "CLASS_LABEL")]
train_label_knn_class <- train_data_class$CLASS_LABEL
test_label_knn_class <- test_data_class$CLASS_LABEL



# Model Training for Classification
k_value_class <- 5
knn_model_class <- knn(train = train_data_knn_class, test = test_data_knn_class, cl = train_label_knn_class, k = k_value_class)



# Model Evaluation for Classification
conf_matrix_class <- table(Predicted = knn_model_class, Actual = test_label_knn_class)
print(conf_matrix_class)

accuracy_class <- sum(diag(conf_matrix_class)) / sum(conf_matrix_class)
print(paste("Accuracy for Classification:", accuracy_class))



# Plot Predicted Classes (Classification)
test_data_class$predicted_class <- knn_model_class
ggplot(test_data_class, aes(x = feature1, y = feature2, color = predicted_class)) + 
  geom_point() + 
  labs(title = "KNN Classification Predicted Classes")


# Regression Example (continuous target)
set.seed(123)
regression_data <- data.frame(
  feature1 = rnorm(100),
  feature2 = rnorm(100),
  target_value = rnorm(100)
)



# Split into training and testing data
train_data_reg <- regression_data[1:80, ]
test_data_reg <- regression_data[81:100, ]


# Data Preparation for Regression
train_data_knn_reg <- train_data_reg[, -which(names(train_data_reg) == "target_value")]
test_data_knn_reg <- test_data_reg[, -which(names(test_data_reg) == "target_value")]
train_label_knn_reg <- train_data_reg$target_value
test_label_knn_reg <- test_data_reg$target_value



# Model Training for Regression
k_value_reg <- 5
knn_model_reg <- knn.reg(train = train_data_knn_reg, test = test_data_knn_reg, y = train_label_knn_reg, k = k_value_reg)

# Model Evaluation for Regression
predictions_reg <- knn_model_reg$pred
mse_reg <- mean((predictions_reg - test_label_knn_reg)^2)  # Mean Squared Error
print(paste("Mean Squared Error for Regression:", mse_reg))

# Plot Predicted vs Actual for Regression
ggplot(data.frame(actual = test_label_knn_reg, predicted = predictions_reg), aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "KNN Regression: Predicted vs Actual", x = "Actual Values", y = "Predicted Values")



#Naive Bayes

# Install and load required packages
if (!require("e1071")) install.packages("e1071")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("ROCR")) install.packages("ROCR")
library(e1071)
library(ggplot2)
library(ROCR)



# Data Preparation --------------------------------------------------------
# Assuming you have pre-loaded 'train_data' and 'test_data' with CLASS_LABEL column

# Remove CLASS_LABEL from features
train_data_nb <- train_data[, -which(names(train_data) == "CLASS_LABEL")]
test_data_nb <- test_data[, -which(names(test_data) == "CLASS_LABEL")]



# Extract labels
train_label_nb <- train_data$CLASS_LABEL
test_label_nb <- test_data$CLASS_LABEL



# Model Training ----------------------------------------------------------
nb_model <- naiveBayes(train_data_nb, train_label_nb)



# Model Testing -----------------------------------------------------------
pred_nb <- predict(nb_model, test_data_nb)



# Basic Evaluation --------------------------------------------------------
# Confusion Matrix
conf_matrix_nb <- table(Predicted = pred_nb, Actual = test_label_nb)
print("Confusion Matrix:")
print(conf_matrix_nb)



# Accuracy Calculation
accuracy_nb <- sum(diag(conf_matrix_nb)) / sum(conf_matrix_nb)
print(paste("Accuracy:", round(accuracy_nb, 4)))



# Confusion Matrix Visualization ------------------------------------------
conf_matrix_df <- as.data.frame(as.table(conf_matrix_nb))
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Freq")



ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#2c7bb6") +
  labs(title = "Naïve Bayes Confusion Matrix",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))



# ROC Curve and AUC -------------------------------------------------------
# Get predicted probabilities
pred_prob_nb <- predict(nb_model, test_data_nb, type = "raw")



# Create prediction object (assuming binary classification)
pred_obj <- prediction(pred_prob_nb[,2], test_label_nb)



# Calculate performance metrics
roc_perf <- performance(pred_obj, "tpr", "fpr")
auc_perf <- performance(pred_obj, "auc")




# Plot ROC curve
plot(roc_perf,
     main = "ROC Curve for Naïve Bayes Model",
     col = "#2c7bb6",
     lwd = 2,
     xlab = "False Positive Rate (FPR)",
     ylab = "True Positive Rate (TPR)")
abline(a = 0, b = 1, lty = 2, col = "red")
legend("bottomright", 
       legend = paste("AUC =", round(auc_perf@y.values[[1]], 3)),
       col = "#2c7bb6",
       lty = 1,
       lwd = 2,
       cex = 0.8)



# Print AUC value
print(paste("AUC Score:", round(auc_perf@y.values[[1]], 4)))






# Decision Tree Model

# Install and load the required packages
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("reshape2")) install.packages("reshape2")

library(rpart)
library(rpart.plot)
library(ggplot2)
library(reshape2)




# Data Preparation (Assuming the dataset is ready)
train_data_dt <- train_data[, -which(names(train_data) == "CLASS_LABEL")]
test_data_dt <- test_data[, -which(names(test_data) == "CLASS_LABEL")]
train_label_dt <- train_data$CLASS_LABEL
test_label_dt <- test_data$CLASS_LABEL



# Model Training
dt_model <- rpart(CLASS_LABEL ~ ., data = train_data, method = "class")



# Model Testing
pred_dt <- predict(dt_model, test_data, type = "class")



# Model Evaluation
conf_matrix_dt <- table(Predicted = pred_dt, Actual = test_label_dt)
print(conf_matrix_dt)



# Model Evaluation metrics
accuracy_dt <- sum(diag(conf_matrix_dt)) / sum(conf_matrix_dt)
print(paste("Accuracy:", accuracy_dt))



# Confusion Matrix Heatmap
conf_matrix_dt_melt <- melt(conf_matrix_dt)
colnames(conf_matrix_dt_melt) <- c("Predicted", "Actual", "Frequency")

ggplot(conf_matrix_dt_melt, aes(x = Predicted, y = Actual, fill = Frequency)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  labs(title = "Decision Tree Confusion Matrix Heatmap", x = "Predicted", y = "Actual") +
  theme_minimal()



# Plot Decision Tree
rpart.plot(dt_model, main = "Decision Tree")





#Random Forest

# Install and load required packages
if (!require("randomForest")) install.packages("randomForest")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("ROCR")) install.packages("ROCR")
library(randomForest)
library(ggplot2)
library(ROCR)

# Data Preparation --------------------------------------------------------
# Assuming you have pre-loaded 'train_data' and 'test_data' with CLASS_LABEL column



# Prepare features and labels
train_label_rf <- train_data$CLASS_LABEL
test_label_rf <- test_data$CLASS_LABEL




# Model Training ----------------------------------------------------------
rf_model <- randomForest(CLASS_LABEL ~ ., 
                         data = train_data,
                         ntree = 100,
                         importance = TRUE)  # Enable importance calculation




# Model Testing -----------------------------------------------------------
pred_rf <- predict(rf_model, test_data)




# Basic Evaluation --------------------------------------------------------
# Confusion Matrix
conf_matrix_rf <- table(Predicted = pred_rf, Actual = test_label_rf)
print("Confusion Matrix:")
print(conf_matrix_rf)




# Accuracy Calculation
accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
print(paste("Accuracy:", round(accuracy_rf, 4)))




# Confusion Matrix Visualization ------------------------------------------
conf_matrix_df <- as.data.frame(as.table(conf_matrix_rf))
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Freq")


ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#4daf4a") +
  labs(title = "Random Forest Confusion Matrix",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))




# ROC Curve and AUC -------------------------------------------------------
# Get predicted probabilities
pred_prob_rf <- predict(rf_model, test_data, type = "prob")




# Create prediction object (assuming binary classification)
pred_obj <- prediction(pred_prob_rf[,2], test_label_rf)



# Calculate performance metrics
roc_perf <- performance(pred_obj, "tpr", "fpr")
auc_perf <- performance(pred_obj, "auc")



# Plot ROC curve
plot(roc_perf,
     main = "ROC Curve for Random Forest Model",
     col = "#4daf4a",
     lwd = 2,
     xlab = "False Positive Rate (FPR)",
     ylab = "True Positive Rate (TPR)")
abline(a = 0, b = 1, lty = 2, col = "red")
legend("bottomright", 
       legend = paste("AUC =", round(auc_perf@y.values[[1]], 3)),
       col = "#4daf4a",
       lty = 1,
       lwd = 2,
       cex = 0.8)



# Print AUC value
print(paste("AUC Score:", round(auc_perf@y.values[[1]], 4)))



# Enhanced Variable Importance Plot ---------------------------------------
importance_df <- data.frame(
  Feature = rownames(rf_model$importance),
  Importance = rf_model$importance[, "MeanDecreaseGini"]
) |> 
  arrange(desc(Importance)) |> 
  head(20)  # Show top 20 features

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "#4daf4a", alpha = 0.8) +
  coord_flip() +
  labs(title = "Random Forest - Feature Importance (Gini Index)",
       x = "Features",
       y = "Mean Decrease in Gini Index") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))



# Combine Results

# Load the necessary libraries
library(ggplot2)

# Assuming these accuracy values are predefined
accuracy_logistic <- 0.85  # Example accuracy values
accuracy_knn <- 0.80
accuracy_nb <- 0.75
accuracy_dt <- 0.78
accuracy_rf <- 0.88



# Create data frame with models and accuracy values
results <- data.frame(
  Model = c("Logistic Regression", "KNN", "Naïve Bayes", "Decision Tree", "Random Forest"),
  Accuracy = c(accuracy_logistic, accuracy_knn, accuracy_nb, accuracy_dt, accuracy_rf)
)



# Print the results
print(results)




# Assuming accuracy values for different models including Naïve Bayes
accuracy_df <- data.frame(
  Model = c("KNN", "Logistic Regression", "Random Forest", "Decision Tree", "Naïve Bayes"),
  Accuracy = c(accuracy_knn, 0.85, 0.88, 0.82, 0.78)  # Replace with actual values
)


# Create the bar chart with accuracy percentages including Naïve Bayes
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(Accuracy * 100, 1), "%")), 
            vjust = -0.3, size = 5) +  # Positioning the text above the bars
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal()
















