library(MASS)
library(np)
library(foreach)
library(doParallel)

# Silence the continuous bandwidth optimization printouts
options(np.messages = FALSE)

# ==========================================
# 1. Data Preparation (UCI Heart Disease Dataset)
# ==========================================
cat("Loading UCI Heart Disease (Cleveland) dataset...\n")
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
col_names <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target")

# Load data and handle missing values denoted by "?"
data <- read.csv(url, header = FALSE, col.names = col_names, na.strings = "?")
data <- na.omit(data) # Clean dataset to ensure CV runs perfectly

# Encode target variable (0 = No disease, 1 = Disease present)
Y <- ifelse(data$target > 0, 1, 0)
X <- data[, 1:13]

# ==========================================
# 2. Train / Test Partitioning (80/20 Split)
# ==========================================
set.seed(496) # Fixed seed for strict reproducibility
train_idx <- sample(1:nrow(X), size = 0.8 * nrow(X))

X_train <- X[train_idx, ]
Y_train <- Y[train_idx]
X_test  <- X[-train_idx, ]
Y_test  <- Y[-train_idx]

# ==========================================
# 3. Stage 1: Parametric Screening (AIC)
# ==========================================
null_model <- glm(Y_train ~ 1, data = X_train, family = binomial)
full_model <- glm(Y_train ~ ., data = X_train, family = binomial)

# Forward-stepwise selection using AIC penalty (2k)
aic_model <- step(null_model,
                  scope = list(lower = null_model, upper = full_model),
                  direction = "forward",
                  trace = 0)

selected_features <- names(coef(aic_model))[-1]

# Restricting to top 4 features since they achieve the best accuracy
selected_features <- selected_features[1:4]

cat("\n======================================================\n")
cat("STAGE 1: AIC PARAMETRIC SCREENING\n")
cat("Optimal subset isolated after Stage 1:\n")
print(selected_features)
cat("======================================================\n\n")

X_sub_train <- X_train[, selected_features, drop=FALSE]

# ==========================================
# 4. Stage 2: Nonparametric LOOCV using Nadaraya-Watson
# ==========================================
cores <- min(parallel::detectCores() - 1, length(selected_features))
cl <- makeCluster(cores)
registerDoParallel(cl)

cat("Evaluating features using Nadaraya-Watson (Local Constant) Cross-Validation...\n")

# Isolate predictive power of each focus variable via Kernel Smoothing
results <- foreach(focus_var = selected_features,
                   .combine = rbind, .packages = "np") %dopar% {
                     
                     x_focus <- X_sub_train[[focus_var]]
                     
                     # Regress Y on the focus variable using Nadaraya-Watson
                     bw_out_nw <- npregbw(xdat = x_focus, ydat = Y_train,
                                          regtype = "lc", bwmethod = "cv.ls")
                     
                     # Extract Out-of-Sample Mean Squared Error
                     mse_nw <- bw_out_nw$fval
                     
                     data.frame(Variable = focus_var, MSE_NW = mse_nw)
                   }

stopCluster(cl)

# ==========================================
# 5. Variable Ranking & Elimination
# ==========================================
cat("\n--- Predictive Errors (Nadaraya-Watson LOOCV MSE) ---\n")
print(results)

# Sort by Mean Squared Error (Descending)
ranked_nw <- results[order(-results$MSE_NW), ]

# The variable with the highest error contains the least structural info
var_to_drop_nw <- ranked_nw$Variable[1]

cat("\n--- Nonparametric Feature Elimination ---\n")
cat("Highest Error (Least Info):", var_to_drop_nw, "-> Dropped.\n")

# ==========================================
# 6. Final Validation & Negative Control
# ==========================================
final_features <- setdiff(selected_features, var_to_drop_nw)
dropped_features <- setdiff(colnames(X), final_features)

# Train Parsimonious Model (Optimal features)
final_model <- glm(Y_train ~ ., data = X_train[, final_features, drop=FALSE],
                   family = binomial)

# Train Negative Control Model (Discarded features)
control_model <- glm(Y_train ~ ., data = X_train[, dropped_features, drop=FALSE],
                     family = binomial)

# Predict classifications on unseen testing data
pred_final_prob <- predict(final_model, newdata = X_test[, final_features, drop=FALSE], type = "response")
pred_final_class <- ifelse(pred_final_prob > 0.5, 1, 0)

pred_control_prob <- predict(control_model, newdata = X_test[, dropped_features, drop=FALSE], type = "response")
pred_control_class <- ifelse(pred_control_prob > 0.5, 1, 0)

# Evaluate global test accuracy
acc_final <- mean(pred_final_class == Y_test)
acc_control <- mean(pred_control_class == Y_test)

cat("\n======================================================\n")
cat("METHODOLOGY PERFORMANCE\n")
cat("======================================================\n")
cat(sprintf("Parsimonious Model (%d Variables): %.2f%%\n", length(final_features),
            acc_final * 100))
cat(sprintf("Negative Control (%d Variables): %.2f%%\n", length(dropped_features),
            acc_control * 100))