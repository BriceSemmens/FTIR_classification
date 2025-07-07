# Title: FT-IR Analysis using Peak-Sensitive Elastic-Net Logistic Regression
# Description: This script performs a complete, robust analysis pipeline:
#              1. Imports raw rockfish otolith FT-IR data.
#              2. Filters, cleans, and reshapes the data for the outer otolith edge.
#              3. Converts Transmittance to Absorbance.
#              4. **Applies Asymmetrical Least Squares (ALS) Baseline Correction using the 'baseline' package.**
#              5. Applies smoothing and normalization.
#              6. Trains and evaluates the peak-sensitive elastic-net model.
# Date: 2025-07-02

# --- 1. SETUP: Load Libraries ---
# Note the addition of the 'baseline' package.
# install.packages(c("tidyverse", "prospectr", "glmnet", "pROC", "zoo", "baseline"))

library(tidyverse)
library(prospectr) # Still used for savitzkyGolay and standardNormalVariate
library(glmnet)
library(pROC)
library(zoo)
library(baseline)  # The dedicated package for baseline correction

# --- 2. DATA IMPORT & WRANGLING ---
# Why: This creates vectors that act as a "dictionary" to assign each fish to its
# correct location (Los Angeles or San Diego).
raw_data <- read_csv("VR_RockfishData.csv")

la_fish_ids <- c(
  "VR_R_096", "VR_R_105", "VR_R_282", "VR_R_783", "VR_R_798",
  "VR_R_799", "VR_R_804", "VR_R_841", "VR_L_1022", "VR_R_1250",
  "VR_R_1252", "VR_R_1253", "VR_R_1254", "VR_R_1255", "VR_R_1256",
  "VR_R_1260", "VR_R_1264", "VR_R_1275"
)
sd_fish_ids <- c(
  "VR_R_273", "VR_L_457", "VR_R_927", "VR_R_1157", "VR_R_1158",
  "VR_R_1160", "VR_R_1163", "VR_R_1206", "VR_R_1208", "VR_R_1140",
  "VR_R_1141", "VR_R_1143", "VR_R_1146", "VR_R_1150", "VR_R_1211",
  "VR_R_1212", "VR_R_1214"
)

spectra_wide <- raw_data %>%
  # Why: Change column names like "FISH ID" to "FISH_ID" to avoid issues with spaces.
  rename_with(~str_replace_all(., " ", "_")) %>%
  # Why: Convert the data from a "wide" format (many replicate columns) to a "long"
  # format, which is easier to group and summarize.
  pivot_longer(
    cols = contains("Transmittance"),
    names_to = "measurement",
    values_to = "transmittance"
  ) %>%
  # Why: Create new columns to explicitly label the sample type (Core vs. Edge)
  # and the fish's location based on the ID vectors we defined earlier.
  mutate(
    sample_type = case_when(
      str_detect(measurement, "Core") ~ "Core",
      str_detect(measurement, "Outer_Edge") ~ "Edge"
    ),
    location = case_when(
      FISH_ID %in% la_fish_ids ~ "Los Angeles",
      FISH_ID %in% sd_fish_ids ~ "San Diego",
      TRUE ~ NA_character_ # Assign NA to any fish not in our lists.
    )
  ) %>%
  # Why: Remove the fish that don't belong to our two target groups.
  filter(!is.na(location)) %>%
  # Why: For each individual fish and each wavenumber, average the 7 replicate readings.
  # This reduces noise and creates a single, more stable spectrum for each sample.
  group_by(FISH_ID, location, Wavenumber, sample_type) %>%
  summarise(avg_transmittance = mean(transmittance, na.rm = TRUE), .groups = "drop") %>%
  # Why: Pivot the data back to a "wide" format. This is the required input for
  # most modeling functions: one row per sample, one column per variable (wavenumber)
  pivot_wider(
    names_from = Wavenumber,
    values_from = avg_transmittance,
    id_cols = c(FISH_ID, location, sample_type)
  )

edge_spectra <- spectra_wide %>% filter(sample_type == "Edge") %>% select(-sample_type)

# --- 3. PRE-PROCESSING & IMPUTATION ---
# Why: This is the most critical section for ensuring data quality. Raw spectral
# data contains many sources of unwanted variation (e.g., noise, baseline drift).
# These steps remove that variation, allowing the model to focus on the true
# chemical differences between the groups.

# Separate the identifying columns from the numerical spectral data.
identifiers <- edge_spectra[, 1:2]
spectra_matrix_T <- as.matrix(edge_spectra[, 3:ncol(edge_spectra)])

# Step 3.1: Convert from Transmittance to Absorbance
# Why: According to the Beer-Lambert Law, absorbance is linearly proportional to the
# concentration of chemical components. Transmittance is not. Linearizing the data
# this way makes the relationship between the spectra and the sample's chemistry
# much easier for a model to learn.
spectra_matrix_A <- -log10(spectra_matrix_T)

# Step 3.2: Impute missing/infinite values
# Why: The log transformation in the previous step can create infinite values if
# transmittance was zero. The 'glmnet' model cannot handle any missing or non-finite
# values. We replace these with the average value for that specific wavenumber, which
# is a standard and effective way to fill the gaps without distorting the data.
if (any(!is.finite(spectra_matrix_A))) {
  cat("Missing or non-finite values detected. Imputing now...\n")
  spectra_matrix_A[!is.finite(spectra_matrix_A)] <- NA
  spectra_matrix_imputed <- apply(spectra_matrix_A, 2, function(x) na.aggregate(x, FUN = mean))
} else {
  spectra_matrix_imputed <- spectra_matrix_A
  cat("No missing values found.\n")
}

# ** Step 3.3: Apply Asymmetrical Least Squares (ALS) Baseline Correction **
# Using the baseline() function from the 'baseline' package
# This function requires the matrix to have columns as wavenumbers and rows as samples
# Why: Spectra often have a shifting, curved baseline caused by instrument effects or
# light scattering. ALS is a powerful algorithm that fits a flexible baseline to the
# spectrum and subtracts it, making the peak heights more accurate and comparable
# across samples. We use the specialized 'baseline' package for this.
als_corrected <- baseline::baseline(spectra_matrix_imputed, method = 'als', lambda = 6, p = 0.05)
# The corrected spectra are in the `corrected` slot of the output object
spectra_corrected <- getCorrected(als_corrected)


# Step 3.4: Smoothing & Normalization
# Why (Savitzky-Golay): This filter smooths the data by fitting a polynomial to a
# small window of points, effectively reducing high-frequency instrumental noise
# without distorting the underlying peak shapes.
spectra_smoothed <- savitzkyGolay(spectra_corrected, p = 2, w = 11, m = 0)
# Why (Standard Normal Variate): SNV scales each individual spectrum. This removes
# multiplicative effects (e.g., differences in sample thickness) and puts all
# spectra onto a common scale, which is essential for fair comparison in the model.
spectra_processed <- standardNormalVariate(spectra_smoothed)
cat("Pre-processing complete including ALS baseline correction.\n\n")


# --- 4. MODELING: Peak-Sensitive Elastic-Net Logistic Regression ---
# Why: We use elastic net because it is perfectly suited for "wide" data (more
# features than samples). It automatically performs feature selection by forcing the
# coefficients of unimportant wavenumbers to zero, making it "peak-sensitive."

y <- factor(identifiers$location) # The response variable (what we want to predict).
x <- spectra_processed # The predictor variables (the processed spectra).

# Step 4.1: Split the data into training and testing sets.
# Why: This is fundamental for honest model assessment. We train the model on one
# part of the data (the training set) and then test its performance on a separate,
# unseen part (the testing set) to see how well it generalizes.
# `set.seed(123)` ensures the split is the same every time we run the code.
set.seed(123)
train_indices <- sample(1:nrow(x), size = floor(0.80 * nrow(x)))
x_train <- x[train_indices, ]
y_train <- y[train_indices]
x_test <- x[-train_indices, ]
y_test <- y[-train_indices]

# Step 4.2: Train the cross-validated elastic net model.
# Why: 'cv.glmnet' automatically performs cross-validation (by default, 10-fold)
# to find the optimal strength of the regularization penalty (lambda). This prevents
# us from having to guess the best value.
# - family = "binomial": Tells the model to perform logistic regression for a two-class problem.
# - alpha = 0.5: This sets the elastic net mixing parameter, balancing the L1 (Lasso) and L2 (Ridge) penalties.
# - type.measure = "auc": We tell the model to find the lambda that maximizes the Area Under the ROC Curve, a robust metric for classification performance.
cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5, type.measure = "auc")

# --- 5. PERFORMANCE EVALUATION ---
# Why: After training, we must evaluate how well the model performs on the unseen test data.

# Step 5.1: Make predictions on the test set.
# Why: We use 's = "lambda.min"' to select the lambda value that gave the best
# performance during cross-validation.
predictions <- predict(cv_model, newx = x_test, s = "lambda.min", type = "class")
predictions <- factor(predictions, levels = levels(y_test))

# Step 5.2: Create and interpret a confusion matrix.
# Why: This table is the primary way to see the model's performance. It shows how
# many samples were correctly and incorrectly classified for each class.
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
cat("\n--- Model Performance on Outer Edge Data ---\n")
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("\nAccuracy:", round(accuracy, 3), "\n\n")

# Step 5.3: Generate and plot the ROC curve.
# Why: The ROC curve visualizes the trade-off between the true positive rate and the
# false positive rate. The Area Under this Curve (AUC) is a single number summarizing
# the model's overall discriminative power; a value of 1.0 is perfect, and 0.5 is random.
prob_predictions <- predict(cv_model, newx = x_test, s = "lambda.min", type = "response")
roc_curve <- roc(response = y_test, predictor = as.vector(prob_predictions), quiet = TRUE)
plot(roc_curve, main = paste0("ROC Curve (AUC = ", round(auc(roc_curve), 3), ")"))

# --- 6. IDENTIFY IMPORTANT PEAKS (FEATURES) ---
# Why: The great advantage of elastic net is its interpretability. We can look
# "inside" the model to see which wavenumbers (i.e., chemical peaks) it found most
# useful for distinguishing between the two locations.

# Step 6.1: Extract the model coefficients.
# Why: The coefficients represent the "weight" the model gives to each wavenumber.
# The L1 (Lasso) penalty forces most of these coefficients to be exactly zero.
model_coeffs <- coef(cv_model, s = "lambda.min")

# Step 6.2: Filter for the non-zero coefficients.
# Why: The wavenumbers with non-zero coefficients are the ones the model selected
# as being informative for the classification.
coeffs_df <- data.frame(
  wavenumber = as.numeric(model_coeffs@Dimnames[[1]]),
  coefficient = as.numeric(model_coeffs)
) %>%
  filter(coefficient != 0 & !is.na(wavenumber))

cat("\n--- Peak-Sensitive Feature Selection ---\n")
cat("The model selected", nrow(coeffs_df), "out of", ncol(x), "wavenumbers as important.\n\n")
print(coeffs_df)

# Step 6.3: Plot the important features.
# Why: This plot provides a visual summary of the results. It shows which wavenumbers
# were selected and whether their presence corresponds with a higher probability of
# being in one class versus the other (based on the sign of the coefficient).
ggplot(coeffs_df, aes(x = wavenumber, y = coefficient)) +
  geom_col(aes(fill = coefficient > 0)) +
  scale_x_reverse(name = "Wavenumber (cm⁻¹)") +
  labs(
    title = "Important Wavenumbers Selected by the Model (Outer Edge)",
    y = "Coefficient Value"
  ) +
  theme_minimal() +
  guides(fill = "none")