# Title: Final Analysis using PCA + LDA and LOOCV (Corrected)
# Description: This script addresses poor performance on small datasets by:
#              1. Using Principal Component Analysis (PCA) to reduce dimensionality.
#              2. Using Linear Discriminant Analysis (LDA) for classification on the PCs.
#              3. Using robust Leave-One-Out Cross-Validation (LOOCV) for evaluation.
#              4. **Manually creating the final confusion matrix from LOOCV predictions.**
#              5. Visualizing the group separation with a PCA plot.
# Author: Gemini
# Date: 2025-07-02

# --- 1. SETUP: Load Libraries ---
# install.packages(c("tidyverse", "prospectr", "zoo", "baseline", "caret", "e1071"))

library(tidyverse)
library(prospectr)
library(zoo)
library(baseline)
library(caret)     # For streamlined model training and validation

# --- 2. DATA IMPORT & PRE-PROCESSING ---
# Why: This entire section is dedicated to transforming the raw data into a clean,
# pre-processed matrix suitable for machine learning. 

# Step 2.1: Load the raw data and define group IDs.
raw_data <- read_csv("VR_RockfishData.csv")
la_fish_ids <- c("VR_R_096", "VR_R_105", "VR_R_282", "VR_R_783", "VR_R_798", "VR_R_799", "VR_R_804", "VR_R_841", "VR_L_1022", "VR_R_1250", "VR_R_1252", "VR_R_1253", "VR_R_1254", "VR_R_1255", "VR_R_1256", "VR_R_1260", "VR_R_1264", "VR_R_1275")
sd_fish_ids <- c("VR_R_273", "VR_L_457", "VR_R_927", "VR_R_1157", "VR_R_1158", "VR_R_1160", "VR_R_1163", "VR_R_1206", "VR_R_1208", "VR_R_1140", "VR_R_1141", "VR_R_1143", "VR_R_1146", "VR_R_1150", "VR_R_1211", "VR_R_1212", "VR_R_1214")

# Step 2.2: Reshape and clean the data using a 'tidyverse' pipeline.
spectra_wide <- raw_data %>%
  rename_with(~str_replace_all(., " ", "_")) %>%
  pivot_longer(cols = contains("Transmittance"), names_to = "measurement", values_to = "transmittance") %>%
  mutate(
    sample_type = case_when(str_detect(measurement, "Core") ~ "Core", str_detect(measurement, "Outer_Edge") ~ "Edge"),
    location = case_when(FISH_ID %in% la_fish_ids ~ "Los Angeles", FISH_ID %in% sd_fish_ids ~ "San Diego", TRUE ~ NA_character_)
  ) %>%
  # Why: We filter for both !is.na(location) AND sample_type == "Edge" to ensure we only process the desired data.
  filter(!is.na(location), sample_type == "Edge") %>%
  group_by(FISH_ID, location, Wavenumber) %>%
  summarise(avg_transmittance = mean(transmittance, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = Wavenumber, values_from = avg_transmittance, id_cols = c(FISH_ID, location))

# Step 2.3: Separate identifiers from the spectral matrix and convert to Absorbance.
# Why (Absorbance): According to the Beer-Lambert Law, absorbance is linearly proportional to the
# concentration of chemical components. Transmittance is not. Linearizing the data
# this way makes the relationship between the spectra and the sample's chemistry
# much easier for a model to learn.
identifiers <- spectra_wide[, 1:2]
spectra_matrix_T <- as.matrix(spectra_wide[, 3:ncol(spectra_wide)])
spectra_matrix_A <- -log10(spectra_matrix_T)

# Step 2.4: Impute missing values.
# Why: The log transformation in the previous step can create infinite values if
# transmittance was zero. Machine learning models cannot handle any missing or non-finite
# values. We must fill them in before proceeding.
if (any(!is.finite(spectra_matrix_A))) {
  spectra_matrix_A[!is.finite(spectra_matrix_A)] <- NA
  spectra_matrix_imputed <- apply(spectra_matrix_A, 2, function(x) na.aggregate(x, FUN = mean))
} else {
  spectra_matrix_imputed <- spectra_matrix_A
}

# Step 2.5: Apply ALS Baseline Correction.
# Why: This function from the 'baseline' package fits a flexible baseline to each spectrum and subtracts it.
# This removes background noise and drift, making peak heights more accurate. `lambda` and `p` are
# tuning parameters that control the stiffness and asymmetry of the fitted baseline; these are common default values.
als_corrected <- baseline::baseline(spectra_matrix_imputed, method = 'als', lambda = 6, p = 0.05)
# The `baseline` function returns a complex object containing multiple pieces of information (like the baseline itself).
# We must use the `getCorrected()` function to extract the final, cleaned spectra matrix from this object.
spectra_corrected <- getCorrected(als_corrected)

# Step 2.6: Apply final smoothing and normalization.
# Why (Savitzky-Golay): This filter smooths the data by fitting a polynomial to a
# small window of points, effectively reducing high-frequency instrumental noise
# without distorting the underlying peak shapes.
spectra_smoothed <- savitzkyGolay(spectra_corrected, p = 2, w = 11, m = 0)
# Why (Standard Normal Variate): SNV scales each individual spectrum. This removes
# multiplicative effects (e.g., differences in sample thickness) and puts all
# spectra onto a common scale, which is essential for fair comparison in the model.
spectra_processed <- standardNormalVariate(spectra_smoothed)
cat("Pre-processing complete.\n\n")

# Why: This is the key step to solve the "small N, large P" problem (35 samples, 3000+ features).
# PCA finds the major axes of variation in the data and creates new, uncorrelated variables
# called Principal Components (PCs). Instead of using all 3000+ noisy features, we can use a
# small number of these PCs (e.g., 10) that capture the most important information. This
# makes the classification task much more stable and less likely to overfit.

# Step 3.1: Perform Principal Component Analysis.
# Why (`center = TRUE`, `scale. = TRUE`): This is standard practice. It ensures that all wavenumbers
# are on a common scale before calculating the components, so that high-absorbance peaks
# do not artificially dominate the analysis over low-absorbance peaks.
pca_result <- prcomp(spectra_processed, center = TRUE, scale. = TRUE)

# Step 3.2: Extract the PC scores.
# Why: The scores are the values of each sample along the new PC axes. We will use these
# scores as the new predictor variables for our classification model. We select the
# first 10 PCs, which typically capture the vast majority of the relevant variance.
pca_scores <- as.data.frame(pca_result$x[, 1:10])

# Step 3.3: Combine the PC scores with the original location labels.
# Why: This creates the final, tidy data frame that will be fed into our model for training.
pca_data <- bind_cols(location = identifiers$location, pca_scores)

# --- 4. MODELING & VALIDATION (PCA + LDA with LOOCV) ---
# Why (LDA): Linear Discriminant Analysis is a classic and powerful classifier that works
# very well on a small number of predictive features. It's an excellent choice to use
# after reducing dimensionality with PCA.
# Why (LOOCV): Leave-One-Out Cross-Validation is the most robust validation method for
# very small datasets. It trains the model 35 times, each time leaving out one sample
# to test on. This gives a very reliable estimate of how the model will perform on new data.

# Step 4.1: Configure the training control for LOOCV.
# Why (`method = "LOOCV"`): This explicitly tells `caret` to use the LOOCV strategy.
# Why (`savePredictions = "final"`): This is a crucial argument that tells `caret` to store the
# prediction for each left-out sample, which we need to build our confusion matrix manually.
loocv_control <- trainControl(method = "LOOCV", savePredictions = "final")

# Step 4.2: Train the LDA model.
# Why (`set.seed(123)`): This ensures that any random processes within the function (if any)
# are reproducible, so the model results will be identical every time the script is run.
set.seed(123)
# Why (`train`): `caret`'s `train` function provides a unified interface for this. We specify the
# formula (`location ~ .` means predict location using all other variables), the data,
# the method ("lda"), and our training control settings.
lda_model <- train(location ~ ., data = pca_data, method = "lda", trControl = loocv_control)

# --- 5. PERFORMANCE EVALUATION ---
# Why: We need to quantify how well our model performed during the LOOCV process.

# Step 5.1: Print the overall model performance.
# Why: The `lda_model` object contains a summary of the cross-validation, including
# the overall accuracy and Kappa statistic, providing a quick overview of performance.
cat("--- Overall Model Performance (from LOOCV) ---\n")
print(lda_model)

# Step 5.2: Manually create the confusion matrix from LOOCV predictions.
# Why: because we used `savePredictions = "final"`, the predictions are stored in
# the `$pred` slot of the model object. We can use these saved predictions to build
# the confusion matrix using the `table()` function, which provides a detailed
# breakdown of correct and incorrect classifications.
cat("\n--- Final Confusion Matrix (from LOOCV) ---\n")
loocv_predictions <- lda_model$pred
# The 'pred' column has the predicted class, 'obs' has the observed (true) class
confusion_matrix <- table(Predicted = loocv_predictions$pred, Actual = loocv_predictions$obs)
print(confusion_matrix)

# --- 6. VISUALIZE RESULTS ---
# Plot the first two principal components to see how the groups are separated.
pca_plot_data <- bind_cols(identifiers, pca_scores)

ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = location, shape = location)) +
  geom_point(size = 4, alpha = 0.8) +
  labs(
    title = "PCA of Otolith Edge Spectra",
    x = paste0("Principal Component 1 (", round(summary(pca_result)$importance[2,1]*100, 1), "%)"),
    y = paste0("Principal Component 2 (", round(summary(pca_result)$importance[2,2]*100, 1), "%)"),
    color = "Location",
    shape = "Location"
  ) +
  theme_minimal() +
  stat_ellipse(level = 0.95)