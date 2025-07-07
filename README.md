# Project Overview: Classifying Fish Otoliths with FT-IR Spectroscopy
This project explores the use of Fourier-Transform Infrared (FT-IR) spectroscopy to classify the geographic origin of rockfish based on the chemical composition of their otoliths (ear bones). We compare two distinct machine learning workflows to address the common challenges associated with spectroscopic data, particularly the "small N, large P" problem (few samples, many features).

The Analytical Challenge
The core challenge of this dataset is its structure: we have a very small number of samples (35 fish) but a very large number of features for each sample (over 3,000 wavenumbers from the FT-IR spectrum). This high dimensionality makes it easy for models to "overfit"—that is, to learn the random noise in the training data rather than the true underlying chemical differences between the groups. Our goal is to find a method that can overcome this noise and build a robust, generalizable classification model.

The Shared Pre-processing Pipeline
Both analytical approaches begin with an identical, robust pre-processing pipeline. This is the most critical part of any spectroscopic analysis, as it ensures the data is clean and that the model can focus on meaningful chemical signals.

Data Wrangling: The raw data, which includes multiple replicates for each sample, is reshaped. The replicates are averaged to create a single, more stable spectrum for the outer edge of each otolith.

Transmittance to Absorbance Conversion: The data is converted from Transmittance (T) to Absorbance (A) using the formula A = -log10(T). This is essential because, according to the Beer-Lambert Law, absorbance is linearly proportional to chemical concentration, which is the relationship most models are designed to work with.

Asymmetrical Least Squares (ALS) Baseline Correction: A flexible baseline is fitted to each spectrum and subtracted. This removes background drift and noise caused by instrument effects or light scattering, making the heights and areas of chemical peaks more accurate and comparable between samples.

Smoothing and Normalization: A Savitzky-Golay filter is applied to reduce high-frequency instrumental noise. Finally, Standard Normal Variate (SNV) normalization is used to scale each spectrum, removing multiplicative effects (like differences in sample thickness) and putting all spectra on a common scale for fair comparison.

After pre-processing, the two workflows diverge.

# Approach 1: Peak-Sensitive Elastic-Net Logistic Regression
This approach uses a single, powerful machine learning model to perform classification and feature selection simultaneously.

How it Works: Elastic Net is a penalized regression method. It builds a logistic regression model but applies a penalty that shrinks the coefficients (i.e., the importance) of the thousands of wavenumbers. Its key feature is that it can shrink the coefficients of unimportant features to exactly zero, effectively removing them from the model. This makes the model "peak-sensitive" because it automatically identifies the most informative spectral peaks and ignores the noisy baseline.

Strengths: The primary advantage of this method is interpretability. The output directly tells you which specific wavenumbers the model found most useful for distinguishing between the Los Angeles and San Diego fish. This is invaluable for biomarker discovery and understanding the underlying chemical differences.

When to Use It: This method is best when the primary goal is feature selection and interpretation. If the main question is "Which specific chemical bonds or functional groups differ between these two locations?", Elastic Net provides a direct answer. However, it can sometimes struggle with very small sample sizes, as the penalty might be too aggressive and shrink all coefficients to zero.

# Approach 2: PCA + LDA (Principal Component Analysis + Linear Discriminant Analysis)
This is a classic, two-step chemometric approach that prioritizes predictive accuracy by first reducing the complexity of the data.

How it Works:

PCA: First, Principal Component Analysis is applied to the 3,000+ wavenumbers. PCA finds the major patterns of variation in the data and creates a small number of new, uncorrelated variables called "Principal Components" (PCs). Instead of thousands of features, we might now have only 10 PCs that capture most of the original information.

LDA: Second, a simpler and more traditional classifier, Linear Discriminant Analysis, is trained using only these 10 PCs as its input. LDA is very effective when the number of features is small.

Strengths: The main advantage of this method is robustness and predictive power, especially with small sample sizes. By drastically reducing the number of features before classification, it avoids the "curse of dimensionality" and is much less likely to overfit. It is often the most successful approach when classification accuracy is the top priority.

When to Use It: This method is best when the primary goal is achieving the highest possible classification accuracy. If the main question is "How accurately can we classify a new fish to its correct location?", the PCA + LDA workflow is often superior for small datasets. The trade-off is that the results are less interpretable; a Principal Component is a complex combination of all original wavenumbers, so you lose the direct link to specific chemical peaks.

# Final Thought...
For this specific project, given the very small sample size of 35 otoliths, the PCA + LDA approach is the recommended primary method. It is the most likely to yield a statistically robust and accurate classification model. The Elastic Net approach remains highly valuable as a complementary, exploratory tool to identify potentially significant wavenumbers that can be investigated further.
