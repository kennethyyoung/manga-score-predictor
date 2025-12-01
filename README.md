# Project: Uncovering the Drivers of Manga Score on MyAnimeList
**Author:** Kenneth Young   
**Date:** November 2025    
**Platform:** [MyAnimeList Dataset (Kaggle) ](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist)    
**BC Email:** kenneth.young@bellevuecollege.edu   
**Google Email:** youngyken@gmail.com

## 1. Introduction

### 1.1 Problem Statement

**Domain Context**: The global market for manga (Japanese comics) and similar formats like Manhwa (Korean) and Manhua (Chinese) is highly competitive. For publishers, content platforms, and creators, understanding the factors that contribute to a manga's commercial success and critical acclaim is crucial for making informed business and creative decisions. High user engagement, measured by metrics like community size (members) and user scores, often translates to higher sales and longevity for a series.

**Problem**: There is a need to identify the key attributes of a manga that correlate with high user ratings and popularity. Are certain genres consistently more popular? Does the publication format (e.g., Manga, Manhwa) or the number of chapters influence its reception? More importantly, can we build a model that predicts a new manga's potential score before it is published, using only the information available at its conception (like its genre, theme, and type)? Answering these questions can help stakeholders predict a new title's potential and optimize their content strategy.

### 1.2 Objective & Modeling Approach

**Objective**: The objective of this project is twofold:



1.   First, to conduct an in-depth Exploratory Data Analysis (EDA) to understand the current market dynamics, identifying which features (content, popularity, etc.) are most strongly correlated with high user scores.
2.   Second, to use these insights to build, evaluate, and interpret predictive regression models that can forecast a new manga's potential score.





**Modeling Strategy**: We will be building regression models to predict the continuous score variable.

*   **Explanatory Model (Linear Regression)**: We will
first build a model using all features, including post-publication metrics like favorites and members. This is an inferential model to understand the current market and quantify the statistical relationship between popularity and score.

*   **Predictive Models (Linear Regression & KNN Regression)**: We will then build two models using only pre-publication (T=0) features (e.g., genres, themes, type). This simulates a real-world scenario of forecasting a new title's success. We will compare a Linear Regression (a simple, interpretable model) against a K-Nearest Neighbors (KNN) Regressor (a non-parametric model) to find the best tool for the job.

We will then select and interpret the best predictive model to provide actionable insights for a publisher or creator.


### 1.3 Data Source

The data for this project is the **MyAnimeList Dataset**, sourced from Kaggle. It was compiled by user Andreu Vall Hernández and contains data scraped from the popular anime and manga tracking website, MyAnimeList.net.

We will access the data directly within this Colab notebook using the Kaggle API. The following code cells handles the setup, download, and loading of the dataset into a pandas DataFrame.

## 2. Data Cleaning & Preprocessing

### Data Dictionary

| Variable | Data Type | Description |
| :--- | :--- | :--- |
| **score** | Numerical (float) | **(Target)** The weighted average user score, on a scale of 1 to 10. |
| **members** | Numerical (int) | **(Tier 1)** The total number of users who have the manga in their personal list. A key measure of community size. |
| **favorites**| Numerical (int) | **(Tier 1)** The number of users who have designated the manga as a "favorite". A strong signal of user passion. |
| **type** | Categorical (str) | **(Tier 1)** The format of the publication (e.g., Manga, Manhwa, Novel). |
| **status** | Categorical (str) | **(Tier 1)** The current publication status (e.g., 'Finished', 'Publishing'). |
| **chapters** | Numerical (int) | **(Tier 1)** The total number of chapters. A proxy for the series' length and longevity. |
| **volumes** | Numerical (int) | **(Tier 1)** The total number of published volumes. Also a proxy for length. |
| **genres** | Categorical (str) | **(Tier 2)** A multi-label string of genres associated with the manga (e.g., 'Action, Adventure'). |
| **themes** | Categorical (str) | **(Tier 2)** A multi-label string of themes (e.g., 'Gore, Mythology'). Provides deeper content insight. |
| **demographics**| Categorical (str) | **(Tier 2)** The target audience demographic (e.g., 'Shounen', 'Seinen'). |
| **authors** | Categorical (str) | **(Tier 2)** A structured string containing author details. Captures the "brand name" effect of popular creators. |
| **serializations**| Categorical (str) | **(Tier 2)** The magazine or platform where the manga was published. Can be a signal of quality or style. |

### Feature Selection Rationale

For this analysis, a subset of the original 30 columns was strategically selected to focus on the most impactful predictors of a manga's `score`. The goal is to build a robust model without introducing data leakage or unnecessary complexity. The chosen variables are categorized into two tiers:

* **Tier 1 (Core Predictors):** These are variables with direct, high-value information that are relatively simple to process. They form the foundation of our analysis and initial model.
* **Tier 2 (High-Potential Features):** These variables contain rich, predictive information but require more complex processing (e.g., parsing, splitting, and "exploding" the data). They are essential for building a high-performance model.

Variables like `ranked`, `popularity`, and `scored_by` were intentionally **excluded** because they are direct results of the `score` itself. Including them would cause **data leakage**, creating a model that cheats by using the answer to predict the answer, making it useless for predicting the score of a new manga. Other variables like `synopsis` were excluded as they require advanced Natural Language Processing, which is beyond the scope of this initial project.

### 2.1 Data Quality Assessment

#### Detecting and Handling Missing Values

The goal of this section is to identify and resolve data quality issues within our selected features to ensure our subsequent analysis is accurate and reliable. Our cleaning process will follow a strategic, multi-step approach:


1.   **Standardize Missing Values**: First, we address "semantically missing" values—entries like `'[]'` that are not technically null but represent missing information. We will convert these into a standard `NaN` format.
2.   **Primary Filtering**: We will filter the dataset based on our core project objective, which requires a `score`. Any row without a score is unusable for our supervised model.
3.   **Intelligent Imputation**: To preserve as much data as possible, we will impute (fill in) the remaining high-volume missing values. We'll use the "Unknown" category for categorical features and the median for numerical features to avoid distortion from outliers.
4.   **Final Cleanup**: We will drop the small number of remaining rows with missing data to finalize our clean dataset.
    

    

    

**Step 1: Feature Subsetting and Initial Setup**

First, we create a new DataFrame, df_analysis, containing only the 12 features selected in our Data Dictionary. This focuses our efforts and makes the entire process more efficient.


**Step 4.1 - Primary Filter on Target Variable** (`score`)

Our goal is to predict the score, so any row where the score is missing is unusable for training our model. This is the first and most critical filter.

** Step 4.2 - High-Volume Imputation (Categorical & Numerical)**

Of the remaining rows, many are missing `themes`, `demographics`, `chapters`, and `volumes`. Dropping these would discard over half our data. Instead, we will impute them intelligently.
    

#### Outlier Detection and Handling

In this section, we investigate the presence of extreme values in our key numerical features. Outliers can disproportionately influence the results of our analysis and the performance of certain machine learning models.

**Detection**: We will use visual methods, specifically histograms and box plots to identify the presence and scale of outliers in our numerical columns (`members`, `favorites`, `chapters`, `volumes`).


**Handling**: Most extreme values in this dataset represent legitimate, highly successful manga and are not data errors. Therefore, instead of removing them (which would bias our analysis), we will apply a logarithmic transformation. This transformation compresses the range of the data, reducing the skew and mitigating the influence of these extreme values without deleting valuable information

**Visual Detection of Outliers**

First, let's plot histograms to visualize the distribution of our numerical features. A heavy skew to the right will indicate that a few entries have vastly larger values than the majority.

**Observations**

All four numerical features are extremely right-skewed. The interactive histograms show that the vast majority of data points are clustered at the low end, with a very long tail of high values.

**Handling Outliers with Log Transformation**

Since our investigation showed that the extreme values are legitimate data from hyper-popular manga, our goal is not to remove them but to reduce their disproportionate influence. The standard and most effective method for this is a logarithmic transformation.

We will apply a log(x+1) transformation, which compresses the range of the data, pulling the long tail of the distribution inward and making it more symmetrical. This helps many machine learning algorithms perform better and ensures our analysis isn't dominated by a few famous titles. We use np.log1p() which is equivalent to np.log(1+x) to gracefully handle any values that might be zero.
    

**Verifying the Transformation**

To confirm the effectiveness of our transformation, we will now create histograms and box plots of our new log-transformed columns. We expect to see distributions that are much closer to a symmetrical "bell curve" (normal distribution), indicating that the extreme skew has been successfully mitigated.


## 3. Exploratory Data Analysis (EDA)

With a clean and preprocessed dataset, we can now explore the relationships between our features and the target variable, `score`. Our goal is to identify which attributes are most correlated with high user ratings.

### 3.1 Correlation Analysis of Numerical Features

**Heatmap**

A correlation heatmap is a powerful tool for getting a quick overview of the linear relationships between our numerical variables. An interactive heatmap allows us to explore these relationships in more detail. The values range from -1 (perfect negative correlation) to +1 (perfect positive correlation), with 0 indicating no linear correlation.

We are most interested in the top row: which features have the strongest positive or negative correlation with `score`?


**Scatter Plot Observations**

This scatter plot clearly reveals a strong positive relationship between a manga's user `score` and the log of its `favorites`. As a manga's popularity and user passion increase (moving from left to right), its average score also tends to rise.

A key insight lies in the variance of the data. At lower levels of popularity (left side of the plot), the scores are widely distributed, ranging from poor to excellent. This indicates that an obscure manga can be of any quality. However, as popularity increases, the range of scores becomes much narrower and concentrated in the higher tiers (above 7.5).

This strongly suggests that while not all good manga become popular, it is exceptionally rare for a highly popular manga to be poorly rated. High user engagement, particularly in the form of "favorites," serves as a statement of strong statistical association.

### 3.2 Score Distribution by Categorical Features

Next, let's see how the score is distributed across our main categorical features: `type` and `status`. Box plots are perfect for this, as they show the median, spread, and outliers for each category's score distribution.


**Box Plot Observations**

**Content Type Shows Subtle Differences**:
*   While major formats like "Manga," "Manhwa," and "Manhua" have relatively similar score distributions, "Novels" stand out with the highest median score of all types. This indicates a potential user preference for the storytelling in the novel format. The "Manga" category displays the widest range and the most outliers, reflecting its vastness and the sheer variety of titles it encompasses.

**Publication Status**
*   **"On Hiatus" Manga Receive the Highest Scores**: The most significant finding is that manga currently "On Hiatus" have the highest median score (7.38). This is a strong signal. Often, a series only goes on hiatus if it is popular enough that the publisher and fanbase are willing to wait for the author, which implies a certain level of quality and success
*   **"Currently Publishing" Outperforms "Finished"**: Counterintuitively, manga that are "Currently Publishing" (median 7.07) are rated slightly higher than those that are "Finished" (median 6.89). One possible explaination for this is "hype bias", ongoing series may benefit from active community engagement and hype, leading to temporarily inflated scores from an active fanbase.

*   **"Discontinued" Status is Not a Strong Quality Signal**: Interestingly, "Discontinued" manga (median 6.94) are rated similarly to "Finished" ones, suggesting that discontinuation is not always a sign of poor quality alone and can be due to other external factors.

### 3.3 Identifying High-Scoring Genres and Themes

This is a key part of our analysis. To find which content attributes correlate with high scores, we will move beyond simple averages and ask a more powerful question: **"Does a given genre or theme tend to score better or worse than the overall average manga?"**

To answer this, we will:

*    Establish a Baseline: Calculate the global average score for all manga in our dataset.

*    Analyze Each Category: For every genre and theme, calculate its average score and then find the difference between its score and the baseline.

*    Visualize the Impact: Create diverging bar charts to clearly show which categories have a positive impact (score above average) and which have a negative impact (score below average).

This method provides a much more insightful view than just looking at raw scores. We will also filter for categories that appear on at least 50 titles to ensure our findings are statistically meaningful. Finally, we will remove the "Award Winning" meta-tag from the genre analysis to focus purely on content-based attributes.

**Observations on Genre and Theme**

By visualizing the score impact relative to the global average, we can identify clear tiers of content that resonate differently with the MyAnimeList user base.

*    **High-Impact Attributes** (**Positive Correlation**): The categories with the strongest positive impact are consistently associated with well-defined, niche genres or character-driven narratives, rather than broad, action-oriented tropes.

       * Genres like "**Gourmet**," "**Mystery**," and "**Suspense**" significantly outperform the average. This suggests that focused, tightly-plotted stories are highly rewarded by the user base.

       * Interestingly, the highest-rated Themes are "**Iyashikei**" (healing/slice-of-life), "**Childcare**," and "**Delinquents**." This seemingly diverse group points towards a preference for stories centered on character development and interpersonal relationships, whether they are wholesome or dramatic.

*    **Low-Impact Attributes** (**Negative Correlation**): Conversely, content-driven attributes associated with fan service (e.g., 'Ecchi,' 'Harem') show the most significant negative score impact.

       * Genres like "**Ecchi**" and "**Hentai**" show the most significant negative impact on scores.

       * Similarly, Themes like "**Magical Sex Shift**" and "**Harem**" are also strongly correlated with lower-than-average scores.

Conclusion: This analysis reveals a clear preference among the MyAnimeList user base for manga with focused narratives and strong character work. While trope-heavy and fan-service-oriented content may find an audience (potentially measured by `members` count), it does not tend to achieve the same level of critical acclaim (`score`) as more niche or character-driven stories.

### 3.4 Investigating the "Mega-Hit" Effect

Our analysis has shown that `members_log` and `favorites_log` are strong predictors. But some manga are "mega-hits" (like *Berserk* or *One Piece*) and may behave differently than "normal" manga.

Before we engineer an "Elite Tier" or `is_mega_hit` feature for our model, let's investigate two things:
1.  **Visualize the Impact:** If we define "mega-hits" as the top 5% by `members`, what is the actual effect on the `score` distribution?
2.  **Validate the Threshold:** Is 5% the right threshold? Or does the "elite" effect kick in at the top 10%? Or only the top 1%?

### 4.4 Model Refinement: p-value Feature Selection

Before we build our final models, we will first refine our "Predictive" feature set. Our `X_pred_train_processed` DataFrame has over 90 features, many of which are likely just noise and not statistically significant.

We will be using statsmodels.api.OLS to fit a model and inspect the p-values for each coefficient. We will then create a new, "refined" list of features that only includes those that are statistically significant (e.g., p-value < 0.05).


**Observations on Initial Linear Regression Model Performance**

This table provides a powerful and clear answer to our primary question.

*    **"Leaky" Features are Extremely Powerful**: The Explanatory LR (R²: 0.583) performs well. This tells us that if we know a manga's current popularity (members, favorites), we can explain about 58% of the variance in its score.

*    **The "Honest" Prediction is Much Harder**: The Predictive Linear Regression (R²: 0.320) has a significantly lower R-squared. It quantifies the challenge of "T=0" forecasting. It proves that a manga's pre-publication content (genres, themes) alone can explain about 32% of its future score, while the other 26% of the explanatory power (0.58 - 0.32) comes after publication from market popularity.

*    **Linear Regression Beats KNN**: For the "honest" T-0 predictive models, our Predictive LR (Full) (R²: 0.320) outperformed the Predictive KNN (R²: 0.297). This is an example of the "Curse of Dimensionality". With over 50 features, the distance-based KNN model struggled to find "near" neighbors, and the simpler, parametric Linear Regression model produced a more accurate result.

*   **The "Full" vs. "Refined" Models**:

    Predictive LR (Full) (90 features): R² = 0.320

    Predictive LR (Refined) (53 features): R² = 0.313

    The performance of the two models is almost identical. Our "Refined" model, which uses 37 fewer features, saw its R-squared drop by less than 1%.

### 4.6 Model Evaluation and Selection

We have now trained and evaluated four distinct models. Our final comparison table is:

| Model Name | Type | # Features | R-squared (R²) | Mean Absolute Error (MAE) |
| :--- | :--- | :--- | :--- | :--- |
| Explanatory LR | Inference (Leaky) | 93 | 0.582913 | 0.255828 |
| Predictive LR (Full) | Predictive Baseline | 90 | 0.320159 | 0.332924 |
| **Predictive LR (Refined)** | **Predictive (Refined)** | **53** | **0.312530** | **0.334963** |
| Predictive KNN (Refined) | Predictive (Refined) | 53 | 0.297468 | 0.336865 |

**Model Selection Justification:**

Based on this table, we select the **Predictive LR (Refined)** as our best model.

While the Explanatory LR has the highest R-squared, it is an inferential-only model that uses "leaky" (post-publication) data.

Among the true predictive models, the Predictive LR (Full) has a slightly higher R-squared. However, our Predictive LR (Refined) achieves 98% of that performance (0.313 vs 0.320) while using 37 fewer features. The refined model is simpler, less complex, and less prone to overfitting.

However, we might still prefer to use the full predictive model. The full set is not demonstrably overfit given the R² drop of less than 1%. Also, simply dropping features with p > 0.05 ignores the fact that a group of highly correlated features can collectively be predictive even if individually non-significant.

For now, we will use Predictive LR (Refined) as our forecasting tool of choice.

### 4.8 Model Interpretation

We have selected our Predictive LR (Refined) as our best forecasting tool. But our Explanatory LR is a powerful inferential tool.

**Observations for the Predictive Model**:

This model (Model 3) was trained only on the 53 significant T-0 features. Its coefficients tell us what a publisher should look for to forecast a new manga's success.

*    Format is King: With the leaky popularity features gone, type_novel (coeff: +0.94) and type_light_novel (coeff: +0.50) are by far the most dominant positive predictors. The model is saying that if you want to predict a high score, the surest bet is that it's a novel.

*    Content is Critical: Wholesome and character-driven themes like themes_Childcare (+0.39), themes_Iyashikei (+0.38), and themes_Visual Arts (+0.37) are the next strongest signals.

*    Consistent Negatives: Just as in our Explanatory model, genres_Ecchi (-0.38) and genres_Horror (-0.19) are strong negative predictors. This confirms they are inherently correlated with low scores, not just because of low popularity.

*    status_finished is Negative: The model has learned that, holding all else constant, a "Finished" manga is associated with a -0.22 drop in score compared to the baseline ("Currently Publishing"). This confirms our EDA finding that ongoing hype and survivorship bias give a score boost to publishing manga.

### 4.9 Predictions on New Data

To demonstrate the "real-world" business use of our model, we will use our best model—the **Predictive LR (Refined)** (Model 3)—to run a controlled experiment.

We will create a "baseline" manga and then test the impact of adding "good" vs. "bad" content features to it, holding all other variables (like length) constant.


##5. Limitations & Future Work

While our analysis successfully produced a working, interpretable predictive model, its primary value comes from understanding its limitations. These limitations define the boundaries of our conclusions and provide a clear roadmap for future, more advanced analysis.

**The Business Problem: "Quality" vs. "Popularity"**

Our model was built to predict score (critical acclaim), but for a publisher, the true business goal might be predicting members (commercial reach) or favorites (engagement).

*    Wrong Target? We've proven that score (a proxy for "quality") and favorites (a proxy for "passion") are linked, but a publisher might prefer a "low-quality" (e.g., 6.5 score) manga that sells 10 million copies over a "high-quality" (e.g., 8.5 score) niche hit. Our model does not optimize for this business goal.

*    Regression vs. Classification: As we explored in our EDA, the most valuable business question might not be, "What will this manga's score be?" but rather, "Does this manga have the potential to be a 'mega-hit'?" A more advanced project would treat this as a classification problem, building a model to predict a binary is_mega_hit = True/False.

**The "Honest R-squared": Quantifying the Unpredictable**

Our best predictive model (the "Refined LR") achieved an R-squared of ~0.31.

It proves that the T-0 (pre-publication) features we can actually use for forecasting—genres, themes, length, type, and status—can only explain about 31% of a manga's future score. This strongly implies that the remaining ~69% of a manga's success is driven by factors that cannot be known on Day 0, such as:

*    Market forces and "hype"

*    Post-publication word-of-mouth

*    Publisher marketing budget

*    Sheer, unquantifiable luck

Our "leaky" Explanatory Model (R²: 0.58) proved this by showing that over 27% of the score is tied directly to these post-publication popularity metrics.

**The Data's Limitations: Proxies and High-Cardinality**
*   **Proxy Metrics**: Our entire analysis is based on MyAnimeList data. The "score" is a proxy for quality as perceived by the MAL user base. This data has no connection to the real-world business metrics that a publisher truly values: official sales figures, merchandise revenue, or ad revenue. A manga could have a low MAL score but be a massive commercial success, and our model would be blind to this.

*   **High-Cardinality Features**: To build a simple, interpretable model, we made a decision to exclude high-cardinality features like authors and serializations. This is a significant limitation. The T-0 feature "written by Hajime Isayama" (author of Attack on Titan) is an enormous predictor of a new manga's success, but our simple get_dummies approach would create 22,000+ columns.

**Future Work: A Roadmap**

These limitations provide a path forward for a more advanced model:

1.    **Refine the Target**: Re-build the project as a classification model to predict is_mega_hit = 1/0.

2.    **Incorporate Advanced Features**: Use more advanced techniques (beyond the scope of this project, such as Target Encoding or Feature Hashing) to incorporate the powerful authors and serializations features without creating 20,000+ columns.

3.    **Acquire Better Data**: The ideal of this project would be to merge this MAL dataset with real-world sales data to build a model that predicts a true financial outcome, not just a proxy score.

##6. Final Conclusions

This project began with a broad objective: to understand the drivers of manga success on MyAnimeList. Through a rigorous, multi-stage process, we successfully navigated a complex dataset to produce not only a comprehensive exploratory analysis but also a functional, interpretable predictive model.

**Summary of the Process**

Our journey involved two distinct phases:

1. **Exploratory Data Analysis** (EDA): We first cleaned and preprocessed over 64,000 manga entries. This involved handling missing data (including semantic nulls like '[]') and outliers (using log transformations). Our EDA revealed that while post-publication "leaky" features like favorites were the strongest predictors (R² ~0.58), key T-0 (pre-publication) content features like "Gourmet" and "Iyashikei" also showed a strong positive correlation with high scores.

2. **Predictive Modeling**: We then set out to build a "real-world" forecasting tool that used only T-0 features.

    *    We compared a Linear Regression model against a K-Nearest Neighbors (KNN) model and found that Linear Regression was the superior choice, as KNN's performance was degraded by the high-dimensional nature of our 90+ features (a classic "Curse of Dimensionality").

    *   Using the p-value refinement, we refined our model from 90 features down to 53 statistically significant features.

    *    This Predictive LR (Refined) model was selected as our final, best model, as it provided the ideal balance of performance (R²: 0.313) and interpretability.

**Key Insights & Business Value**

Our final, refined model provided clear, actionable insights for a publisher or creator:

*   **Content is a Quantifiable Predictor**: We can forecast a manga's score with ~31% accuracy based only on its pre-publication attributes.

*   **Positive Drivers**: The model dictates that Format is the single most actionable T-0 predictor, with `type_novel` carrying a coefficient of +0.94, suggesting a novel format is the surest bet for critical acclaim. Content-wise, wholesome, character-driven themes like `themes_Childcare` (+0.39) and `themes_Iyashikei` (+0.38) are highly valued.

*   **Negative Drivers**: The model confirmed that fan-service genres like `genres_Ecchi` (-0.38) are the single strongest negative predictor of a manga's critical score.

We successfully demonstrated this model's business value in our final step. When we tested two hypothetical manga, our model correctly forecasted a high score of 7.66 for a "Gourmet/Iyashikei" title and a low score of 6.86 for an "Ecchi/Fantasy" title. This proves the model's utility as a tool to help publishers "greenlight" projects and invest in content that is statistically proven to align with user preference.
