Here is a detailed inventory of every figure-style graphic in that sample report (from OCR of the PDF; captions and body text). I also note two tables that act as quantitative summaries (not graphs).

Fig. 1 — Bivariate choropleth map (England)
Type: Geographic choropleth, specifically a bivariate map (two variables encoded together on the same map—here health deprivation and loneliness index tertiles; the abstract/body also frames this as “geographical visualisations”).
Why they used it: To show where deprivation and loneliness co-occur or diverge spatially at once, instead of two separate maps. That supports a narrative about regional patterns and mismatch (e.g. North–South patterns they discuss later). Bivariate maps are appropriate when the research question is joint geographic structure of two indices.
Fig. 2 — Side‑by‑side boxplots
Type: Box-and-whisker plots (one distribution for loneliness score, one for IMD).
Why they used it: Classic ED A for continuous variables: median, spread (IQR), skew, and especially outliers (they explicitly point to points beyond the whiskers). Boxplots are the standard choice when you need robust comparison of distributions and outlier visibility before cleaning rules (IQR removal).
Fig. 3 — Three choropleth maps (panel a–c)
Type: Choropleth maps of loneliness scores at three spatial scales: (a) LSOA, (b) MSOA, (c) Local Authority.
Why they used it: To show how aggregation level changes what you see: sparsity and extreme values at small geographies vs full coverage and compressed range at LA level. Maps are the right tool when the claim is about geographic coverage and scale effects, not just a single number.
Fig. 4 — Correlation display for loneliness vs IMD (and related structure)
Type: In the caption this is “Correlation between loneliness score and IMD data.” From the surrounding text, it functions as a correlation matrix / heatmap-style summary of Pearson correlations (they define PMCC and interpret blocks of variables—e.g. housing/living/outdoor/indoor correlating with each other and with loneliness).
Why they used it: To summarize many pairwise linear relationships in one view—appropriate when you have many deprivation sub-indices and need a compact pattern summary rather than dozens of scatter plots.
Fig. 5 — Scatter plot
Type: Scatter plot: health deprivation score (x-axis) vs loneliness index (y-axis) (caption).
Why they used it: The direct visualization of a bivariate linear association (they discuss weak / slight negative correlation). Scatter plots are the right choice to see curvature, heteroscedasticity, and outliers that a single correlation coefficient can hide.
Fig. 6 — 2D decile frequency plot (heatmap-style)
Type: A two-way frequency table visualization with Health Deprivation Decile vs Loneliness Decile (caption references red hues and a diagonal where correlation would concentrate). Practically: a heatmap (or tiled heatmap) of counts/frequencies of LSOAs in each decile pair.
Why they used it: To test whether high loneliness aligns with high deprivation across deciles jointly—a non-parametric view that can reveal hotspots off the diagonal. This complements Fig. 5 by showing co-movement across the full decile grid, not just a linear fit line.
Fig. 7 — Histogram
Type: Histogram of the loneliness score distribution.
Why they used it: To assess shape (they argue it looks Gaussian after z-score construction) and to motivate thresholding / outlier strategies. Histograms are appropriate for univariate distribution shape and bin-level frequency inspection.
Fig. 8 — Heatmap of z‑scores (top LSOAs × conditions)
Type: Heatmap (matrix): rows correspond to top ~20 high-loneliness LSOAs, columns to illness/prescription z-score components (depression, Alzheimer’s, blood pressure, etc.), with red/green indicating magnitude (caption).
Why they used it: To compare multivariate prescription profiles across areas—useful for spotting specialist GP skew (they give Middlesbrough / addiction example). Heatmaps excel when you need many variables at once for case comparisons.
Fig. 9 — Feature importance chart (random forest)
Type: Horizontal bar chart of feature importances (typical presentation; OCR lists feature names along an importance axis).
Why they used it: To interpret the random forest in terms of which IMD inputs drive predictions at LA level. Bar charts of importances are standard because importances are scalar rankings with unclear “units”—bars make relative importance immediately comparable.
Fig. 10 — Correlation heatmap among loneliness index components
Type: Heatmap of correlations between prescription-based variables contributing to the loneliness index (and related quantities).
Why they used it: To argue the inputs are not independent—important because summing z-scores assumes additivity; this figure supports a methodological critique (non-independence). A correlation heatmap is the right tool for dependency structure in many correlated features.
Fig. 11 — Clustering scatter plot (2D embedding)
Type: Scatter plot in 2D space after dimensionality reduction (e.g., PCA-like projection), with points = GP practices, clusters (they report 4 clusters).
Why they used it: To visualize clustering and separation of groups in a low-dimensional view humans can read. Scatter plots are standard for clustering results when you reduce to 2D; color/cluster overlays show structure beyond the raw high-dimensional prescription vectors.
Non‑graph visuals (tables)
Table I: Decile differences between cleaned data and outliers.

Why: Precise quantitative summary of how outliers differ across deprivation domains—tables beat charts when readers need exact numbers across many categories.
Table II: Model performance (baseline vs random forest MAE, by setting).

Why: Standard for ML evaluation—compact comparison of error metrics across models and settings.
Quick “why this chart type” pattern in the report
Maps → geography and spatial aggregation choices.
Boxplots / histogram → univariate EDA and outliers.
Scatter → bivariate linear relationship.
Correlation matrix / heatmaps → many pairwise relationships or feature dependence.
Decile heatmap → joint categorical distribution (two rankings).
Bar importances → model interpretability in one glance.
2D scatter after DR → clustering structure.