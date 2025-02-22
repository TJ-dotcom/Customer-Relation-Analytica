from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
from pyspark.ml.clustering import BisectingKMeans, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

# Sample the data
sample_size = 10000
df_sample = df.sample(withReplacement=False, fraction=sample_size/df.count(), seed=42)

# Prepare features
feature_cols = ['Recency', 'Frequency', 'Monetary']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vector = assembler.transform(df_sample)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

# 1. Hierarchical Clustering (Using BisectingKMeans as alternative since MLlib doesn't have hierarchical)
bkm = BisectingKMeans(k=4, featuresCol="scaled_features")
model_bkm = bkm.fit(df_scaled)
df_bkm = model_bkm.transform(df_scaled)

evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="scaled_features")
silhouette_bkm = evaluator.evaluate(df_bkm)
print(f"Bisecting KMeans Silhouette Score: {silhouette_bkm:.4f}")

# 2. Agglomerative (Using BisectingKMeans as alternative)
# Already covered above

# 3. BIRCH (Using KMeans as alternative since MLlib doesn't have BIRCH)
kmeans_birch = KMeans(k=4, featuresCol="scaled_features")
model_birch = kmeans_birch.fit(df_scaled)
df_birch = model_birch.transform(df_scaled)
silhouette_birch = evaluator.evaluate(df_birch)
print(f"KMeans (BIRCH alternative) Silhouette Score: {silhouette_birch:.4f}")

# 4. K-Means
kmeans = KMeans(k=4, featuresCol="scaled_features")
model_kmeans = kmeans.fit(df_scaled)
df_kmeans = model_kmeans.transform(df_scaled)
silhouette_kmeans = evaluator.evaluate(df_kmeans)
print(f"KMeans Silhouette Score: {silhouette_kmeans:.4f}")

# 5. Linear Regression
# Prepare data for regression
assembler_lr = VectorAssembler(inputCols=['Recency', 'Frequency'], outputCol="features")
df_lr = assembler_lr.transform(df_sample)

lr = LinearRegression(featuresCol="features", labelCol="Monetary")
lr_model = lr.fit(df_lr)
predictions_lr = lr_model.transform(df_lr)

evaluator_lr = RegressionEvaluator(labelCol="Monetary", predictionCol="prediction")
r2 = evaluator_lr.evaluate(predictions_lr, {evaluator_lr.metricName: "r2"})
mse = evaluator_lr.evaluate(predictions_lr, {evaluator_lr.metricName: "mse"})
print(f"Linear Regression R2: {r2:.4f}")
print(f"Linear Regression MSE: {mse:.4f}")

# 6. Logistic Regression
# First convert RFM_Score to numeric type
df_sample = df_sample.withColumn("RFM_Score_numeric", col("RFM_Score").cast(DoubleType()))

feature_cols_log = ['Recency', 'Frequency', 'Monetary', 'Total_Price', 'Quantity']
assembler_log = VectorAssembler(inputCols=feature_cols_log, outputCol="features")
df_log = assembler_log.transform(df_sample)

# Use the new numeric column
lr_classifier = LogisticRegression(featuresCol="features", labelCol="RFM_Score_numeric")
lr_model = lr_classifier.fit(df_log)
predictions_lr = lr_model.transform(df_log)

# Update evaluator to use the numeric column
evaluator_lr = MulticlassClassificationEvaluator(labelCol="RFM_Score_numeric", predictionCol="prediction")
accuracy_lr = evaluator_lr.evaluate(predictions_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")

# Encode the RFM_Score_numeric to fit within the required range
indexer = StringIndexer(inputCol="RFM_Score_numeric", outputCol="label")
df_log = indexer.fit(df_log).transform(df_log)

# 7. Random Forest
# Prepare data for Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)

# Train the Random Forest model
rf_model = rf.fit(df_log)
predictions_rf = rf_model.transform(df_log)

# Evaluate the Random Forest model
evaluator_rf = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_rf = evaluator_rf.evaluate(predictions_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# 8. Gradient Boosted Trees
# Prepare data for Gradient Boosted Trees
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=100)

# Train the Gradient Boosted Trees model
gbt_model = gbt.fit(df_log)
predictions_gbt = gbt_model.transform(df_log)

# Evaluate the Gradient Boosted Trees model
evaluator_gbt = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_gbt = evaluator_gbt.evaluate(predictions_gbt)
print(f"Gradient Boosted Trees Accuracy: {accuracy_gbt:.4f}")

