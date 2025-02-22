import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

spark.stop()

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Retail Data Cleaning") \
    .getOrCreate()

# Read the CSV file
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load('Online Retail.csv')

# DISTRIBUTED OPERATION 1: Remove duplicates using distributed distinct operation
df = df.distinct()

# DISTRIBUTED OPERATION 2: Clean and transform data using distributed map operations
df = df.dropna() \
    .withColumn("CustomerID", col("CustomerID").cast("integer")) \
    .withColumn("InvoiceDate", to_timestamp("InvoiceDate")) \
    .withColumn("Total_Price", col("Quantity") * col("UnitPrice")) \
    .withColumn("Year", year("InvoiceDate")) \
    .withColumn("Month", month("InvoiceDate")) \
    .withColumn("DayOfWeek", dayofweek("InvoiceDate"))

# DISTRIBUTED OPERATION 3: Season calculation using distributed UDF
@udf(returnType=StringType())
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

df = df.withColumn("Season", get_season(col("Month")))

# DISTRIBUTED OPERATION 4: Filter invalid data using distributed filtering
df = df.filter((col("Quantity") > 0) & (col("UnitPrice") > 0))

# DISTRIBUTED OPERATION 5: Text cleaning using distributed UDF
@udf(returnType=StringType())
def clean_text(text):
    if text is None: return None
    # Remove special characters and extra spaces
    cleaned = re.sub(r'[^\w\s]', '', str(text)).strip()
    return cleaned

string_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
for column in string_columns:
    df = df.withColumn(column, clean_text(col(column)))

# DISTRIBUTED OPERATION 6: RFM Analysis using window functions
window_spec = Window.orderBy("InvoiceDate")
max_date = df.agg(max("InvoiceDate")).collect()[0][0]

rfm_df = df.groupBy("CustomerID").agg(
    datediff(lit(max_date), max("InvoiceDate")).alias("Recency"),
    countDistinct("InvoiceNo").alias("Frequency"),
    sum("Total_Price").alias("Monetary")
)

# Calculate quartiles using window functions
window_quartile = Window.orderBy("Recency")
rfm_df = rfm_df.withColumn("R", ntile(4).over(window_quartile))
window_quartile = Window.orderBy("Frequency")
rfm_df = rfm_df.withColumn("F", ntile(4).over(window_quartile))
window_quartile = Window.orderBy("Monetary")
rfm_df = rfm_df.withColumn("M", ntile(4).over(window_quartile))

# Create RFM Score
rfm_df = rfm_df.withColumn("RFM_Score", 
    concat(col("R").cast("string"), 
          col("F").cast("string"), 
          col("M").cast("string")))

# Join RFM metrics back to main dataframe
df = df.join(rfm_df, "CustomerID", "left")

# Cache the DataFrame for better performance in subsequent operations
df.cache()

# Show the results
df.select("CustomerID", "Recency", "Frequency", "Monetary", "RFM_Score").show(5)