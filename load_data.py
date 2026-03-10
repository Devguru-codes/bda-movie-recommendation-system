from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("ALS Movie Recommendation") \
    .getOrCreate()

# Load ratings dataset
ratings = spark.read.csv("ratings.csv", header=True, inferSchema=True)

# Load movies dataset
movies = spark.read.csv("movies.csv", header=True, inferSchema=True)

# Show schema
ratings.printSchema()
movies.printSchema()

# Show sample data
ratings.show(5)
movies.show(5)

# Count records
print("Total ratings:", ratings.count())
print("Total movies:", movies.count())
