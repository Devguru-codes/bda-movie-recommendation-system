from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode

# Create Spark session
spark = SparkSession.builder \
    .appName("Movie Recommendation ALS") \
    .getOrCreate()

# Load datasets
ratings = spark.read.csv("ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv("movies.csv", header=True, inferSchema=True)

ratings = ratings.select("userId", "movieId", "rating")

# Train test split
(train, test) = ratings.randomSplit([0.8, 0.2])

# ALS model
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    nonnegative=True,
    coldStartStrategy="drop"
)

model = als.fit(train)

# Predictions
predictions = model.transform(test)

# Evaluation
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)

print("RMSE =", rmse)

# Generate recommendations
user_recs = model.recommendForAllUsers(5)

# Convert nested recommendations
recs = user_recs.withColumn("rec", explode("recommendations")) \
                .select("userId", "rec.movieId", "rec.rating")

# Join with movie titles
final_recommendations = recs.join(movies, on="movieId")

# Show results
final_recommendations.select("userId","title","rating").show(20, False)
