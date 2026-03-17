import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;

import org.apache.spark.ml.evaluation.RegressionEvaluator;

public class MovieRecommender {
    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder()
                .appName("Movie Recommendation ALS - Java")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> ratings = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("ratings.csv");

        ratings = ratings.select("userId", "movieId", "rating");

        Dataset<Row>[] splits = ratings.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        ALS als = new ALS()
                .setUserCol("userId")
                .setItemCol("movieId")
                .setRatingCol("rating")
                .setColdStartStrategy("drop");

        ALSModel model = als.fit(train);

        Dataset<Row> predictions = model.transform(test);

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rating")
                .setPredictionCol("prediction");

        double rmse = evaluator.evaluate(predictions);

        System.out.println("RMSE = " + rmse);

        Dataset<Row> userRecs = model.recommendForAllUsers(5);
        userRecs.show(false);

        spark.stop();
    }
}
