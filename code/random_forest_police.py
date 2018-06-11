from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import year, month, dayofmonth, weekofyear, isnan, when, count, col, avg
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

app_name = "logistic_police_test"
mongo_uri = "mongodb://54.71.183.99/"
file_name = "dc_project.wa_1m_rand_clean"
conf = SparkConf().setMaster("local").setAppName(app_name)
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# create encoder
def indexStringColumns(df, cols):
    newdf = df
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c + "-num")
        sm = si.fit(newdf)
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c + "-num", c)
    return newdf


def oneHotEncodeColumns(dframe, cols):
    newdf = dframe
    for c in cols:
        onehotenc = OneHotEncoder(inputCol=c, outputCol=c + "-onehot", dropLast=False)
        newdf = onehotenc.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c + "-onehot", c)
    return newdf


df_feat1 = sqlContext.read.format("com.mongodb.spark.sql.DefaultSource") \
    .option("uri", mongo_uri + file_name) \
    .load()

# Drop columns resulting in data leakage (Redundant Fields in second line)
df_clean1 = df_feat1.drop("contraband_found", "main_violation", "main_enforcement") \
    .drop('driver_age_raw', 'driver_race_raw', 'violation_raw', 'search_type_raw', 'stop_time') \
    .cache()

df_clean = df_clean1 \
    .withColumn('stop_date_year', year(df_clean1.stop_date.cast(DateType()))) \
    .withColumn('stop_date_month', month(df_clean1.stop_date.cast(DateType()))) \
    .withColumn('stop_date_dayofmonth', dayofmonth(df_clean1.stop_date.cast(DateType()))) \
    .withColumn('stop_date_weekofyear', weekofyear(df_clean1.stop_date.cast(DateType()))) \
    .withColumn('county_fips_clean', df_clean1.county_fips.cast(IntegerType())) \
    .withColumn('driver_age_clean', df_clean1.driver_age.cast(IntegerType())) \
    .withColumn('officer_id_clean', df_clean1.officer_id.cast(IntegerType())) \
    .withColumn('road_number_clean', df_clean1.road_number.cast(IntegerType())) \
    .withColumn('milepost_clean', df_clean1.milepost.cast(IntegerType())) \
    .withColumn('lat_clean', df_clean1.lat.cast(IntegerType())) \
    .withColumn('lon_clean', df_clean1.lon.cast(IntegerType())) \
    .withColumn('stop_hour', df_clean1.stop_hour.cast(IntegerType())) \
    .withColumn('id_new', df_clean1.id_new.cast(IntegerType())) \
    .drop('stop_date') \
    .drop('county_fips') \
    .drop('driver_age') \
    .drop('officer_id') \
    .drop('road_number') \
    .drop('milepost') \
    .drop('lat') \
    .drop('lon') \
    .withColumnRenamed("county_fips_clean", "county_fips") \
    .withColumnRenamed("driver_age_clean", "driver_age") \
    .withColumnRenamed("officer_id_clean", "officer_id") \
    .withColumnRenamed("road_number_clean", "road_number") \
    .withColumnRenamed("milepost_clean", "milepost") \
    .withColumnRenamed("lat_clean", "lat") \
    .withColumnRenamed("lon_clean", "lon") \
    .withColumnRenamed('id_new', 'id')


df_clean.cache()
df = df_clean.na.drop()

# numerically encode categorical variables
df_string = indexStringColumns(df,
                               ["contact_type", "driver_gender", "driver_race", "drugs_related_stop", "highway_type",
                                "officer_gender", "officer_race", "search_conducted", "stop_outcome"])

df_hot = oneHotEncodeColumns(df_string, ['contact_type', 'driver_race', 'highway_type', 'officer_race', 'driver_gender',
                                  'officer_gender'])

input_cols = ['stop_hour', 'id', 'drugs_related_stop', 'search_conducted', 'stop_date_year', 'stop_date_month',
              'stop_date_dayofmonth', 'stop_date_weekofyear', 'county_fips', 'driver_age', 'officer_id', 'road_number',
              'milepost', 'lat', 'lon', 'contact_type', 'driver_race', 'highway_type', 'driver_gender',
              'officer_gender', 'officer_race', 'gender_diff', 'race_diff', 'time_of_day']

va = VectorAssembler(outputCol="features", inputCols=input_cols)
df_assembled = va.transform(df_hot).select("features", "stop_outcome").withColumnRenamed("stop_outcome", "label")

splits = df_assembled.randomSplit([0.8, 0.2])
df_train = splits[0].cache()
df_test = splits[1].cache()

# Train
rf = RandomForestClassifier(maxDepth=30, minInstancesPerNode=5)
rf_model = rf.fit(df_train)

# Evaluate
rf_predicts = rf_model.transform(df_test)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(rf_predicts)
print("Test Error = %g" % (1.0 - accuracy))

bceval = BinaryClassificationEvaluator(metricName="areaUnderPR")
print bceval.evaluate(rf_predicts)

rf_model.save("rf_rand")
