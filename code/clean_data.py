from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import year, month, dayofmonth, weekofyear, isnan, when, count, col, avg
import pyspark.sql.functions as func


# Define Functions
def blank_as_null(x):
    return when(col(x) != "", col(x)).otherwise(None)


app_name = "clean_data"
mongo_uri = "mongodb://54.71.183.99/"
file_name = "dc_project.wa_1m_rand"
conf = SparkConf().setMaster("local").setAppName(app_name)
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

df_raw = sqlContext.read.format("com.mongodb.spark.sql.DefaultSource")\
                    .option("uri", mongo_uri+file_name)\
                    .load()

df_raw = df_raw.select([col(c).cast("string") for c in df_raw.columns])
df_raw.cache()

# Data pre-processing

# Split Violations and consider only the first one
split_col = func.split(df_raw['violation'], ',')
df_raw = df_raw.withColumn('violation1', split_col.getItem(0))
# df_raw = df_raw.withColumn('violation2', split_col.getItem(1))
df_raw = df_raw.withColumnRenamed('violation1', 'main_violation')

# Split Enforcements and considered on the first one
split_col = func.split(df_raw['enforcements'], ',')
df_raw = df_raw.withColumn('enforcement1', split_col.getItem(0))
df_raw = df_raw.withColumnRenamed('enforcement1', 'main_enforcement')

# Split stop_time to get an hour of stop
split_col = func.split(df_raw['stop_time'], ':')
df_raw = df_raw.withColumn('stop_hour', split_col.getItem(0))

# Extract ID
split_col = func.split(df_raw['id'], '-')
df_raw = df_raw.withColumn('id_new', split_col.getItem(2))

# Drop columns that have duplicates and/or irrelevant, have too many categories or mostly missing values
to_drop = ['_id', 'id', 'violation', 'violations', 'violation_raw', 'violation2', 'fine_grained_location',
           'county_name', 'driver_race_raw', 'driver_age_raw', 'location_raw', 'is_arrested', 'enforcements',
           'police_department', 'stop_time', 'search_type', 'search_type_raw', 'state']
df_raw1 = df_raw.select([c for c in df_raw.columns if c not in to_drop])
df_raw1.cache()


# Handle Missing Values
df_raw2 = df_raw1
for c in df_raw2.columns:
    df_raw2 = df_raw2.withColumn(c, blank_as_null(c))

# Drop rows where stop_outcome is Null, county_fips and driver_race is Null
df_raw3 = df_raw2.where(col("stop_outcome").isNotNull())\
                .where(col("county_fips").isNotNull())\
                .where(col("driver_race").isNotNull())\
                .where(col("search_type_raw").isNotNull())
df_raw3.cache()

# Replace stop_outcome
df_raw3 = df_raw3.withColumn("stop_out", when((col('stop_outcome') != "Arrest or Citation"),"Not_Arrested")\
                             .otherwise(col("stop_outcome")))
df_raw3 = df_raw3.drop("stop_outcome")
df_raw3 = df_raw3.withColumnRenamed("stop_out", "stop_outcome")

df_impute = df_raw3\
    .withColumn("driver_gender_imp", when(col('driver_gender').isNull(), 'M').otherwise(col('driver_gender')))
df_impute = df_impute.drop('driver_gender')
df_impute = df_impute.withColumnRenamed('driver_gender_imp', 'driver_gender')
df_impute.cache()

mean_age_male = df_impute.where(col('driver_gender') == 'M').agg(avg('driver_age')).first()[0]
mean_age_female = df_impute.where(col('driver_gender') == 'F').agg(avg('driver_age')).first()[0]
df_impute1 = df_impute\
    .withColumn("driver_age_male_imp",
                when((col('driver_gender') == 'M') & (col('driver_age').isNull() | isnan(col('driver_age'))),
                     mean_age_male).otherwise(col("driver_age")))
df_impute1= df_impute1\
    .withColumn("driver_age_imp",
                when((col('driver_gender') == 'F') & (col('driver_age').isNull() | isnan(col('driver_age'))),
                     mean_age_female).otherwise(col("driver_age_male_imp")))
df_impute1 = df_impute1.drop('driver_age_male_imp', 'driver_age')
df_impute2 = df_impute1.withColumnRenamed('driver_age_imp', 'driver_age')
df_impute2.cache()

df_feat1 = df_impute2.withColumn("gender_diff", when(col("driver_gender") == col("officer_gender"), 0).otherwise(1))
df_feat1 = df_feat1.withColumn("race_diff", when(col("driver_race") == col("officer_race"), 0).otherwise(1))
df_feat1 = df_feat1.withColumn("time_of_day", when(col("stop_hour") < '07', 1)
                               .when(col("stop_hour") < '13',2)
                               .when(col("stop_hour") < '19', 3)
                               .otherwise(4))
df_feat1.cache()

df_feat1.write.format("com.mongodb.spark.sql.DefaultSource").mode("append")\
                    .option("uri", mongo_uri+file_name+"_clean")\
                    .save()
