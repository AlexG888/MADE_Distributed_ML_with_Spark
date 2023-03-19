package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.made.WithSpark._sqlc
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

trait WithSpark {
  lazy val spark: SparkSession = WithSpark._spark
  lazy val sqlc: SQLContext = WithSpark._sqlc

  lazy val schema: StructType = new StructType()
    .add("x1", DoubleType)
    .add("x2", DoubleType)
    .add("x3", DoubleType)
    .add("label", DoubleType)

  lazy val test_dataset_path: String = getClass.getResource("/test_data.csv").getPath

  lazy val df_raw: DataFrame = _sqlc.read
    .option("header", "true")
    .schema(schema)
    .csv(test_dataset_path)

  lazy val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("x1", "x2", "x3"))
    .setOutputCol("features")

  lazy val df: DataFrame = assembler
    .transform(df_raw)
    .drop("x1", "x2", "x3")
}

object WithSpark {
  lazy val _spark: SparkSession = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  lazy val _sqlc: SQLContext = _spark.sqlContext
}