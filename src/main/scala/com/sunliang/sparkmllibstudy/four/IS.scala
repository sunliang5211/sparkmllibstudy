package com.sunliang.sparkmllibstudy.four

import org.apache.spark.mllib.regression.IsotonicRegression
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/20.
  */
object IS {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("IS")
    val sc = new SparkContext(conf)
    val data = sc.textFile("D://work//testdata//000000_0//isotonic//isotonic2.txt")

    val parsedData = data.map { line =>
      val parts = line.toString().split(",").map(_.toDouble)
      (parts(0),parts(1),1.0)
    }

    val model = new IsotonicRegression().setIsotonic(true).run(parsedData)
    model.predictions.foreach(println)

    //val res = model.predict(5)
    //println(res)

  }
}
