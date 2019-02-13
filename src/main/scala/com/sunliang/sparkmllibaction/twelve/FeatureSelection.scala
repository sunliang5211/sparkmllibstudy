package com.sunliang.sparkmllibaction.twelve

import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/22.
  */
object FeatureSelection {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("TF_IDF")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,"D://work//testdata//000000_0//FeatureSelection//FeatureSelection1.txt")
    val discretizedData = data.map { lp =>
      LabeledPoint(lp.label,Vectors.dense(lp.features.toArray.map { x => x / 2}))
    }

    val selector = new ChiSqSelector(2)
    val transformer = selector.fit(discretizedData)
    val filteredData = discretizedData.map { lp =>
      LabeledPoint(lp.label,transformer.transform(lp.features))
    }
    filteredData.foreach(println)

  }
}
