package com.sunliang.sparkmllibstudy.ten

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/16.
  */
object KMeans {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("KMeans")
    val sc = new SparkContext(sc)

    val data = sc.textFile("")
    val parseData = data.map(s => Vectors.dense(s.split(" ").map(_.toDouble))).cache()

    val initMode = "k-means||"
    val numClusters = 2
    val numIterations = 20
    val model = new KMeans().setInitializationMode(initMode).setK(numClusters).setMaxIterations(numIterations).run(parseData)

    val WSSSE = model.computeCost(parseData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    val ModelPath = ""
    model.save(sc,ModelPath)
    val sameModel = KMeansModel.load(sc,ModelPath)
  }
}
