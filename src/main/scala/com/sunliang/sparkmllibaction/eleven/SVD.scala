package com.sunliang.sparkmllibaction.eleven

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/22.
  */
object SVD {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("SVD")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D://work//testdata//000000_0//svd//svd1.txt").map(_.split(" ").map(_.toDouble)).map(line => Vectors.dense(line))
    val rm = new RowMatrix(data)
    val svd = rm.computeSVD(2,computeU = true)
    println(svd)
    println(svd.U)
    println(svd.s)
    println(svd.V)
  }

}
