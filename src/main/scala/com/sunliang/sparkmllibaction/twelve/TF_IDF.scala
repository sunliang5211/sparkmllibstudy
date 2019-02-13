package com.sunliang.sparkmllibaction.twelve

import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/22.
  */
object TF_IDF {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("TF_IDF")
    val sc = new SparkContext(conf)

    val documents = sc.textFile("D://work//testdata//000000_0//TF-IDF//tf-idf1.txt").map(_.split(" ").toSeq)

    val hashingTF = new HashingTF()
    val tf = hashingTF.transform(documents)
    val idf = new IDF().fit(tf)

    val tf_idf = idf.transform(tf)
    tf_idf.foreach(println)

  }
}
