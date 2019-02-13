package com.sunliang.sparkmllibaction.twelve

import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/22.
  */
object word2Vec {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("TF_IDF")
    val sc = new SparkContext(conf)

    val documents = sc.textFile("D://work//testdata//000000_0//TF-IDF//tf-idf1.txt").map(_.split(" ").toSeq)
    val word2vec = new Word2Vec()
    val model = word2vec.fit(documents)
    println(model.getVectors)
    val synonyms = model.findSynonyms("spar",2)
    for(synonym <- synonyms) {
      println(synonym)
    }

  }
}
