package com.sunliang.sparkmllibstudy.ten

import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/16.
  */
object lda {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("lda")
    val sc = new SparkContext(sc)

    val data = sc.textFile("")
    val parseData = data.map(s => Vectors.dense(s.trim.split(" ").map(_.toDouble)))
    val corpus = parseData.zipWithIndex.map(_.swap).cache()

    val ldaModel = new LDA().setK(3).setDocConcentration(5).setTopicConcentration(5).setMaxIterations(20).setSeed(0L).setCheckpointInterval(10).setOptimizer("em").run(corpus)
    println("Learned topics (as distributions voer vocab of " + ldaModel.vocabSize + "words):")

    val topics = ldaModel.topicsMatrix
    for(topic <- Range(0,3)) {
      print("Topic " + topic + ":")
      for(word <- Range(0,ldaModel.vocabSize)) {
        print(" " + topics(word,topic))
      }
      println()
    }

    ldaModel.describeTopics(4)

    val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
    distLDAModel.topicDistributions.collect
  }
}
