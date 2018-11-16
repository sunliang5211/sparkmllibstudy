package com.sunliang.sparkmllibstudy.seven

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/16.
  */
object Naive_Bayes {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("NaiveBayes")
    val sc = new SparkContext(conf)

    val data = sc.textFile("")
    val parseData = data.map { line =>
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }

    val splits = parseData.randomSplit(Array(0.6,0.4),seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training,lambda = 1.0,modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features),p.label))
    val print_predict = predictionAndLabel.take(20)
    println("prediction" + "\t" + "label")
    for ( i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    val ModelPath = ""
    model.save(sc,ModelPath)
    val sameModel = NaiveBayesModel.load(sc,ModelPath)
  }

}
