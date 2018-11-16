package com.sunliang.sparkmllibstudy.seven

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/16.
  */
object svm {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("svm")
    val sc = new SparkContext(sc)

    val data = MLUtils.loadLibSVMFile(sc,"")

    val splits = data.randomSplit(Array(0.6,0.4),seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val numIterations = 100
    val model = SVMWithSGD.train(training,numIterations)

    val predictionAndLabel = test.map { point =>
      val score = model.predict(point.features)
      (score,point.label)
    }

    val print_predict = predictionAndLabel.take(20)
    println("prediction" + "\t" + "label")
    for(i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    println("area under ROC = " + accuracy)

    val ModelPath = ""
    model.save(sc,ModelPath)
    val sameModel = SVMModel.load(sc,ModelPath)

  }
}
