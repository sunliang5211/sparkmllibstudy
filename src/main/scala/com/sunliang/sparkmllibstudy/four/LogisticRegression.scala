package com.sunliang.sparkmllibstudy.four

import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/15.
  */
object LogisticRegression {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("LogisticRegression")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,"D://work//testdata//000000_0//logistic//logistic2.txt")

    val splits = data.randomSplit(Array(0.8,0.2),seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(training)

    val predictionAndLabels = test.map{
      case LabeledPoint(label,features) =>
        val prediction = model.predict(features)
        (prediction,label)
    }

    val print_predict = predictionAndLabels.take(200)
    println("prediction" + "\t" + "label")
    for(i <- 0 to print_predict.length - 1 ) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Precision = " + precision)

/*    val ModelPath = ""
    model.save(sc,ModelPath)
    val sameModel = LogisticRegressionModel.load(sc,ModelPath)*/

  }

}
