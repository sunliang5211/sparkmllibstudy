package com.sunliang.sparkmllibstudy.four

import org.apache.spark.mllib.regression.IsotonicRegressionModel
import org.apache.spark.mllib.regression.IsotonicRegression
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/15.
  */
object IsotonicRegression {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("IsotonicRegression")
    val sc = new SparkContext(conf)

    val data = sc.textFile("")
    val parseData = data.map { line =>
      val parts = line.split(",").map(_.toDouble)
      (parts(0),parts(1),1.0)
    }

    val splits = parseData.randomSplit(Array(0.6,0.4),seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = new IsotonicRegression().setIsotonic(true).run(training)
    val x = model.boundaries
    val y = model.predictions
    println("boundaries" + "\t" + "predictions")
    for(i <- 0 to x.length - 1){
      println(x(i) + "\t" + y(i))
    }

    val predictionAndLabel = test.map { point =>
      val predictedblabel = model.predict(point._2)
      (predictedblabel,point._1)
    }

    val print_predict = predictionAndLabel.take(20)
    println("prediction" + "\t" + "label")
    for(i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    val meanSquareError = predictionAndLabel.map{
      case(p,1) => math.pow((p-1),2 )
    }.mean()
    println("Mean Squared Error = " + meanSquareError)

    val modelPath = ""
    model.save(sc,modelPath)
    val sameModel = IsotonicRegressionModel.load(sc,modelPath)

  }

}
