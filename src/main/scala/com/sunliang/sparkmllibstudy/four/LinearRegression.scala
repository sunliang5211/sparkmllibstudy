package com.sunliang.sparkmllibstudy.four

import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/15.
  */
object LinearRegression {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local").setAppName("LinearRegression")
    val sc = new SparkContext(conf)
    //System.setProperty("hadoop.home.dir","D://ProgramFiles//winutils-master//hadoop_home_bin//bin")

    val data_path = "D://work//testdata//000000_0//linear//linear2.txt"
    //val data_path = args(0)
    val data = sc.textFile(data_path)

    // 读取样本数据
    val examples = data.map{ line =>
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }.cache()

    val numExamples = examples.count()

    //3 新建线性回归模型，并设置训练参数
    val numIterations = 1000
    val stepSize = 0.0000001105
    val miniBatchFraction = 1
    val model = LinearRegressionWithSGD.train(examples,numIterations,stepSize,miniBatchFraction)
    model.weights
    model.intercept

    //4 对样本进行测试
    val prediction = model.predict(examples.map(_.features))
    val predictionAndLabel = prediction.zip(examples.map((_.label)))
    val print_predict = predictionAndLabel.take(200)
    println("prediction" + "\t" + "label")

    for (i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    //5 计算测试误差
    //val loss = predictionAndLabel.map {
    //  case (p,1) =>
    //    val err = p - 1
    //    err * err
    //}.reduce(_ + _)
    //val rmse = math.sqrt(loss / numExamples)
    //println(s"Test RMSE = $rmse.")

    //6 模型保存
    //val ModelPath = "D://work//testdata//000000_0//linear//out"
    //val ModelPath = args(1)
    //model.save(sc,ModelPath)
    //val sameModel = LinearRegressionModel.load(sc,ModelPath)
  }

}
