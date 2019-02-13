package com.sunliang.sparkmllibstudy.twelve

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/16.
  */
object als1 {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("als1")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D://work//testdata//000000_0//als//als1.txt")
    val ratings = data.map(_.split(",") match {
      case Array(user,item,rate) =>
        println("user=" + user.toInt + " item=" + item.toInt + " rate=" + rate.toDouble)
        Rating(user.toInt,item.toInt,rate.toDouble)
    })

    val rank = 10
    val numIterations = 20
    val model = ALS.train(ratings,rank,numIterations,0.01)

    val usersProducts = ratings.map {
      case Rating(user,product,rate) =>
        (user,product)
    }

    val predictions =
      model.predict(usersProducts).map {
        case Rating(user,product,rate) =>
          ((user,product),rate)
      }

    val ratesAndPreds = ratings.map {
      case Rating(user,product,rate) =>
        ((user,product),rate)
    }.join(predictions)

    val MSE = ratesAndPreds.map {
      case ((user,product),(r1,r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()
    println("Mean Squared Error = " +  MSE)

    //val ModelPath = ""
    //model.save(sc,ModelPath)
    //val sameModel = MatrixFactorizationModel.load(sc,ModelPath)
  }

}
