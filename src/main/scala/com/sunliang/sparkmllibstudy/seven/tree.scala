package com.sunliang.sparkmllibstudy.seven

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/16.
  */
object tree {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("tree")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,"D://work//testdata//000000_0//tree//tree1.txt")
    //val data = MLUtils.loadLibSVMFile(sc,"D://work//testdata//000000_0//logistic//logistic2.txt")
    val splits = data.randomSplit(Array(0.7,0.3))
    val (trainingData,testData) = (splits(0),splits(1))

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int,Int]()
    val impruity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData,numClasses,categoricalFeaturesInfo,impruity,maxDepth,maxBins)

    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label,prediction)
    }

    val print_predict = labelAndPreds.take(20)
    println("label" + "\t" + "prediction")
    for(i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    val testErr = labelAndPreds.filter(r => r._1 !=  r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classfication tree model:\n" + model.toDebugString)

    //val ModelPath = ""
    //model.save(sc,ModelPath)
    //val sameModel = DecisionTreeModel.load(sc,ModelPath)
  }
}
