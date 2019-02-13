package com.sunliang.sparkmllibstudy.twelve

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sun on 2018/11/16.
  */
object fpg {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("fpg")
    val sc = new SparkContext(conf)

    val data_path = "D://work//testdata//000000_0//fpg//fpg1.txt"
    val data = sc.textFile(data_path)
    val examples = data.map(_.split(" ")).cache()

    val minSupport = 0.2
    val numPartition = 10
    val model = new FPGrowth().setMinSupport(minSupport).setNumPartitions(numPartition).run(examples)

    println(s"Number of frequent itemsets: ${model.freqItemsets.count()}")
    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[",",","]") + ", " + itemset.freq)
    }
  }

}
