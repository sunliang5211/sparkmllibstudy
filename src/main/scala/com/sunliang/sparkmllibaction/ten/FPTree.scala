package com.sunliang.sparkmllibaction.ten

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.{SparkConf, SparkContext}
object FPTree {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("FPTree ")
    val sc = new SparkContext(conf)
    val data = sc.textFile("D://work//testdata//000000_0//fpg//fpg1.txt")
    val examples = data.map(_.split(" ")).cache()
    val fpg = new FPGrowth().setMinSupport(0.3).setNumPartitions(10)
    val model = fpg.run(examples)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[",",","]") + ", " + itemset.freq)
    }

  }
}
