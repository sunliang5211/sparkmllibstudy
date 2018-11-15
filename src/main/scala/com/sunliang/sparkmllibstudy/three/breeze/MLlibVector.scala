package com.sunliang.sparkmllibstudy.three.breeze

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix


/**
  * Created by sun on 2018/11/13.
  */
object MLlibVector {

  def main(args: Array[String]): Unit = {
    //mllibtest
    //matrixtest
    distributeMatrix
  }

  def mllibtest(): Unit = {
    val v1 = Vectors.dense(1,2,3,4,5,6,7,8,9,10)
    val v11 = Vectors.dense(Array[Double](1,2,3,4,5,6,7,8,9,12))
    println(v1)
    println(v11)

    val v111 = Vectors.norm(v1,1)
    println("v111=" + v111)
    val v112 = Vectors.sqdist(v1,v11)
    println("v112=" + v112)

    val v2 = Vectors.zeros(5)
    println(v2)

    val v3 = Vectors.sparse(10,Array(1,3,5,7,9),Array(1,1,1,1,1))
    println(v3)
    val v31 = Vectors.sparse(10,Seq((1,2.0),(3,5.0)))
    println(v31)

    val v4 = new DenseVector(Array(1,2,3,4,5,6,7,8,9,10))
    println(v4)
    val v5 = new SparseVector(10,Array(1,3,5,7,9),Array(1,1,1,1,1))
    println(v5)
    val v51 = v5.toDense
    val v52 = v5.toArray
    println(v51)
    println(v52)

  }

  def matrixtest(): Unit ={
    val v4 = new DenseVector(Array(1,2,3,4,5,6,7,8,9,10))
    val m1 = Matrices.dense(3,3,Array(1,2,3,4,5,6,7,8,9))

    println(m1)
    println(Matrices.diag(v4))
    println(Matrices.eye(5))
    println(Matrices.ones(3,2))
    println(Matrices.zeros(3,3))

    val m2 = new DenseMatrix(3,3,Array(1,2,3,4,5,6,7,8,9))
    println("m2=" + m2)
    println(m2.isTransposed)
    println(m2.transpose)

  }

  def distributeMatrix(): Unit ={
    val conf = new SparkConf().setMaster("local").setAppName("distributeMatrix")
    val sc = new SparkContext(conf)
    val rdd1 = sc.parallelize(Array(Array(1.0,2.0,3.0),Array(4.0,5.0,6.0),Array(7.0,8.0,9.0))).map(f => Vectors.dense(f))
    val rm = new RowMatrix(rdd1)
    val simic1 = rm.columnSimilarities()
    println(simic1.numCols())
    println(simic1.numRows())

    val simic3 = rm.computeColumnSummaryStatistics()
    println(simic3.max)
    println(simic3.min)
    println(simic3.mean)

    val cc1 = rm.computeCovariance()
    println(cc1)

    val cc2 = rm.computeGramianMatrix()
    println(cc2)

    val cc3 = rm.computePrincipalComponents(3)
    println(cc3)
  }

}
