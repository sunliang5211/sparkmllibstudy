package com.sunliang.sparkmllibstudy.three.breeze

import breeze.linalg._
import breeze.numerics._

/**
  * Created by sun on 2018/11/12.
  */
object BreezeTest {

  def main(args: Array[String]): Unit = {
    //breezeTest
    //access
    //operate
    //cal
    //sumtest
    //bool
    //linear
    //coundtest
    //constant
    trigonometric
  }

  def breezeTest(): Unit = {
    val m1 = DenseMatrix.zeros[Double](2,3)
    println(m1)
    val v1 = DenseVector.zeros[Double](3)
    println(v1)
    val v2 = DenseVector.ones[Double](3)
    println(v2)
    val v3 = DenseVector.fill(3){5.0}
    println(v3)
    val v4 = DenseVector.range(1,10,2)
    println(v4)
    val m2 = DenseMatrix.eye[Double](3)
    println(m2)
    val m3 = diag(DenseVector(1.0,2.0,3.0))
    println(m3)
    val m4 = DenseMatrix((1.0,2.0),(3.0,4.0))
    println(m4)
    val v5 = DenseVector(1,2,3,4)
    println(v5)
    val v6 = DenseVector(1,2,3,4).t
    println(v6)
    val v7 = DenseVector.tabulate(3){i => 2*i}
    println(v7)
    val m5 = DenseMatrix.tabulate(3,2){case (i,j) => i+j}
    println(m5)
    val v8 = new DenseVector(Array(1,2,3,4))
    println(v8)
    val m6 = new DenseMatrix(2,3,Array(11,12,13,21,22,23))
    println(m6)
    val v9 = DenseVector.rand(4)
    println(v9)
    val m7 = DenseMatrix.rand(2,3)
    println(m7)
  }

  def access(): Unit = {
    val a = DenseVector(1,2,3,4,5,6,7,8,9,10)
    println(a)
    println(a(0))
    println(a(0,1))
    println(a(1 to 4))
    println(a(5 to 0 ))
    println(a(5 to 0 by -1))
    println(a(1 to -1))
    println(a(-1))

    val m = DenseMatrix((1.0,2.0,3.0),(3.0,4.0,5.0))
    println(m)
    println(m(0,1))
    println(m(::,1))
    println(m(1,::))
  }

  def operate(): Unit ={
    val m = DenseMatrix((1.0,2.0,3.0),(3.0,4.0,5.0))
    println(m)
    println(m.reshape(3,2))
    println(m.toDenseVector)

    val m1 = DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0))
    println(m1)
    println(lowerTriangular(m1))
    println(upperTriangular(m1))

    val m2 = m.copy
    println(m2)

    println(diag(m1))
    m1(::,2) := 5.0
    println(m1)

    m1(1 to 2,1 to 2) := 5.0
    println(m1)

    val a = DenseVector(1,2,3,4,5,6,7,8,9,10)
    a(1 to 4) := 5
    println(a)
    a(1 to 4) := DenseVector(1,2,3,4)
    println(a)

    val a1 = DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0))
    val a2 = DenseMatrix((1.0,1.0,1.0),(2.0,2.0,2.0))

    val va = DenseMatrix.vertcat(a1,a2)
    println(va)
    val ha = DenseMatrix.horzcat(a1,a2)
    println(ha)

    val b1 = DenseVector(1,2,3,4)
    val b2 = DenseVector(1,1,1,1)
    val vb = DenseVector.vertcat(b1,b2)
    println(vb)
    val hb = DenseVector.horzcat(b1,b2)
    println(hb)

  }

  def cal(): Unit ={
    val a = DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0))
    val b = DenseMatrix((1.0,1.0,1.0),(2.0,2.0,2.0))

    println(a + b)
    println(a :* b)
    println(a :/ b)
    println(a :< b)
    println(a :== b)
    println(a :+= b)
    println(a :*= b)
    println(max(a))
    println(argmax(a))

    val va = DenseVector(1,2,3,4)
    val vb = DenseVector(1,1,1,2)
    println(va dot vb)
  }

  def sumtest(): Unit ={
    val a = DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0))
    println(sum(a))
    println(sum(a,Axis._0))
    println(sum(a,Axis._1))
    println(trace(a))
    println(accumulate(DenseVector(1,2,3,4)))
  }

  def bool(): Unit ={
    val a = DenseVector(true,false,true)
    val b = DenseVector(false,true,true)

    println(a :& b)
    println(a :| b)
    println(!a)

    val v = DenseVector(1.0,0.0,-2.0)
    println(any(a))
    println(all(a))

  }

  def linear(): Unit ={
    val a = DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0))
    val b = DenseMatrix((1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0))

    println(a)
    println(a \ b)
    println(a.t)
    println(det(a))
    println(inv(a))
    println(pinv(a))
    //println(norm(a))
    //println(eigSym(a))
    //val svd.SVD(u,s,v) = svd(a)
    println(svd(a))
    println(rank(a))
    println(a.rows)
    println(a.cols)
  }

  def coundtest(): Unit ={
    val a = DenseVector(1.2,0.6,-2.3)

    println(round(a))
    println(ceil(a))
    println(floor(a))
    println(signum(a))
    println(abs(a))
  }

  def constant(): Unit ={
    println(NaN)
    println(Inf)

  }

  def trigonometric(): Unit ={

  }

  def logarithm(): Unit ={

  }

}
