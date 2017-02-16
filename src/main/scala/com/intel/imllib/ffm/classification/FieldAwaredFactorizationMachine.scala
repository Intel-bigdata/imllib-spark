/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.imllib.ffm.classification

import java.io._

import breeze.linalg.{DenseVector => BDV}

import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.optimization.Gradient

import scala.util.Random

/**
  * Created by vincent on 16-12-19.
  */
/**
  *
  * @param numFeatures number of features
  * @param numFields number of fields
  * @param dim A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the
  *            one-way interactions should be used, and the number of factors that are used for pairwise
  *            interactions, respectively.
  * @param n_iters number of iterations
  * @param eta step size to be used for each iteration
  * @param lambda regularization for pairwise interations
  * @param isNorm whether normalize data
  * @param random whether randomize data
  * @param weights weights of FFMModel
  * @param sgd "true": parallelizedSGD, parallelizedAdaGrad would be used otherwise
  */
class FFMModel(numFeatures: Int,
               numFields: Int,
               dim: (Boolean, Boolean, Int),
               n_iters: Int,
               eta: Double,
               lambda: Double,
               isNorm: Boolean, random: Boolean,
               weights: Array[Double],
               sgd: Boolean = true ) extends Serializable {

  private var n: Int = numFeatures
  //numFeatures
  private var m: Int = numFields
  //numFields
  private var k: Int = dim._3
  //numFactors
  private var k0 = dim._1
  private var k1 = dim._2
  private var normalization: Boolean = isNorm
  private var initMean: Double = 0
  private var initStd: Double = 0.01

  require(n > 0 && k > 0 && m > 0)

  def radomization(l: Int, rand: Boolean): Array[Int] = {
    val order = Array.fill(l)(0)
    for (i <- 0 to l - 1) {
      order(i) = i
    }
    if (rand) {
      val rand = new Random()
      for (i <- l - 1 to 1) {
        val tmp = order(i - 1)
        val index = rand.nextInt(i)
        order(i - 1) = order(index)
        order(index) = tmp
      }
    }
    return order
  }

  def setOptimizer(op: String): Boolean = {
    if("sgd" == op) true else false
  }

  def predict(data: Array[(Int, Int, Double)], r: Double = 1.0): Double = {

    var t = if (k0) weights(weights.length - 1) else 0.0

    val (align0, align1) = if(sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }

    // j: feature, f: field, v: value
    val valueSize = data.size //feature length
    val indicesArray = data.map(_._2) //feature index
    val valueArray: Array[(Int, Double)] = data.map(x => (x._1, x._3))
    var i = 0
    var ii = 0
    val pos = if (sgd) n * m * k else n * m * k * 2
    // j: feature, f: field, v: value
    while (i < valueSize) {
      val j1 = data(i)._2
      val f1 = data(i)._1
      val v1 = data(i)._3

      if(k1) t += weights(pos + j1) * v1

      ii = i + 1
      if (j1 < n && f1 < m) {
        while (ii < valueSize) {
          val j2 = data(ii)._2
          val f2 = data(ii)._1
          val v2 = data(ii)._3
          if (j2 < n && f2 < m) {
            val w1_index: Int = j1 * align1 + f2 * align0
            val w2_index: Int = j2 * align1 + f1 * align0
            val v: Double = v1 * v2 * r
            for (d <- 0 to k - 1) {
              t += weights(w1_index + d) * weights(w2_index + d) * v
            }
          }
          ii += 1
        }
      }
      i += 1
    }
    t
  }
}

class FFMGradient(m: Int, n: Int, dim: (Boolean, Boolean, Int), sgd: Boolean = true) extends Gradient {

  private val k0 = dim._1
  private val k1 = dim._2
  private val k = dim._3

  private def predict (data: Array[(Int, Int, Double)], weights: Array[Double], r: Double = 1.0): Double = {

    var t = if (k0) weights(weights.length - 1) else 0.0

    val (align0, align1) = if(sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }
    val valueSize = data.size //feature length
    val indicesArray = data.map(_._2) //feature index
    val valueArray: Array[(Int, Double)] = data.map(x => (x._1, x._3))
    var i = 0
    var ii = 0
    val a = data.size
    val b = indicesArray.length
    val c = valueArray.length
    val tt = 0
    val pos = if (sgd) n * m * k else n * m * k * 2
    // j: feature, f: field, v: value
    while (i < valueSize) {
      val j1 = data(i)._2
      val f1 = data(i)._1
      val v1 = data(i)._3
      ii = i + 1

      if(k1) t += weights(pos + j1) * v1

      if (j1 < n && f1 < m) {
        while (ii < valueSize) {
        val j2 = data(ii)._2
        val f2 = data(ii)._1
        val v2 = data(ii)._3
        if (j2 < n && f2 < m) {
          val w1_index: Int = j1 * align1 + f2 * align0
          val w2_index: Int = j2 * align1 + f1 * align0
          val v: Double = v1 * v2 * r
          for (d <- 0 to k - 1) {
            t += weights(w1_index + d) * weights(w2_index + d) * v
          }
        }
        ii += 1
      }
    }
      i += 1
    }
    t
  }

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    throw new Exception("This part is merged into computeFFM()")
  }

  override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    throw new Exception("This part is merged into computeFFM()")
  }
  def computeFFM(label: Double, data2: Array[(Int, Int, Double)], weights: Vector,
                 r: Double = 1.0, eta: Double, lambda: Double,
                 do_update: Boolean, iter: Int, solver: Boolean = true): (BDV[Double], Double) = {
    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values
    val t = predict(data2, weightsArray, r)
    val expnyt = math.exp(-label * t)
    val tr_loss = math.log(1 + expnyt)
    val kappa = -label * expnyt / (1 + expnyt)
    val (align0, align1) = if(sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }
    val valueSize = data2.size //feature length
    val indicesArray = data2.map(_._2) //feature index
    val valueArray: Array[(Int, Double)] = data2.map(x => (x._1, x._3))
    var i = 0
    var ii = 0

    val r0, r1 = 0.0
    val pos = if (sgd) n * m * k else n * m * k * 2
    if(k0) weightsArray(weightsArray.length - 1) -= eta * (kappa + r0 * weightsArray(weightsArray.length - 1))
    // j: feature, f: field, v: value
    while (i < valueSize) {
      val j1 = data2(i)._2
      val f1 = data2(i)._1
      val v1 = data2(i)._3
      if(k1) weightsArray(pos + j1) -= eta * (v1 * kappa + r1 * weightsArray(pos + j1))
      if (j1 < n && f1 < m) {
        ii = i + 1
        while (ii < valueSize) {
          val j2 = data2(ii)._2
          val f2 = data2(ii)._1
          val v2 = data2(ii)._3
          if (j2 < n && f2 < m) {
            val w1_index: Int = j1 * align1 + f2 * align0
            val w2_index: Int = j2 * align1 + f1 * align0
            val v: Double = v1 * v2 * r
            val wg1_index: Int = w1_index + k
            val wg2_index: Int = w2_index + k
            val kappav: Double = kappa * v
            for (d <- 0 to k - 1) {
              val g1: Double = lambda * weightsArray(w1_index + d) + kappav * weightsArray(w2_index + d)
              val g2: Double = lambda * weightsArray(w2_index + d) + kappav * weightsArray(w1_index + d)
              if (sgd) {
                weightsArray(w1_index + d) -= eta * g1
                weightsArray(w2_index + d) -= eta * g2
              } else {
                val wg1: Double = weightsArray(wg1_index + d) + g1 * g1
                val wg2: Double = weightsArray(wg2_index + d) + g2 * g2
                weightsArray(w1_index + d) -= eta / (math.sqrt(wg1)) * g1
                weightsArray(w2_index + d) -= eta / (math.sqrt(wg2)) * g2
                weightsArray(wg1_index + d) = wg1
                weightsArray(wg2_index + d) = wg2

              }
            }
          }
          ii += 1
        }
      }
      i += 1
    }
    (BDV(weightsArray), tr_loss)
  }
}
