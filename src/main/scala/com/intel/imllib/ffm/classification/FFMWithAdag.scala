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

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import com.intel.imllib.ffm.optimization._

import scala.util.Random
/**
  * Created by vincent on 17-1-4.
  */
/**
  *
  * @param m number of fields of input data
  * @param n number of features of input data
  * @param dim A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the
  *            one-way interactions should be used, and the number of factors that are used for pairwise
  *            interactions, respectively.
  * @param n_iters number of iterations
  * @param eta step size to be used for each iteration
  * @param lambda regularization for pairwise interactions
  * @param normalization whether normalize data
  * @param random whether randomize data
  * @param solver "sgd": parallelizedSGD, parallelizedAdaGrad would be used otherwise
  */
class FFMWithAdag(m: Int, n: Int, dim: (Boolean, Boolean, Int), n_iters: Int, eta: Double, lambda: Double,
                  normalization: Boolean, random: Boolean, solver: String) extends Serializable {
  private val k0 = dim._1
  private val k1 = dim._2
  private val k = dim._3
  private val sgd = setOptimizer(solver)

  println("get numFields:" + m + ",nunFeatures:" + n + ",numFactors:" + k)
  private def generateInitWeights(): Vector = {
    val (num_k0, num_k1) = (k0, k1) match {
      case (true, true) =>
        (n, 1)
      case(true, false) =>
        (n, 0)
      case(false, true) =>
        (0, 1)
      case(false, false) =>
        (0, 0)
    }
    val W = if(sgd){
      val tmpSize = n * m * k + num_k1 + num_k0
      println("allocating:" + tmpSize)
      new Array[Double](n * m * k + num_k1 + num_k0)
    } else {
      val tmpSize = n * m * k * 2 + num_k1 + num_k0
      println("allocating:" + tmpSize)
      new Array[Double](n * m * k * 2 + num_k1 + num_k0)
    }
    val coef = 1.0 / Math.sqrt(k)
    val random = new Random()
    var position = 0
    if(sgd) {
      for (j <- 0 to n - 1; f <- 0 to m - 1; d <- 0 to k - 1) {
        W(position) = coef * random.nextDouble()
        position += 1
      }
    } else {
      for (j <- 0 to n - 1; f <- 0 to m - 1; d <- 0 to 2 * k - 1) {
        W(position) = if (d < k) coef * random.nextDouble() else 1.0
        position += 1
      }
    }
    if (k1) {
      for (p <- 0 to n - 1) {
        W(position) = 0.0
        position += 1
      }
    }
    if (k0) W(position) = 0.0
    Vectors.dense(W)
  }

  /**
    * Create a FFMModle from an encoded vector.
    */
  private def createModel(weights: Vector): FFMModel = {
    val values = weights.toArray
    new FFMModel(n, m, dim, n_iters, eta, lambda, normalization, random, values, sgd)
  }

  /**
    * Run the algorithm with the configured parameters on an input RDD
    * of FFMNode entries.
    */
  def run(input: RDD[(Double, Array[(Int, Int, Double)])]): FFMModel = {
    val gradient = new FFMGradient(m, n, dim, sgd)
    val optimizer = new GradientDescentFFM(gradient, null, k, n_iters, eta, lambda, normalization, random)

    val initWeights = generateInitWeights()
    val weights = optimizer.optimize(input, initWeights,n_iters, eta, lambda, sgd)
    createModel(weights)
  }

  def setOptimizer(op: String): Boolean = {
    if("sgd" == op) true else false
  }

}

object FFMWithAdag {
  /**
    *
    * @param data input data RDD
    * @param m number of fields of input data
    * @param n number of features of input data
    * @param dim A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the
    *            one-way interactions should be used, and the number of factors that are used for pairwise
    *            interactions, respectively.
    * @param n_iters number of iterations
    * @param eta step size to be used for each iteration
    * @param lambda regularization for pairwise interactions
    * @param normalization whether normalize data
    * @param random whether randomize data
    * @param solver "sgd": parallelizedSGD, parallelizedAdaGrad would be used otherwise
    * @return FFMModel
    */
  def train(data: RDD[(Double, Array[(Int, Int, Double)])], m: Int, n: Int,
            dim: (Boolean, Boolean, Int), n_iters: Int, eta: Double, lambda: Double, normalization: Boolean, random: Boolean,
            solver: String = "sgd"): FFMModel = {
    new FFMWithAdag(m, n, dim, n_iters, eta, lambda, normalization, random, solver)
      .run(data)
  }
}
