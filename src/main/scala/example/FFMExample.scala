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

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.rdd.RDD

import com.intel.imllib.ffm.classification._

object FFMExample extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("FFMExample"))

    if (args.length != 7) {
      println("FFMExample <train_file> <k> <n_iters> <eta> <lambda> <normal> <random>")
    }

    val data= sc.textFile(args(0)).map(_.split("\\s")).map(x => {
      val y = if(x(0).toInt > 0 ) 1.0 else -1.0
      val nodeArray: Array[(Int, Int, Double)] = x.drop(1).map(_.split(":")).map(x => {
        (x(0).toInt, x(1).toInt, x(2).toDouble)
      })
      (y, nodeArray)
    }).repartition(4)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (training: RDD[(Double, Array[(Int, Int, Double)])], testing) = (splits(0), splits(1))

    //sometimes the max feature/field number would be different in training/testing dataset,
    // so use the whole dataset to get the max feature/field number
    val m = data.flatMap(x=>x._2).map(_._1).collect.reduceLeft(_ max _) //+ 1
    val n = data.flatMap(x=>x._2).map(_._2).collect.reduceLeft(_ max _) //+ 1

    val ffm: FFMModel = FFMWithAdag.train(training, m, n, dim = (args(5).toBoolean, args(6).toBoolean, args(1).toInt), n_iters = args(2).toInt,
      eta = args(3).toDouble, lambda = args(4).toDouble, normalization = false, false, "adagrad")

    val scores: RDD[(Double, Double)] = testing.map(x => {
      val p = ffm.predict(x._2)
      val ret = if (p >= 0.5) 1.0 else -1.0
      (ret, x._1)
    })
    val accuracy = scores.filter(x => x._1 == x._2).count().toDouble / scores.count()
    println(s"accuracy = $accuracy")
  }
}

