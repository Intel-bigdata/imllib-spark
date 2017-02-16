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
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import com.intel.imllib.fm.regression._

object FMExample extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("FMExample"))

    if (args.length != 5) {
      println("FMExample <train_file> <partitions> <n_iters> <stepSize> <k>")
    }

    //customer dataset format convertor
    
    val rawData = sc.textFile(args(0)).map(_.split("\\s")).map(x => {
      if (x(0).toInt > 0)
        x(0) = "1"
      else
        x(0) = "-1"
      val v: Array[(Int, Double)] = x.drop(1).map(_.split(":"))
        .map(x => (x(0).toInt - 1, x(1).toDouble))
        .sortBy(_._1)
      (x(0).toInt, v)
    }).repartition(args(1).toInt)

    val length = rawData.map(_._2.last._1).max + 1
println("data size:" + rawData.count +",feature size:" + length + ",k:" + args(4).toInt + ",stepSize:" + args(3))
    val data: RDD[LabeledPoint] = rawData.map{case(label, v) => LabeledPoint(label, Vectors.sparse(length, v.map(_._1), v.map(_._2)))}
    
    val splits = data.randomSplit(Array(0.8, 0.2))
    val (training, testing) = (splits(0), splits(1))
   
    val fm1: FMModel = FMWithSGD.train(training, task = 1, numIterations
      = args(2).toInt, stepSize = args(3).toDouble, dim = (false, true, args(4).toInt), regParam = (0, 0.0, 0.01), initStd = 0.01)

    val scores: RDD[(Int, Int)] = fm1.predict(testing.map(_.features)).map(x => if(x >= 0.5) 1 else -1).zip(testing.map(_.label.toInt))
    val accuracy = scores.filter(x => x._1 == x._2).count().toDouble / scores.count()

    println(s"accuracy = $accuracy")
    sc.stop()
  }
}
