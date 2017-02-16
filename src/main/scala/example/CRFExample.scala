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

import java.io.{DataOutputStream, DataInputStream, FileInputStream, FileOutputStream}

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import com.intel.imllib.crf.nlp._

object CRFExample {

  def main(args: Array[String]) {
    if (args.length != 3) {
      println("CRFExample <templateFile> <trainFile> <testFile>")
    }

    val templateFile = args(0)
    val trainFile = args(1)
    val testFile = args(2)

    val conf = new SparkConf().setAppName("CRFExample")
    val sc = new SparkContext(conf)

    val templates: Array[String] = scala.io.Source.fromFile(templateFile).getLines().filter(_.nonEmpty).toArray
    val trainRDD: RDD[Sequence] = sc.textFile(trainFile).filter(_.nonEmpty).map(Sequence.deSerializer)

    val model: CRFModel = CRF.train(templates, trainRDD, 0.25, 1, 100, 1E-3, "L1")

    val testRDD: RDD[Sequence] = sc.textFile(testFile).filter(_.nonEmpty).map(Sequence.deSerializer)

    /**
      * an example of model saving and loading
      */
    new java.io.File("target/model").mkdir()
    //model save as String
    new java.io.PrintWriter("target/model/model1") { write(CRFModel.save(model)); close() }
    val modelFromFile1 = CRFModel.load(scala.io.Source.fromFile("target/model/model1").getLines().toArray.head)
    // model save as RDD
    sc.parallelize(CRFModel.saveArray(model)).saveAsTextFile("target/model/model2")
    val modelFromFile2 = CRFModel.loadArray(sc.textFile("target/model/model2").collect())
    // model save as BinaryFile
    val path = "target/model/model3"
    new java.io.File(path).mkdir()
    CRFModel.saveBinaryFile(model, path)
    val modelFromFile3 = CRFModel.loadBinaryFile(path)

    /**
      * still use the model in memory to predict
      */
    val results: RDD[Sequence] = model.setNBest(10)
      .setVerboseMode(VerboseLevel1)
      .predict(testRDD)

    val score = results
      .zipWithIndex()
      .map(_.swap)
      .join(testRDD.zipWithIndex().map(_.swap))
      .map(_._2)
      .map(x => x._1.compare(x._2))
      .reduce(_ + _)
    val total = testRDD.map(_.toArray.length).reduce(_ + _)
    println(s"Prediction Accuracy: $score / $total")

    sc.stop()
  }
}
