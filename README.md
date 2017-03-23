# IMLLIB

[![Build Status](https://travis-ci.org/Intel-bigdata/imllib-spark.svg?branch=master)](https://travis-ci.org/Intel-bigdata/imllib-spark)

A package contains three Spark-based implementations. It includes
 * Factorization Machines (LibFM)
 * Field-Awared Factorization Machine (FFM)
 * Conditional Random Fields (CRF)

This package can be imported as a dependency in other codes. Then, all functions of LibFM, FFM and CRF in this package can be used.

# Build from Source
```scala
sbt package
```

# How to Use
IMLLIB package can be either imported directly in Spark-Shell, or be imported as a dependency in other codes.

## Use in Spark-Shell
### Temporary Use
(1) Run Spark-Shell with IMLLIB package
```
spark-shell --jars 'Path/imllib_2.11-0.0.1.jar'
```
`Path` is the path of `imllib_2.11-0.0.1.jar`

### Permanent Use
(1) On driver node, add following codes into `conf/spark-default.conf` 
```
spark.executor.extraClassPath    /usr/local/spark/lib/*
spark.driver.extraClassPath      /usr/local/spark/lib/*
```
(2) Create `/usr/local/spark/lib`<br>
(3) Copy `imllib_2.11-0.0.1.jar` to `/usr/local/spark/lib`<br>
(4) Copy `conf/spark-default.conf` and `/usr/local/spark/lib` to all worker nodes <br>
(5) Run Spark-Shell
```
spark-shell
```

## Use as a denpendency
(1) build from source and publish locally
```scala
sbt compile publish-local
```
(2) Move the whole directory `com.intel` from `.ivy2/local` to `.ivy2/cache`<br>
(3) Add following codes into `build.sbt` when you want to import IMLLIB package as a denpendency
```
libraryDependencies += "com.intel" % "imllib_2.11" % "0.0.1"
```

# How to Import
```scala
import com.intel.imllib._
```
# Test Examples
There are three shell scripts in `bin/` for testing LibFM, FFM, and CRF respectively.
You can first modify the script with necessary changes, such as hostname, port for hadoop, etc.
Then run the script to test if the algorithm works.

# Test Examples
There are three shell scripts in `bin/` for testing LibFM, FFM, and CRF respectively. The script runs in a local mode Spark with the data on hadoop.
You can first modify the script with necessary changes, such as hostname, port for hadoop, etc. Then run the script to test if the algorithm works.

---
## FM-Spark
A Spark-based implementation of Factorization Machines (LibFM)

The code base structure of this project is from [spark-libFM](https://github.com/zhengruifeng/spark-libFM), but the optimization method is based on [parallel-sgd](http://www.research.rutgers.edu/~lihong/pub/Zinkevich11Parallelized.pdf
) which has stronger convergence than miniBatch-sgd.

## FFM-Spark
A Spark-based implementation of Field-Awared Factorization Machine with parallelled AdaGrad solver.
See http://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf

Need to rework the data format the fit FFM, that is to extends LIBSVM data format by adding field
information to each feature to have formation like:
        label field1:feat1:val1 field2:feat2:val2

## CRF-Spark
A Spark-based implementation of Conditional Random Fields (CRF) for segmenting/labeling sequential data.

`CRF-Spark` provides following features:
* Training in parallel based on Spark RDD
* Support a simple format of training and test file. Any other format also can be read by a simple tokenizer.
* A common feature templates design, which is also used in other machine learning tools, such as [CRF++](https://taku910.github.io/crfpp/) and [miralium](https://code.google.com/archive/p/miralium/)
* Fast training based on Limited-memory BFGS optimizaton algorithm (fitting L2-regularized models) or Orthant-wise Limited Memory Quasi-Newton optimizaton algorithm (fitting L1-regularized models)
* Support two verbose levels to provide extra information. VerboseLevel1 shows marginal probabilities for each tag and a conditional probability for the entire output; VerboseLevel2 shows marginal probabilities for all other candidates additionally.
* Support n-best outputs
* Linear-chain (first-order Markov) CRF
* Test can run both in parallel and in serial

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
 Also you can mail to:
 * hqzizania(Intel)
 * [mpjlu](https://github.com/mpjlu)
 * [VinceShieh](https://github.com/VinceShieh)
 * [chenghao-intel](https://github.com/chenghao-intel)
 * [ynXiang](https://github.com/ynXiang)

