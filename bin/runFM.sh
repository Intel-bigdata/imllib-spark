$SPARK_HOME/bin/spark-submit \
    --class FMExample \
    --master local[*] \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    hdfs://localhost:9000/data/fm/a9a \
    2 \
    3 \
    0.01 \
    2
