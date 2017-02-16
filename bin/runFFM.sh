$SPARK_HOME/bin/spark-submit \
    --class FFMExample \
    --master local[*] \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    hdfs://localhost:9000/data/ffm/a9a_ffm \
    2 \
    3 \
    0.01 \
    0.00002 \
    false \
    false \
    2
