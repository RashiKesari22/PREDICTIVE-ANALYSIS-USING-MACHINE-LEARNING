#!/usr/bin/env python3
# spark_ml_demo.py
# Demonstrates a full ML workflow: feature selection, pipeline, CV & evaluation.

import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def create_session(app_name: str = "SparkMLDemo"):
    return (SparkSession.builder.appName(app_name)
            .config("spark.sql.execution.arrow.enabled", "true")
            .getOrCreate())

def load_csv(spark, path):
    return spark.read.csv(path, header=True, inferSchema=True, sep=",")

def preprocess(df):
    # simple imputation
    return (df.na.fill(0).na.fill("unknown"))

def feature_selection(train_df, label_col):
    # 1️⃣ correlation filter
    numeric_cols = [c for c, t in train_df.dtypes
                    if t in ("int", "double") and c != label_col]
    corr = train_df.select(numeric_cols + [label_col]).toPandas().corr()
    high_corr = [c for c in numeric_cols if abs(corr[label_col][c]) > 0.05]

    # 2️⃣ categorical handling
    cat_cols = [c for c, t in train_df.dtypes
                if t == "string" and c != label_col]

    stages = []
    if cat_cols:
        indexer = StringIndexer(
            inputCols=cat_cols,
            outputCols=[f"{c}_idx" for c in cat_cols],
            handleInvalid="keep"
        )
        stages.append(indexer)
        assembled_cols = high_corr + [f"{c}_idx" for c in cat_cols]
    else:
        assembled_cols = high_corr

    # Ensure we have something to assemble
    if not assembled_cols:
        # fallback: use all numeric columns (or add a dummy)
        assembled_cols = numeric_cols
        if not assembled_cols:
            raise ValueError("No numeric or categorical columns found for feature selection.")

    assembler = VectorAssembler(inputCols=assembled_cols, outputCol="features")
    chi_sq = ChiSqSelector(numTopFeatures=10, labelCol=label_col,
                           featuresCol="features")
    stages.extend([assembler, chi_sq])

    selector_pipe = Pipeline(stages=stages)
    selector_model = selector_pipe.fit(train_df)

    # 3️⃣ extract selected feature names
    transformed = selector_model.transform(train_df)
    meta = transformed.select("features").schema["features"].metadata
    if "ml_attr" in meta and "attrs" in meta["ml_attr"]:
        feats = [f["name"] for f in meta["ml_attr"]["attrs"].get("numeric", [])]
        if feats:
            return feats
    # fallback
    return assembled_cols

def train_model(train_df, label_col, feature_cols):
    if not feature_cols:
        raise ValueError("No features selected – check your data or thresholds.")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    rf = RandomForestClassifier(labelCol=label_col, featuresCol="features",
                                numTrees=200, maxDepth=10, seed=42)
    pipeline = Pipeline(stages=[assembler, rf])

    (train, test) = train_df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train)

    preds = model.transform(test)
    mc_eval = MulticlassClassificationEvaluator(labelCol=label_col)
    bin_eval = BinaryClassificationEvaluator(labelCol=label_col)

    acc = mc_eval.evaluate(preds, {mc_eval.metricName: "accuracy"})
    f1 = mc_eval.evaluate(preds, {mc_eval.metricName: "f1"})
    auc = bin_eval.evaluate(preds)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1‑score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: spark-submit spark_ml_demo.py <input_csv> <target_column>")
        sys.exit(1)

    input_path, target_col = sys.argv[1], sys.argv[2]

    spark = create_session()
    raw = load_csv(spark, input_path)
    df = preprocess(raw)

    # run feature selection on a small sample for speed
    sample_df = df.sample(0.2, seed=42)
    selected = feature_selection(sample_df, target_col)
    print("Selected features:", selected)

    #model = train_model(df, target_col, selected)
    #model.write().overwrite().save("spark_rf_model")
    spark.stop()

if __name__ == "__main__":
    main()