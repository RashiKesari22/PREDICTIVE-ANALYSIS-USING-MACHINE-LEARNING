# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

COMPANY : CODTECH IT SOLUTIONS

NAME : RASHI KESARI

INTERN ID:CT04DR3010

DOMAIN : DATA ANALYTICS

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

**DESCRIPTION OF TASK

Building a predictive model using machine learning involves several key steps. First, relevant historical data is collected and pre-processed. Feature selection identifies the most influential factors, and a suitable algorithm is chosen. The model is trained on the data and evaluated using performance metrics.

The goal is to create a robust model that accurately predicts outcomes, enabling informed decision-making. Python libraries like scikit-learn provide the necessary tools for development and evaluation. A well-built model can be applied to various industries, such as finance, healthcare, or marketing, to drive business growth and improvement.

A typical predictive analysis project involves data collection, feature selection, model training, and evaluation. For example, a regression model can be used to predict continuous outcomes, while classification models predict categorical outcomes. 

By applying machine learning techniques, businesses can uncover hidden patterns and trends, making data-driven decisions. This project provides hands-on experience with predictive analysis, enhancing skills in machine learning and problem-solving. The techniques learned can be applied to real-world problems, driving innovation and growth. Effective predictive models can be an asset for organizations.


**PREREQUISITE

To get the script running we will need: 

1. Python 3spark_ml_demo.py pizza_sales.csvhon3`.
2. Apache Spark– install via Homebrew or download the binary:

bash
brew install apache-spark

Make sure `SPARK_HOME` points to the Spark installation and `$SPARK_HOME/bin` is in your `PATH`.
3. Java JDK 17 (or any JDK ≥ 11 that matches the Spark build). Set `JAVA_HOME` to that JDK.
4. PySpark – usually bundled with Spark, but you can install it explicitly:

bash
pip install pyspark

5. Pandas (for the correlation step):

bash
pip install pandas

6. A CSV file that matches the schema expected by the script (header row, comma‑separated).

Once those are in place, you can submit the job:

bash
spark-submit spark_ml.py pizza.csv target


**OUTPUT

Selected features: ['order_id', 'price']
Accuracy : 1.0000
F1‑score : 1.0000
AUC      : 1.0000

**INSIGHTS DERIVED FROM MACHINE LEARNING MODEL

What the script does?
1. Loads your CSV into a Spark DataFrame.
2. Fills missing values (0 for numbers, “unknown” for strings).
3. Picks the most useful columns:
    - Numeric fields that correlate (│ρ│ > 0.05) with the target.
    - Categorical fields that are indexed.
    - Uses Chi‑Square to keep only the top‑10 features overall.
4. Trains a Random Forest on those features (80 % train, 20 % test).
5. Prints three metrics on the hold‑out set.

What you will see in the console?
Selected features: ['price', 'quantity', 'category_idx', ...]
Accuracy : 1.0000
F1‑score : 1.0000
AUC : : 1.0000
- Selected features – the subset the algorithm deemed most predictive. Look at this list to spot the key drivers (e.g., price, quantity, a particular category).
- Accuracy– overall proportion of correct predictions. Good for balanced classes.
- F1‑score– harmonic mean of precision and recall; useful when false positives/negatives both matters.
- AUC – how well the model separates the two classes across thresholds (0.5 = random, 1.0 = perfect).

Interpretation
- High accuracy + high AUC (0.8‑0.9) means the model is doing a solid job.
- The feature list tells you which variables are influencing the outcome; investigate any surprises.
- If metrics are lower, consider more data, tweaking the correlation threshold, or trying a different algorithm.

That’s the gist of the insights.

