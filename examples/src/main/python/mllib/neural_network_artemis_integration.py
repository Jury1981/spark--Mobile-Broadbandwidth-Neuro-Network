#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Spark Neural Network Integration Example with Artemis1981 Configuration

This example demonstrates how to use Spark MLlib's MultilayerPerceptronClassifier
(native neural network support) along with a configuration framework for potential
integration with the Artemis1981 repository for mobile broadband neuro network processing.

Note: This example demonstrates Spark's built-in neural network capabilities. The
Artemis1981 configuration properties shown are part of a configuration framework and
would require additional implementation for actual integration.

Usage:
    spark-submit neural_network_artemis_integration.py

This example uses the MultilayerPerceptronClassifier from Spark MLlib
and demonstrates a configuration framework for Artemis1981 integration.
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors


def create_spark_session():
    """Create and configure Spark session with Neural Network and integration settings."""
    spark = SparkSession.builder \
        .appName("NeuralNetworkArtemisIntegration") \
        .config("spark.mllib.neuralnetwork.enabled", "true") \
        .config("spark.artemis.integration.enabled", "true") \
        .config("spark.artemis.repository", "https://github.com/Jury1981/Artemis1981") \
        .getOrCreate()
    
    print("=" * 80)
    print("Spark Neural Network with Artemis1981 Configuration Framework")
    print("=" * 80)
    print(f"Spark Version: {spark.version}")
    print(f"Neural Network Config Flag: {spark.conf.get('spark.mllib.neuralnetwork.enabled', 'false')}")
    print(f"Artemis Config Flag: {spark.conf.get('spark.artemis.integration.enabled', 'false')}")
    print(f"Artemis Repository: {spark.conf.get('spark.artemis.repository', 'not set')}")
    print("Note: Custom properties above are configuration framework settings")
    print("=" * 80)
    
    return spark


def create_sample_data(spark):
    """Create sample data for neural network training."""
    # Sample dataset for demonstration
    # Features: 4 dimensions, Labels: 3 classes
    data = [
        (Vectors.dense([0.0, 0.0, 0.0, 0.0]), 0.0),
        (Vectors.dense([0.0, 0.0, 1.0, 1.0]), 1.0),
        (Vectors.dense([1.0, 1.0, 0.0, 0.0]), 1.0),
        (Vectors.dense([1.0, 1.0, 1.0, 1.0]), 2.0),
        (Vectors.dense([0.5, 0.5, 0.5, 0.5]), 1.0),
    ]
    
    df = spark.createDataFrame(data, ["features", "label"])
    return df


def configure_neural_network():
    """Configure the neural network architecture."""
    # Specify layers for the neural network:
    # - Input layer: 4 features
    # - Hidden layers: 5 and 4 neurons
    # - Output layer: 3 classes
    layers = [4, 5, 4, 3]
    
    # Create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(
        maxIter=100,
        layers=layers,
        blockSize=128,
        seed=1234
    )
    
    print("\nNeural Network Configuration:")
    print(f"  Architecture: {layers}")
    print(f"  Input neurons: {layers[0]}")
    print(f"  Hidden layers: {layers[1:-1]}")
    print(f"  Output neurons: {layers[-1]}")
    print(f"  Max iterations: 100")
    
    return trainer


def main():
    """Main function to demonstrate Spark Neural Network with Artemis1981 integration."""
    # Create Spark session with Neural Network configuration
    spark = create_spark_session()
    
    try:
        # Create sample data
        print("\nCreating sample data...")
        train_data = create_sample_data(spark)
        test_data = create_sample_data(spark)
        
        # Configure and train neural network
        print("\nConfiguring neural network...")
        trainer = configure_neural_network()
        
        print("\nTraining neural network model...")
        model = trainer.fit(train_data)
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.transform(test_data)
        
        # Evaluate the model
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        
        print(f"\nModel Training Complete!")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Display predictions
        print("\nPrediction Results:")
        predictions.select("features", "label", "prediction").show()
        
        # Artemis1981 Configuration Framework Status
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("✓ Spark MLlib MultilayerPerceptronClassifier demonstrated")
        print("✓ Neural network training and prediction completed")
        print("✓ Configuration framework for Artemis1981 applied")
        print("  (Configuration properties can be used for custom integration)")
        print("=" * 80)
        
    finally:
        spark.stop()
        print("\nSpark session stopped.")


if __name__ == "__main__":
    main()
