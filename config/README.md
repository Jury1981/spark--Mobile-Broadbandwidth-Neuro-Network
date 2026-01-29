# Spark Neural Network and Artemis1981 Integration Configuration

This directory contains configuration templates for using Spark MLlib Neural Networks and establishing a configuration framework for integration with the Jury1981/Artemis1981 repository.

## Files

- **neural-network.conf.template**: Configuration template with settings for neural network operations and Artemis integration framework
- **artemis-integration.properties.template**: Properties template for organizing Artemis1981 repository integration settings

## Important Notes

- **Native Spark Functionality**: Apache Spark MLlib natively supports neural networks through `MultilayerPerceptronClassifier` without requiring additional configuration.
- **Configuration Framework**: The custom properties in these files provide a structured way to organize application-level settings and integration configurations.
- **Integration Implementation**: Actual integration with Artemis1981 would require additional implementation based on specific requirements.

## Usage

1. Copy the configuration files to the `conf/` directory:

```bash
# Copy neural network configuration
cp config/neural-network.conf.template conf/spark-defaults.conf

# Copy Artemis integration properties
cp config/artemis-integration.properties.template conf/artemis-integration.properties
```

2. Edit the files to customize settings for your environment.

3. Start Spark with the configuration:

```bash
./bin/spark-shell
# or
./bin/pyspark
```

## Configuration Options

### Neural Network Settings

These are custom configuration properties for application use:

- `spark.mllib.neuralnetwork.enabled`: Application flag for neural network mode (default: true)
- `spark.mllib.neuralnetwork.defaultLayers`: Default number of hidden layers for applications (default: 3)
- `spark.mllib.neuralnetwork.learningRate`: Default learning rate for applications (default: 0.01)
- `spark.mllib.neuralnetwork.maxIterations`: Default maximum training iterations (default: 100)

### Mobile Broadband Settings

Custom properties for broadband processing applications:

- `spark.broadband.neuro.enabled`: Enable broadband neuro network mode (default: true)
- `spark.broadband.neuro.batchSize`: Batch size for training (default: 32)
- `spark.broadband.neuro.optimizer`: Optimizer algorithm preference (default: adam)

### Artemis1981 Integration Framework

Configuration framework for Artemis1981 integration:

- `spark.artemis.repository`: URL to Artemis1981 repository
- `spark.artemis.integration.enabled`: Enable integration framework (default: true)
- `artemis.data.sync.enabled`: Flag for data synchronization feature (default: true)
- `artemis.neuralnet.coordination`: Neural network coordination setting (default: enabled)

**Note**: These properties serve as a configuration framework. Actual integration requires additional implementation.

## See Also

- [Neural Network Integration Guide](../docs/neural-network-artemis-integration.md)
- [Integration Overview](../NEURAL_NETWORK_INTEGRATION.md)
