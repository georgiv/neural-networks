package joro.nn.impl.learningrules.backpropagation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Stream;

import joro.nn.api.Layer;
import joro.nn.api.LearningRule;
import joro.nn.impl.core.BasicNeuron;
import joro.nn.impl.core.DerivativeFactory;
import joro.nn.impl.core.Feed;
import joro.nn.impl.core.TransferFunctionFactory;
import joro.nn.impl.core.TransferFunctionType;
import joro.nn.impl.utils.Calculator;

/**
 * Levenberg-Marquardt backpropagation
 */
public final class LMBP implements LearningRule {
  private Layer[] ffLayers;
  private List<Feed> calibrationFeed;
  private BasicNeuron[][] neurons;
  private double[][] inputs;
  private double[][] targets;
  private TransferFunctionType[] transferFunctions;

  public LMBP(Layer[] ffLayers, List<Feed> calibrationFeed) {
    this.ffLayers = ffLayers;
    this.calibrationFeed = calibrationFeed;
  }

  @Override
  public void converge() {
    initNeurons();

    int stepsCounter = 0;
    int successfulStepsCounter = 0;
    while (true) {
      stepsCounter++;

      double[][][] allOutputs = new double[calibrationFeed.size()][][];
      double[][] errors = new double[calibrationFeed.size()][];
      double[][][][] sensitivities = new double[calibrationFeed.size()][][][];

      for (int i = 0; i < calibrationFeed.size(); i++) {
        double[][] layersOutput = new double[ffLayers.length][];
        double[] networkOutput = ffLayers[0].activate(calibrationFeed.get(i).getInputs());
        double[][][] currentSensitivities = new double[ffLayers.length][][];

        // Calculate all outputs from all layers for each input and errors
        layersOutput[0] = negateTransferFunction(networkOutput, transferFunctions[0]);
        for (int j = 1; j < ffLayers.length; j++) {
          networkOutput = ffLayers[j].activate(networkOutput);
          layersOutput[j] = negateTransferFunction(networkOutput, transferFunctions[j]);
        }
        allOutputs[i] = layersOutput;
        errors[i] = Calculator.subtractDoubleArrays(calibrationFeed.get(i).getOutputs(), networkOutput);

        // Calculating last layer sensitivities
        DoubleUnaryOperator derivative = DerivativeFactory.getInstance().getDerivative(transferFunctions[transferFunctions.length - 1]);
        double[] layerOutput = layersOutput[layersOutput.length - 1];
        double[][] multiplier = new double[layerOutput.length][layerOutput.length];
        for (int j = 0; j < layerOutput.length; j++) {
          multiplier[j][j] = Calculator.roundDouble(derivative.applyAsDouble(layerOutput[j]));
        }
        currentSensitivities[currentSensitivities.length - 1] = Calculator.multiplyMatrix(multiplier, -1);

        // Backpropagating the sensitivities through the hidden layers
        for (int j = currentSensitivities.length - 2; j >= 0; j--) {
          derivative = DerivativeFactory.getInstance().getDerivative(transferFunctions[j]);
          layerOutput = layersOutput[j];
          multiplier = new double[layerOutput.length][layerOutput.length];
          for (int k = 0; k < layerOutput.length; k++) {
            multiplier[k][k] = Calculator.roundDouble(derivative.applyAsDouble(layerOutput[k]));
          }
          double[][] weights = new double[neurons[j + 1].length][];
          for (int k = 0; k < neurons[j + 1].length; k++) {
            weights[k] = neurons[j + 1][k].getWeights();
          }
          double[][] previousLayerSensitivities = currentSensitivities[j + 1];
          currentSensitivities[j] = Calculator.multiplyMatrices(Calculator.multiplyMatrices(multiplier, Calculator.transposeMatrix(weights)), previousLayerSensitivities);
        }
        sensitivities[i] = currentSensitivities;
      }

      // Calculating the Jacobian matrix
      int variablesCounter = 0;
      for (int i = 0; i < neurons.length; i++) {
        for (int j = 0; j < neurons[i].length; j++) {
          variablesCounter += neurons[i][j].getWeights().length;
          variablesCounter++;
        }
      }
      double[][] jacobian = new double[sensitivities.length][variablesCounter];
      
    }
  }

  private void initNeurons() {
    inputs = new double[calibrationFeed.size()][];
    targets = new double[calibrationFeed.size()][];

    int inputCount = calibrationFeed.get(0).getInputs().length;
    int outputCount = calibrationFeed.get(0).getOutputs().length;

    for (int i = 0; i < calibrationFeed.size(); i++) {
      inputs[i] = calibrationFeed.get(i).getInputs();
      targets[i] = calibrationFeed.get(i).getOutputs();
    }

    neurons = new BasicNeuron[ffLayers.length][];

    transferFunctions = new TransferFunctionType[ffLayers.length];
    transferFunctions[0] = TransferFunctionType.LOG_SIGMOID;
    transferFunctions[1] = TransferFunctionType.LINEAR;

    DoubleUnaryOperator hiddenLayerTransferFunction = TransferFunctionFactory.getInstance().getTransferFunction(transferFunctions[0]);
    BasicNeuron[] hiddenLayerNeurons = Stream.generate(() -> new BasicNeuron(hiddenLayerTransferFunction)).limit(2).toArray(BasicNeuron[]::new);
    Stream.of(hiddenLayerNeurons).forEach(neuron -> { neuron.setBias(generateRandomWeight());
                                                      neuron.setWeights(generateRandomWeights(inputCount)); });
    neurons[0] = hiddenLayerNeurons;
    ffLayers[0].adjust(neurons[0]);

    BasicNeuron[] testhiddenLayerNeurons = new BasicNeuron[] {
      new BasicNeuron(n -> n * n)
    };
    testhiddenLayerNeurons[0].setWeights(new double[] { 1 });
    testhiddenLayerNeurons[0].setBias(0);
    neurons[0] = testhiddenLayerNeurons;
    ffLayers[0].adjust(testhiddenLayerNeurons);

    DoubleUnaryOperator outputLayerTransferFunction = TransferFunctionFactory.getInstance().getTransferFunction(transferFunctions[1]);
    BasicNeuron[] outputLayerNeurons = Stream.generate(() -> new BasicNeuron(outputLayerTransferFunction)).limit(outputCount).toArray(BasicNeuron[]::new);
    Stream.of(outputLayerNeurons).forEach(neuron -> { neuron.setBias(generateRandomWeight());
                                                      neuron.setWeights(generateRandomWeights(hiddenLayerNeurons.length)); });
    neurons[1] = outputLayerNeurons;
    ffLayers[1].adjust(neurons[1]);
    BasicNeuron[] testOutputLayerNeurons = new BasicNeuron[] {
        new BasicNeuron(outputLayerTransferFunction)
    };
    testOutputLayerNeurons[0].setWeights(new double[] { 2 });
    testOutputLayerNeurons[0].setBias(1);
    neurons[1] = testOutputLayerNeurons;
    ffLayers[1].adjust(testOutputLayerNeurons);
  }

  private double[] generateRandomWeights(int weightsCount) {
    double[] weights = new double[weightsCount];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = generateRandomWeight();
    }
    return weights;
  }

  private double generateRandomWeight() {
    return Calculator.roundDouble(ThreadLocalRandom.current().nextDouble(-0.9, 1), 2);
  }

  private double[] negateTransferFunction(double[] values, TransferFunctionType transferFunction) {
    double[] result = new double[values.length];
    DoubleUnaryOperator antiTransferFunction = null;
    switch (transferFunction) {
      case LINEAR: antiTransferFunction = i -> i;
        break;
      default: antiTransferFunction = i -> Math.sqrt(i);
        break;
    }
    for (int i = 0; i < result.length; i++) {
      result[i] = antiTransferFunction.applyAsDouble(values[i]);
    }
    return result;
  }
}
