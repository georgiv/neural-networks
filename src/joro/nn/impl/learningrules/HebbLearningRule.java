package joro.nn.impl.learningrules;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import joro.nn.api.Layer;
import joro.nn.api.LearningRule;
import joro.nn.impl.core.BasicNeuron;
import joro.nn.impl.core.Feed;
import joro.nn.impl.core.NetworkType;
import joro.nn.impl.core.TransferFunctionType;
import joro.nn.impl.core.TransferFunctionFactory;
import joro.nn.impl.utils.Calculator;

public final class HebbLearningRule implements LearningRule {
  private Layer ffLayer;
  private List<Feed> calibrationFeed;
  private BasicNeuron[] neurons;
  private NetworkType networkType;
  private TransferFunctionType transferFunctionType;
  private double[][] inputs;
  private double[][] outputs;
  private boolean hasBias;
  private boolean isAutoassociator;

  public HebbLearningRule(Layer ffLayer, List<Feed> calibrationFeed, NetworkType networkType) {
    this.ffLayer = ffLayer;
    this.calibrationFeed = calibrationFeed;
    this.networkType = networkType;
  }

  @Override
  public void converge() {
    neurons = initNeurons();
    ffLayer.adjust(neurons);

    if (transferFunctionType == TransferFunctionType.HARD_LIMIT && isAutoassociator) {
      double[][] binaryWeights = calculateBinaryWeights(inputs, outputs);
      adjustNeurons(binaryWeights);

      // Test area
      for (int i = 0; i < neurons.length; i++) {
        System.out.println(neurons[i]);
      }
      // End of test area

      return;
    }

    boolean isHebbianRuleApplicable = false;
    if (transferFunctionType == TransferFunctionType.LINEAR) {
      isHebbianRuleApplicable = checkForOrthonormality(inputs);
    } else {
      isHebbianRuleApplicable = checkForOrthogonality(inputs);
    }

    double[][] weights = calculateWeights(inputs, outputs, isHebbianRuleApplicable);
    adjustNeurons(weights);

    // Test area
    for (int i = 0; i < neurons.length; i++) {
      System.out.println(neurons[i]);
    }
    // End of test area

  }

  private BasicNeuron[] initNeurons() {
    inputs = new double[calibrationFeed.size()][];
    outputs = new double[calibrationFeed.size()][];

    isAutoassociator = Arrays.equals(calibrationFeed.get(0).getInputs(), calibrationFeed.get(0).getOutputs());

    int inputsCount = calibrationFeed.get(0).getInputs().length;
    int outputsCount = calibrationFeed.get(0).getOutputs().length;

    boolean containsNegativeOne = false;
    boolean containsZero = false;
    boolean containsPositiveOne = false;
    boolean containsOthers = false;

    for (int i = 0; i < calibrationFeed.size(); i++) {
      inputs[i] = calibrationFeed.get(i).getInputs();
      outputs[i] = calibrationFeed.get(i).getOutputs();

      if (isAutoassociator) {
        isAutoassociator = Arrays.equals(inputs[i], outputs[i]);
      }

      if (!containsOthers) {
        if (!containsNegativeOne) {
          containsNegativeOne = DoubleStream.of(outputs[i]).anyMatch(value -> value == -1);
        }
        if (!containsZero) {
          containsZero = DoubleStream.of(outputs[i]).anyMatch(value -> value == 0);
        }
        if (!containsPositiveOne) {
          containsPositiveOne = DoubleStream.of(outputs[i]).anyMatch(value -> value == 1);
        }
        containsOthers = DoubleStream.of(outputs[i]).anyMatch(value -> value != -1 && value != 0 && value != 1);
      }

      if (networkType == NetworkType.PERCEPTION || isAutoassociator) {
        if (containsOthers || (containsNegativeOne && containsZero)) {
          throw new IllegalArgumentException("The target values for selected neural network should contain only:\n" + 
                                             " - 1 and 0 elements for hard limit transfer function (hardlim)\n" + 
                                             " - 1 and -1 for symetric hard limit transfer function (hardlims).");
        }
      }
    }

    if (networkType == NetworkType.PERCEPTION) {
      double[] bias = new double[inputs.length];
      Arrays.fill(bias, 1);
      inputs = addBias(inputs, bias);
      hasBias = true;
    }

    TransferFunctionType transferFunctionType = null;
    if (networkType == NetworkType.PERCEPTION) {
      transferFunctionType = containsZero ? TransferFunctionType.HARD_LIMIT : TransferFunctionType.SYMMETRICAL_HARD_LIMIT;
    } else if (networkType == NetworkType.LINEAR_ASSOCIATOR) {
      if (isAutoassociator) {
        transferFunctionType = containsZero ? TransferFunctionType.HARD_LIMIT : TransferFunctionType.SYMMETRICAL_HARD_LIMIT;
      } else {
        transferFunctionType = TransferFunctionType.LINEAR;
      }
    }
    this.transferFunctionType = transferFunctionType;
    DoubleUnaryOperator transferFunction = TransferFunctionFactory.getInstance().getTransferFunction(transferFunctionType);

    BasicNeuron[] neurons = Stream.generate(() -> new BasicNeuron(transferFunction)).limit(outputsCount).toArray(BasicNeuron[]::new);

    Stream.of(neurons).forEach(neuron -> { double[] weights = new double[inputsCount];
                                           Arrays.fill(weights, 0);
                                           neuron.setWeights(weights); });
    return neurons;
  }

  private double[][] addBias(double[][] weights, double[] bias) {
    double[][] inputsWithBias = new double[weights.length][];
    for (int i = 0; i < inputsWithBias.length; i++) {
      double[] inputWithBias = new double[weights[i].length + 1];
      System.arraycopy(weights[i], 0, inputWithBias, 0, weights[i].length);
      inputWithBias[inputWithBias.length - 1] = bias[i];
      inputsWithBias[i] = inputWithBias;
    }
    return inputsWithBias;
  }

  private double[][] calculateBinaryWeights(double[][] binaryInputs, double[][] binaryOutputs) {
    double[][] bipolarInputs = convertBinaryToBipolar(binaryInputs);
    double[][] bipolarOutputs = convertBinaryToBipolar(binaryOutputs);
    double[][] bipolarWeights = calculateWeights(bipolarInputs, bipolarOutputs, checkForOrthogonality(bipolarInputs));

    double[][] binaryWeights = Calculator.multiplyMatrix(bipolarWeights, 2);

    double[] binaryBias = new double[bipolarWeights.length];
    double[][] negativeBipolarWeights = Calculator.multiplyMatrix(bipolarWeights, -1);
    for (int i = 0; i < negativeBipolarWeights.length; i++) {
      binaryBias[i] = Calculator.roundDouble(DoubleStream.of(negativeBipolarWeights[i]).sum());
    }

    double[][] binaryWeightsWithBias = addBias(binaryWeights, binaryBias);
    hasBias = true;

    return binaryWeightsWithBias;
  }

  private double[][] convertBinaryToBipolar(double[][] binaryData) {
    double[][] bipolarData = new double[binaryData.length][];
    for (int i = 0; i < bipolarData.length; i++) {
      bipolarData[i] = new double[binaryData[i].length];
      for (int j = 0; j < bipolarData[i].length; j++) {
        if(binaryData[i][j] == 1) {
          bipolarData[i][j] = 1;
        } else {
          bipolarData[i][j] = -1;
        }
      }
    }
    return bipolarData;
  }

  private boolean checkForOrthogonality(double[][] inputs) {
    for (int i = 0; i < inputs.length - 1; i++) {
      for (int j = i + 1; j < inputs.length; j++) {
        double[] product = Calculator.multiplyDoubleArrays(inputs[i], inputs[j]);
        if (DoubleStream.of(product).sum() != 0) {
          return false;
        }
      }
    }

    return true;
  }

  private boolean checkForOrthonormality(double[][] inputs) {
    if (!checkForOrthogonality(inputs)) {
      return false;
    }

    for (int i = 0; i < inputs.length; i++) {
      if (Calculator.getVectorMagnitude(inputs[i]) != 1) {
        return false;
      }
    }

    return true;
  }

  private double[][] calculateWeights(double[][] inputs, double[][] outputs, boolean isHebbianRuleApplicable) {
    double[][] weights = null;
    if (isHebbianRuleApplicable) {
      weights = applyHebbianRule(inputs, outputs);
    } else {
      if (inputs.length == inputs[0].length) {
        weights = Calculator.multiplyMatrices(Calculator.transposeMatrix(outputs), Calculator.getInverseMatrix(inputs));
      } else {
        if (inputs.length < inputs[0].length) {
          weights = Calculator.multiplyMatrices(Calculator.transposeMatrix(outputs), Calculator.getLeftPseudoinverseMatrix(Calculator.transposeMatrix(inputs)));
        } else {
          weights = Calculator.multiplyMatrices(Calculator.transposeMatrix(outputs), Calculator.getRightPseudoinverseMatrix(Calculator.transposeMatrix(inputs)));
        }
      }
    }
    return weights;
  }

  private double[][] applyHebbianRule(double[][] inputs, double[][] outputs) {
    return Calculator.multiplyMatrices(Calculator.transposeMatrix(outputs), inputs);
  }

  private void adjustNeurons(double[][] weights) {
    if (hasBias) {
      for (int i = 0; i < weights.length; i++) {
        double[] weightsWithoutBias = new double[weights[i].length - 1];
        System.arraycopy(weights[i], 0, weightsWithoutBias, 0, weights[i].length - 1);
        neurons[i].setWeights(weightsWithoutBias);
        neurons[i].setBias(weights[i][weights[i].length - 1]);
      }
      return;
    }

    for (int i = 0; i < weights.length; i++) {
      neurons[i].setWeights(weights[i]);
    }
  }
}
