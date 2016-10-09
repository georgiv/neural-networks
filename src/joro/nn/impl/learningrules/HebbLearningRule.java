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
  private BasicNeuron[] neurons;
  private List<Feed> calibrationFeed;
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

    if (isAutoassociator && transferFunctionType == TransferFunctionType.HARD_LIMIT) {
      double[][] result = convertBipolarToBinaryWeights();
      adjustNeurons(result);
      for (int i = 0; i < neurons.length; i++) {
        System.out.println(neurons[i]);
      }
      return;
    }

    boolean isHebbianRuleApplicable = false;
    if ((transferFunctionType != TransferFunctionType.HARD_LIMIT) && (transferFunctionType != TransferFunctionType.SYMMETRICAL_HARD_LIMIT)) {
      isHebbianRuleApplicable = checkForOrthonormality(inputs);
    } else {
      isHebbianRuleApplicable = checkForOrthogonality(inputs);
    }

    if (isHebbianRuleApplicable) {
      adjustNeurons(applyHebbianRule(inputs, outputs));
      for (int i = 0; i < neurons.length; i++) {
        System.out.println(neurons[i]);
      }
      return;
    }

    double[][] weights = null;
    if (inputs.length == inputs[0].length) {
      weights = Calculator.multiplyMatrices(Calculator.transposeMatrix(outputs), Calculator.getInverseMatrix(inputs));
    } else {
      if (inputs.length < inputs[0].length) {
        weights = Calculator.multiplyMatrices(Calculator.transposeMatrix(outputs), Calculator.getLeftPseudoinverseMatrix(Calculator.transposeMatrix(inputs)));
      } else {
        weights = Calculator.multiplyMatrices(Calculator.transposeMatrix(outputs), Calculator.getRightPseudoinverseMatrix(Calculator.transposeMatrix(inputs)));
      }
    }
    adjustNeurons(weights);

    for (int i = 0; i < neurons.length; i++) {
      System.out.println(neurons[i]);
    }
  }

  private BasicNeuron[] initNeurons() {
    if (calibrationFeed.size() < 1) {
      throw new IllegalArgumentException("The calibration tests cannot be zero. There should be at least one test for each output class.");
    }

    inputs = new double[calibrationFeed.size()][];
    outputs = new double[calibrationFeed.size()][];

    isAutoassociator = Arrays.equals(calibrationFeed.get(0).getInput(), calibrationFeed.get(0).getOutput());

    int inputCount = calibrationFeed.get(0).getInput().length;
    int outputCount = calibrationFeed.get(0).getOutput().length;

    boolean containsOutputNegativeOne = false;
    boolean containsOutputZero = false;
    boolean containsOutputPositiveOne = false;
    boolean containsOutputOthers = false;

    for (int i = 0; i < calibrationFeed.size(); i++) {
      double[] input = calibrationFeed.get(i).getInput();
      if (input.length != inputCount) {
        throw new IllegalArgumentException("All input feed entries should contain the same number of elements.\n" + 
                                           "The input feed entry on row " + i + " contains " + input.length + " elements, " + 
                                            "which differs from the previous input feed entries " + inputCount + ".");
      }
      inputs[i] = input;

      double[] output = calibrationFeed.get(i).getOutput();
      if (output.length != outputCount) {
        throw new IllegalArgumentException("All output feed entries should contain the same number of elements.\n" + 
                                           "The output feed entry on row " + i + " contains " + output.length + " elements, " + 
                                            "which differs from the previous output feed entries " + outputCount + ".");
      }
      outputs[i] = output;

      if (isAutoassociator) {
        isAutoassociator = Arrays.equals(input, output);
      }

      if (!containsOutputOthers) {
        if (!containsOutputNegativeOne) {
          containsOutputNegativeOne = DoubleStream.of(output).anyMatch(value -> value == -1);
        }
        if (!containsOutputZero) {
          containsOutputZero = DoubleStream.of(output).anyMatch(value -> value == 0);
        }
        if (!containsOutputPositiveOne) {
          containsOutputPositiveOne = DoubleStream.of(output).anyMatch(value -> value == 1);
        }
        containsOutputOthers = DoubleStream.of(output).anyMatch(value -> value != -1 && value != 0 && value != 1);
      }

      if (networkType == NetworkType.PERCEPTION) {
        if (containsOutputOthers || (containsOutputNegativeOne && containsOutputZero)) {
          throw new IllegalArgumentException("The output values should contain only:\n" + 
                                             " - 0 and 1 for hard limit transfer function (hardlim)\n" + 
                                             " - -1 and 1 for symetric hard limit transfer function (hardlims).");
        }
      }
    }

    TransferFunctionType transferFunctionType = null;

    if (networkType == NetworkType.PERCEPTION) {
      hasBias = true;
      addBias(); // TODO: Wrong place, cause misbehaviour

      transferFunctionType = containsOutputZero ? TransferFunctionType.HARD_LIMIT : TransferFunctionType.SYMMETRICAL_HARD_LIMIT;
    } else if (networkType == NetworkType.LINEAR_ASSOCIATOR) {
//      transferFunctionType = containsOthers || (containsNegativeOne && containsZero && containsPositiveOne) ? 
//          TransferFunctionType.LINEAR : 
//          containsZero ? 
//              TransferFunctionType.HARD_LIMIT : 
//              TransferFunctionType.SYMMETRICAL_HARD_LIMIT;
      if (isAutoassociator) {
        transferFunctionType = containsOutputZero ? TransferFunctionType.HARD_LIMIT : TransferFunctionType.SYMMETRICAL_HARD_LIMIT;
      } else {
        transferFunctionType = TransferFunctionType.LINEAR;
      }
    }
    this.transferFunctionType = transferFunctionType;

    DoubleUnaryOperator transferFunction = TransferFunctionFactory.getInstance().getTransferFunction(transferFunctionType);
    BasicNeuron[] neurons = Stream.generate(() -> new BasicNeuron(transferFunction)).limit(outputs[0].length).toArray(BasicNeuron[]::new);
    Stream.of(neurons).forEach(neuron -> { double[] weights = new double[inputs[0].length];
                                           Arrays.fill(weights, 0);
                                           neuron.setWeights(weights); });
    return neurons;
  }

  private void adjustNeurons(double[][] weights) {
    if (hasBias) {
      for (int i = 0; i < weights.length; i++) {
        double[] weightsWithoutBias = new double[weights[i].length - 1];
        System.arraycopy(weights[i], 0, weightsWithoutBias, 0, weights[i].length - 1);
        neurons[i].setWeights(weightsWithoutBias);
        neurons[i].setBias(weights[i][weights[i].length - 1]);
      }
      ffLayer.adjust(neurons);
      return;
    }

    for (int i = 0; i < weights.length; i++) {
      neurons[i].setWeights(weights[i]);
    }
    ffLayer.adjust(neurons);
  }

  private double[][] applyHebbianRule(double[][] inputs, double[][] outputs) {
    return Calculator.multiplyMatrices(Calculator.transposeMatrix(outputs), inputs);
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

  private void addBias() {
    double[][] inputsWithBias = new double[inputs.length][];
    for (int i = 0; i < inputsWithBias.length; i++) {
      double[] inputWithBias = new double[inputs[i].length + 1];
      System.arraycopy(inputs[i], 0, inputWithBias, 0, inputs[i].length);
      inputWithBias[inputWithBias.length - 1] = 1;
      inputsWithBias[i] = inputWithBias;
    }
    inputs = inputsWithBias;
  }

  private double[][] convertBipolarToBinaryWeights() {
    double[][] bipolarInputs = new double[inputs.length][];
    for (int i = 0; i < bipolarInputs.length; i++) {
      bipolarInputs[i] = new double[inputs[i].length];
      for (int j = 0; j < bipolarInputs[i].length; j++) {
        if(inputs[i][j] == 1) {
          bipolarInputs[i][j] = 1;
        } else {
          bipolarInputs[i][j] = -1;
        }
      }
    }

    double[][] bipolarOutputs = new double[outputs.length][];
    for (int i = 0; i < bipolarOutputs.length; i++) {
      bipolarOutputs[i] = new double[outputs[i].length];
      for (int j = 0; j < bipolarOutputs[i].length; j++) {
        if(outputs[i][j] == 1) {
          bipolarOutputs[i][j] = 1;
        } else {
          bipolarOutputs[i][j] = -1;
        }
      }
    }

    double[][] bipolarWeights = null;
    if (checkForOrthogonality(bipolarInputs)) {
      bipolarWeights = applyHebbianRule(bipolarInputs, bipolarOutputs);
    } else {
      if (bipolarInputs.length == bipolarInputs[0].length) {
        bipolarWeights = Calculator.multiplyMatrices(Calculator.transposeMatrix(bipolarOutputs), Calculator.getInverseMatrix(bipolarInputs));
      } else {
        if (bipolarInputs.length < bipolarInputs[0].length) {
          bipolarWeights = Calculator.multiplyMatrices(Calculator.transposeMatrix(bipolarOutputs), Calculator.getLeftPseudoinverseMatrix(Calculator.transposeMatrix(bipolarInputs)));
        } else {
          bipolarWeights = Calculator.multiplyMatrices(Calculator.transposeMatrix(bipolarOutputs), Calculator.getRightPseudoinverseMatrix(Calculator.transposeMatrix(bipolarInputs)));
        }
      }
    }

    double[][] binaryWeights = Calculator.multiplyMatrix(bipolarWeights, 2);
    double[] binaryBias = new double[bipolarWeights.length];
    bipolarWeights = Calculator.multiplyMatrix(bipolarWeights, -1);
    for (int i = 0; i < bipolarWeights.length; i++) {
      binaryBias[i] = Calculator.roundDouble(DoubleStream.of(bipolarWeights[i]).sum());
    }

    double[][] result = new double[binaryWeights.length][];
    for (int i = 0; i < result.length; i++) {
      result[i] = new double[binaryWeights[i].length + 1];
      for (int j = 0; j < result[i].length - 1; j++) {
        result[i][j] = binaryWeights[i][j];
      }
      result[i][result[i].length - 1] = binaryBias[i];
    }

    hasBias = true;

    return result;
  }
}
