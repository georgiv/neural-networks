package joro.nn.impl.learningrules;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import Jama.Matrix;
import joro.nn.api.Layer;
import joro.nn.api.LearningRule;
import joro.nn.impl.core.BasicNeuron;
import joro.nn.impl.core.Feed;
import joro.nn.impl.core.TransferFunctionFactory;
import joro.nn.impl.core.TransferFunctionType;
import joro.nn.impl.utils.Calculator;

public final class LMSLearningRule implements LearningRule {
  public static final double INITIAL_WEIGHT = 0.7;
  public static final double INITIAL_BIAS = 1.0;

  private Layer ffLayer;
  private List<Feed> calibrationFeed;
  private BasicNeuron[] neurons;
  private double[][] inputs;
  private double[][] outputs;
  private double learningRate;
  private double acceptableError;

  public LMSLearningRule(Layer ffLayer, List<Feed> calibrationFeed) {
    this.ffLayer = ffLayer;
    this.calibrationFeed = calibrationFeed;
  }

  @Override
  public void converge() {
    neurons = initNeurons();
    ffLayer.adjust(neurons);

    try {
      double[][] weights = new double[neurons.length][];
      double[][] transposedOutputs = Calculator.transposeMatrix(outputs);
      for (int i = 0; i < transposedOutputs.length; i++) {
        weights[i] = calculateMinimumPoint(inputs, transposedOutputs[i]);
      }
      adjustNeurons(weights);
      for (int i = 0; i < neurons.length; i++) {
        System.out.println(neurons[i]);
      }
      return;
    } catch (IllegalArgumentException illegalArgEx) {
      System.err.println("There's no unique min point - the meaned square error is not applicable. Trying with the LMS algorithm...");
    }

    double maxStableLearningRate = Calculator.roundDouble(calculateMaxStableLearningRate(inputs), 2);
    double minLearningRate = Calculator.roundDouble(maxStableLearningRate / 10, 2);
    learningRate = maxStableLearningRate;
    acceptableError = 0;

    boolean isConverged = false;
    int successfulIterationsCounter = 0;
    int stepsCounter = 0;
    while (!isConverged) {
      for (int i = 0; i < calibrationFeed.size(); i++) {
        try {
          if (calibrateForInput(calibrationFeed.get(i))) {
            successfulIterationsCounter++;
          } else {
            successfulIterationsCounter = 0;
          }
          stepsCounter++;
          if (successfulIterationsCounter == calibrationFeed.size()) {
            isConverged = true;
            break;
          }
        } catch (IllegalArgumentException e) {
          System.out.println("IAE on step " + stepsCounter + " --> learning rate: " + learningRate + " , acceptable error: " + acceptableError + ". Error message: " + e.getMessage());
          stepsCounter++;
          successfulIterationsCounter = 0;
        }
      }

      if (learningRate <= minLearningRate) {
        acceptableError = Calculator.roundDouble(acceptableError + 0.001);
        learningRate = maxStableLearningRate;
        stepsCounter = 0;
        successfulIterationsCounter = 0;
        neurons = initNeurons();
        ffLayer.adjust(neurons);
        System.out.println("New acceptable error: " + acceptableError);
      }

      if (stepsCounter >= (calibrationFeed.size() * 1000)) {
        learningRate = Calculator.roundDouble(learningRate - 0.01);
        stepsCounter = 0;
        successfulIterationsCounter = 0;
        neurons = initNeurons();
        ffLayer.adjust(neurons);
        System.out.println("New learning rate: " + learningRate);
      }
    }
    System.out.println("Converted in: " + stepsCounter + " steps.");
    System.out.println("Learning rate: " + learningRate);
    System.out.println("Acceptable error: " + acceptableError);
    for (int i = 0; i < neurons.length; i++) {
      System.out.println(neurons[i]);
    }
  }

  private BasicNeuron[] initNeurons() {
    inputs = new double[calibrationFeed.size()][];
    outputs = new double[calibrationFeed.size()][];

    int inputCount = calibrationFeed.get(0).getInputs().length;
    int outputCount = calibrationFeed.get(0).getOutputs().length;

    for (int i = 0; i < calibrationFeed.size(); i++) {
      inputs[i] = calibrationFeed.get(i).getInputs();
      outputs[i] = calibrationFeed.get(i).getOutputs();
    }

    inputs = addBias();

    DoubleUnaryOperator transferFunction = TransferFunctionFactory.getInstance().getTransferFunction(TransferFunctionType.LINEAR);

    BasicNeuron[] neurons = Stream.generate(() -> new BasicNeuron(transferFunction)).limit(outputCount).toArray(BasicNeuron[]::new);

    Stream.of(neurons).forEach(neuron -> { neuron.setBias(INITIAL_BIAS);
                                           double[] weights = new double[inputCount];
                                           Arrays.fill(weights, INITIAL_WEIGHT);
                                           neuron.setWeights(weights); });
    return neurons;
  }

  private double[][] addBias() {
    double[][] inputsWithBias = new double[inputs.length][];
    for (int i = 0; i < inputsWithBias.length; i++) {
      double[] inputWithBias = new double[inputs[i].length + 1];
      System.arraycopy(inputs[i], 0, inputWithBias, 0, inputs[i].length);
      inputWithBias[inputWithBias.length - 1] = 1;
      inputsWithBias[i] = inputWithBias;
    }
    return inputsWithBias;
  }

  private double calculateNeuronTargetsSquare(double[] outputs) {
    double result = 0;
    double probability = Calculator.roundDouble((double) 1 / calibrationFeed.size());
    double[] targetsSquare = Calculator.multiplyDoubleArrays(outputs, outputs);
    for (double targetSquare : targetsSquare) {
      result += Calculator.roundDouble(probability * targetSquare);
    }
    return result;
  }

  private double[] calculateNeuronCrossCorrelation(double[] outputs) {
    double[] result = null;
    double probability = Calculator.roundDouble((double) 1 / calibrationFeed.size());
    for (int i = 0; i < outputs.length; i++) {
      double[] input = inputs[i];
      double output = outputs[i];
      if (result == null) {
        result = new double[input.length];
      }
      for (int j = 0; j < input.length; j++) {
        result[j] += Calculator.roundDouble(probability * input[j] * output);
      }
    }
    return result;
  }

  private double[][] calculateCorrelationMatrix(double[][] inputs) {
    double[][] result = null;
    double probability = Calculator.roundDouble((double) 1 / inputs.length);
    for (int i = 0; i < inputs.length; i++) {
      double[][] input = new double[][] { inputs[i] };
      double[][] product = Calculator.multiplyMatrix((Calculator.multiplyMatrices(Calculator.transposeMatrix(input), input)), probability);
      if (result == null) {
        result = product;
      } else {
        result = Calculator.addMatrices(result, product);
      }
    }

    return result;
  }

  private double[] calculateMatrixEigenvalues(double[][] matrix) {
    if(!Calculator.isSquareMatrix(matrix)) {
      throw new IllegalArgumentException("Eigenvalues can be calculated only for square matrix with minimum size 2.");
    }

    double[] eigenvalues = new Matrix(matrix).eig().getRealEigenvalues(); // call to external library Jama!!!
    for (int i = 0; i < eigenvalues.length; i++) {
      eigenvalues[i] = Calculator.roundDouble(eigenvalues[i]);
    }
    return eigenvalues;
  }

  private double[] calculateMinimumPoint(double[][] inputs, double[] outputs) {
    double[][] correlationMatrix = calculateCorrelationMatrix(inputs);
    double[][] correlationMatrixInverse = Calculator.getInverseMatrix(Calculator.transposeMatrix(correlationMatrix));
    double[] crossCorrelation = calculateNeuronCrossCorrelation(outputs);
    double[][] crossCorrelationMatrix = Calculator.transposeMatrix(new double[][] { crossCorrelation });
    return Calculator.transposeMatrix(Calculator.multiplyMatrices(correlationMatrixInverse, crossCorrelationMatrix))[0];
  }

  private double calculateMinimumMeanSquareError(double[][] inputs, double[] outputs) {
    System.out.println("Entering calculateMinimumMeanSquareError()...");
    System.out.println("The point of this method is calculating F(X) = c - 2 * x(transposed) * h + x(transposed) * R * x");

    double c = calculateNeuronTargetsSquare(outputs);
    System.out.println("Squared target value (c) = " + c);

    double[] x = calculateMinimumPoint(inputs, outputs);
    System.out.println("Minimum point: " + Arrays.toString(x));
    double[] h = calculateNeuronCrossCorrelation(outputs);
    System.out.println("Cross correlation: " + Arrays.toString(h));

    double addend1 = 0;
    if (x.length != h.length) {
      throw new RuntimeException("The minimum point vector and the cross correlation vector should be the same size");
    }
    for (int i = 0; i < h.length; i++) {
      addend1 += Calculator.roundDouble(2 * x[i] * h[i]);
    }
    addend1 = Calculator.roundDouble((-1) * addend1);
    System.out.println("-2 * x(transposed) * h = " + addend1);

    double[][] correlationMatrix = calculateCorrelationMatrix(inputs);
    System.out.println("Correlation matrix:");
    for (int i = 0; i < correlationMatrix.length; i++) {
      System.out.println(Arrays.toString(correlationMatrix[i]));
    }
    double[] product1 = Calculator.multiplyMatrices(new double[][] { x }, Calculator.transposeMatrix(correlationMatrix))[0];
    System.out.println("Product of x(transposed) and correlation matrix: " + Arrays.toString(product1));
    double[] product2 = Calculator.multiplyMatrices(new double[][] { product1 }, Calculator.transposeMatrix(new double[][] { x }))[0];
    System.out.println("Product of x(transposed), correlation matrix and x: " + Arrays.toString(product2));

    double addend2 = 0;
    for (int i = 0; i < product2.length; i++) {
      addend2 += product2[i];
    }
    System.out.println("x(transposed) * R * x = " + addend2);

    double result = Calculator.roundDouble(c + addend1 + addend2);
    System.out.println("Minimum squared error: " + result);
    return result;
  }

  private double calculateMaxStableLearningRate(double[][] inputs) {
    double[][] correlationMatrix = calculateCorrelationMatrix(inputs);
    double[] eigenvalues = calculateMatrixEigenvalues(correlationMatrix);
    return Calculator.roundDouble(1 / DoubleStream.of(eigenvalues).max().getAsDouble());
  }

  private void adjustNeurons(double[][] weights) {
    for (int i = 0; i < weights.length; i++) {
      double[] weightsWithoutBias = new double[weights[i].length - 1];
      System.arraycopy(weights[i], 0, weightsWithoutBias, 0, weights[i].length - 1);
      neurons[i].setWeights(weightsWithoutBias);
      neurons[i].setBias(weights[i][weights[i].length - 1]);
    }
  }

  private boolean calibrateForInput(Feed calibrationFeed) {
    double[] layerOutput = ffLayer.activate(calibrationFeed.getInputs());
    if (Arrays.equals(calibrationFeed.getOutputs(), layerOutput)) {
      return true;
    }

    for (int i = 0; i < calibrationFeed.getOutputs().length; i++) {
      double error = Calculator.roundDouble(calibrationFeed.getOutputs()[i] - layerOutput[i]);
      //System.out.println(error);

      if (Math.abs(error) <= acceptableError) {
        return true;
      }

      if (error != 0) {
        BasicNeuron neuron = neurons[i];
        double[] newWeights = Calculator.addMatrices(new double[][] { neuron.getWeights() }, Calculator.multiplyMatrix(new double[][] { calibrationFeed.getInputs() }, Calculator.roundDouble(2 * learningRate * error)))[0];
        neuron.setWeights(newWeights);
        double newBias = Calculator.roundDouble(neuron.getBias() + Calculator.roundDouble(2 * learningRate * error));
        neuron.setBias(newBias);
        ffLayer.adjust(neurons);
      }
    }
    return false;
  }
}
