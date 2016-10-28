package joro.nn.impl.learningrules;

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

public final class BackpropagationLearningRule implements LearningRule {
  private Layer[] ffLayers;
  private List<Feed> calibrationFeed;
  private List<Feed> calibrationFeedCopy;
  private BasicNeuron[][] neurons;
  private double[][] inputs;
  private double[][] outputs;
  private TransferFunctionType[] transferFunctions;
  private double learningRate;
  //private double acceptableError;

  public BackpropagationLearningRule(Layer[] ffLayers, List<Feed> calibrationFeed) {
    this.ffLayers = ffLayers;
    this.calibrationFeed = calibrationFeed;
    calibrationFeedCopy = new ArrayList<>();
    Collections.addAll(calibrationFeedCopy, calibrationFeed.toArray(new Feed[0]));
  }

  @Override
  public void converge() {
    initNeurons();

    learningRate = 0.1;

    int stepsCounter = 0;
    while (true) {
      stepsCounter++;
      System.out.println("Step: " + stepsCounter);
      //calibrateForInput(pickCalibrationFeed());
      calibrateForInput(calibrationFeed.get(15));

      if (calibrationFeedCopy.size() == 0) {
        Collections.addAll(calibrationFeedCopy, calibrationFeed.toArray(new Feed[0]));
      }
    }
  }

  private void initNeurons() {
    inputs = new double[calibrationFeed.size()][];
    outputs = new double[calibrationFeed.size()][];

    int inputCount = calibrationFeed.get(0).getInputs().length;
    int outputCount = calibrationFeed.get(0).getOutputs().length;

    for (int i = 0; i < calibrationFeed.size(); i++) {
      inputs[i] = calibrationFeed.get(i).getInputs();
      outputs[i] = calibrationFeed.get(i).getOutputs();
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
       new BasicNeuron(hiddenLayerTransferFunction),
       new BasicNeuron(hiddenLayerTransferFunction)
    };
    testhiddenLayerNeurons[0].setWeights(new double[] { -0.27 });
    testhiddenLayerNeurons[0].setBias(-0.48);
    testhiddenLayerNeurons[1].setWeights(new double[] { -0.41 });
    testhiddenLayerNeurons[1].setBias(-0.13);
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
    testOutputLayerNeurons[0].setWeights(new double[] { 0.09, -0.17 });
    testOutputLayerNeurons[0].setBias(0.48);
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

  private Feed pickCalibrationFeed() {
    int index = ThreadLocalRandom.current().nextInt(0, calibrationFeedCopy.size());
    return calibrationFeedCopy.remove(index);
  }

  private boolean calibrateForInput(Feed calibrationFeed) {
    double[][] outputs = new double[ffLayers.length][];
    double[] output = ffLayers[0].activate(calibrationFeed.getInputs());
    outputs[0] = output;
    for (int i = 1; i < ffLayers.length; i++) {
      output = ffLayers[i].activate(output);
      outputs[i] = output;
    }
    if (Arrays.equals(calibrationFeed.getOutputs(), output)) {
      return true;
    }

    System.out.println("Expected output: " + Arrays.toString(calibrationFeed.getOutputs()));
    System.out.println("Actual output: " + Arrays.toString(output));

    double[] error = Calculator.subtractDoubleArrays(calibrationFeed.getOutputs(), output);
    double[][] sensitivities = calculateSensitives(error, outputs);
    System.out.println("Sensitivities:");
    for (int i = 0; i < sensitivities.length; i++) {
      System.out.println("Layer " + (i + 1) + ": " + Arrays.toString(sensitivities[i]));
    }

    for (int i = 0; i < sensitivities.length; i++) {
      
    }
    return true;
  }

  private double[][] calculateSensitives(double[] error, double[][] outputs) {
    double[][] sensitivities = new double[outputs.length][];

    // Calculating the entry point - the last layer sensitivities
    DoubleUnaryOperator derivative = DerivativeFactory.getInstance().getDerivative(transferFunctions[transferFunctions.length - 1]);
    double[][] jacobian = getJacobianMatrix(derivative, outputs[outputs.length - 1]);
    double[][] errorMatrix = new double[][] { error };
    double[][] lastLayerSensitive = Calculator.multiplyMatrices(Calculator.multiplyMatrix(jacobian, -2), errorMatrix);
    sensitivities[sensitivities.length - 1] = Calculator.transposeMatrix(lastLayerSensitive)[0];

    // Backpropagating the sensitivities through the hidden layers
    for (int i = sensitivities.length - 2; i > -1; i--) {
      derivative = DerivativeFactory.getInstance().getDerivative(transferFunctions[i]);
      jacobian = getJacobianMatrix(derivative, outputs[i]);
      double[][] weights = new double[neurons[i + 1].length][];
      for (int j = 0; j < neurons[i + 1].length; j++) {
        weights[j] = neurons[i + 1][j].getWeights();
      }
      double[][] previousLayerSensitivities = new double[][] { sensitivities [i + 1] };
      double[][] currentLayerSensitivities = Calculator.multiplyMatrices(Calculator.multiplyMatrices(getJacobianMatrix(derivative, outputs[i]), Calculator.transposeMatrix(weights)), previousLayerSensitivities);
      sensitivities[i] = Calculator.transposeMatrix(currentLayerSensitivities)[0];
    }

    return sensitivities;
  }

  private static double[][] getJacobianMatrix(DoubleUnaryOperator derivative, double[] output) {
    double[][] matrix = new double[output.length][output.length];
    for (int i = 0; i < output.length; i++) {
      matrix[i][i] = Calculator.roundDouble(derivative.applyAsDouble(output[i]));
    }
    return matrix;
  }

  public static void main(String[] args) {
    int size = 2;
    DoubleUnaryOperator derivative = DerivativeFactory.getInstance().getDerivative(TransferFunctionType.LOG_SIGMOID);
    double[] output = new double[] { 0.321, 0.368 };
    double[][] result = getJacobianMatrix(derivative, output);
    for (int i = 0; i < result.length; i++) {
      System.out.println(Arrays.toString(result[i]));
    }
    
  }
}
