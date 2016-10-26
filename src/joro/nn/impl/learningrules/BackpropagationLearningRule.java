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
  private BasicNeuron[] neurons;
  private double[][] inputs;
  private double[][] outputs;
  private TransferFunctionType[] transferFunctions;
  //private double learningRate;
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
//    Feed calibrationFeed = new Feed();
//    calibrationFeed.setInputs(new double[] { 1 });
//    calibrationFeed.setOutputs(new double[] { 1.7071 });
//    calibrateForInput(calibrationFeed);

    int stepsCounter = 0;
    while (true) {
      stepsCounter++;
      System.out.println("Step: " + stepsCounter);
      calibrateForInput(pickCalibrationFeed());

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

    transferFunctions = new TransferFunctionType[ffLayers.length];
    transferFunctions[0] = TransferFunctionType.LOG_SIGMOID;
    transferFunctions[1] = TransferFunctionType.LINEAR;

    DoubleUnaryOperator hiddenLayerTransferFunction = TransferFunctionFactory.getInstance().getTransferFunction(transferFunctions[0]);
    BasicNeuron[] hiddenLayerNeurons = Stream.generate(() -> new BasicNeuron(hiddenLayerTransferFunction)).limit(2).toArray(BasicNeuron[]::new);
    Stream.of(hiddenLayerNeurons).forEach(neuron -> { neuron.setBias(generateRandomWeight());
                                                      neuron.setWeights(generateRandomWeights(inputCount)); });
    ffLayers[0].adjust(hiddenLayerNeurons);
//    BasicNeuron[] testhiddenLayerNeurons = new BasicNeuron[] {
//       new BasicNeuron(hiddenLayerTransferFunction),
//       new BasicNeuron(hiddenLayerTransferFunction)
//    };
//    testhiddenLayerNeurons[0].setWeights(new double[] { -0.27 });
//    testhiddenLayerNeurons[0].setBias(-0.48);
//    testhiddenLayerNeurons[1].setWeights(new double[] { -0.41 });
//    testhiddenLayerNeurons[1].setBias(-0.13);
//    ffLayers[0].adjust(testhiddenLayerNeurons);

    DoubleUnaryOperator outputLayerTransferFunction = TransferFunctionFactory.getInstance().getTransferFunction(transferFunctions[1]);
    BasicNeuron[] outputLayerNeurons = Stream.generate(() -> new BasicNeuron(outputLayerTransferFunction)).limit(outputCount).toArray(BasicNeuron[]::new);
    Stream.of(outputLayerNeurons).forEach(neuron -> { neuron.setBias(generateRandomWeight());
                                                      neuron.setWeights(generateRandomWeights(hiddenLayerNeurons.length)); });
    ffLayers[1].adjust(outputLayerNeurons);
//    BasicNeuron[] testOutputLayerNeurons = new BasicNeuron[] {
//        new BasicNeuron(outputLayerTransferFunction)
//    };
//    testOutputLayerNeurons[0].setWeights(new double[] { 0.09, -0.17 });
//    testOutputLayerNeurons[0].setBias(0.48);
//    ffLayers[1].adjust(testOutputLayerNeurons);
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

    for (int i = 0; i < calibrationFeed.getOutputs().length; i++) {
      double error = calibrationFeed.getOutputs()[i] - output[i];
      if (error != 0) {
        System.out.println("Error: " + (calibrationFeed.getOutputs()[i] - output[i]));
      }
    }
    return false;
  }

  private double[] calculateSensitives(double[][] outputs) {
    double[][] sensitivities = new double[outputs.length][];

    

    

    return null;
  }

  private double[] calculateLastLayerSensitivity(double error) {
    double[] result = new double[ffLayers.length];

    DoubleUnaryOperator derivative = DerivativeFactory.getInstance().getDerivative(transferFunctions[transferFunctions.length - 1]);

    int neuronsCount = outputs[0].length;

    double lastLayerSensitivity = -2 * derivative.applyAsDouble(0) * error;

    return null;
  }

  private static double[][] getJacobianMatrix(int size, DoubleUnaryOperator derivative, double[] output) {
    double[][] matrix = new double[size][size];
    for (int i = 0; i < size; i++) {
      matrix[i][i] = Calculator.roundDouble(derivative.applyAsDouble(output[i]));
    }
    return matrix;
  }

  public static void main(String[] args) {
    int size = 2;
    DoubleUnaryOperator derivative = DerivativeFactory.getInstance().getDerivative(TransferFunctionType.LOG_SIGMOID);
    double[] output = new double[] { 0.321, 0.368 };
    double[][] result = getJacobianMatrix(size, derivative, output);
    for (int i = 0; i < result.length; i++) {
      System.out.println(Arrays.toString(result[i]));
    }
    
  }
}
