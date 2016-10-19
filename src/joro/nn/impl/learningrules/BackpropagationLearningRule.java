package joro.nn.impl.learningrules;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Stream;

import joro.nn.api.Layer;
import joro.nn.api.LearningRule;
import joro.nn.impl.core.BasicNeuron;
import joro.nn.impl.core.Feed;
import joro.nn.impl.core.TransferFunctionFactory;
import joro.nn.impl.core.TransferFunctionType;
import joro.nn.impl.utils.Calculator;

public final class BackpropagationLearningRule implements LearningRule {
  private Layer[] ffLayers;
  private List<Feed> calibrationFeed;
  private BasicNeuron[] neurons;
  private double[][] inputs;
  private double[][] outputs;
  //private double learningRate;
  //private double acceptableError;

  public BackpropagationLearningRule(Layer[] ffLayers, List<Feed> calibrationFeed) {
    this.ffLayers = ffLayers;
    this.calibrationFeed = calibrationFeed;
  }

  @Override
  public void converge() {
    initNeurons();
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

    DoubleUnaryOperator hiddenLayerTransferFunction = TransferFunctionFactory.getInstance().getTransferFunction(TransferFunctionType.LOG_SIGMOID);
    BasicNeuron[] hiddenLayerNeurons = Stream.generate(() -> new BasicNeuron(hiddenLayerTransferFunction)).limit(2).toArray(BasicNeuron[]::new);
    Stream.of(hiddenLayerNeurons).forEach(neuron -> { neuron.setBias(generateRandomWeight());
                                                      neuron.setWeights(generateRandomWeights(2)); });
    ffLayers[0].adjust(hiddenLayerNeurons);
    BasicNeuron[] testhiddenLayerNeurons = new BasicNeuron[] {
       new BasicNeuron(hiddenLayerTransferFunction),
       new BasicNeuron(hiddenLayerTransferFunction)
    };
    testhiddenLayerNeurons[0].setWeights(new double[] { -0.27 });
    testhiddenLayerNeurons[0].setBias(-0.48);
    testhiddenLayerNeurons[1].setWeights(new double[] { -0.41 });
    testhiddenLayerNeurons[1].setBias(-0.13);
    ffLayers[0].adjust(testhiddenLayerNeurons);

    DoubleUnaryOperator outputLayerTransferFunction = TransferFunctionFactory.getInstance().getTransferFunction(TransferFunctionType.LINEAR);
    BasicNeuron[] outputLayerNeurons = Stream.generate(() -> new BasicNeuron(outputLayerTransferFunction)).limit(outputCount).toArray(BasicNeuron[]::new);
    Stream.of(outputLayerNeurons).forEach(neuron -> { neuron.setBias(generateRandomWeight());
                                                      neuron.setWeights(generateRandomWeights(hiddenLayerNeurons.length)); });
    ffLayers[1].adjust(outputLayerNeurons);
    BasicNeuron[] testOutputLayerNeurons = new BasicNeuron[] {
        new BasicNeuron(outputLayerTransferFunction)
    };
    testOutputLayerNeurons[0].setWeights(new double[] { 0.09, -0.17 });
    testOutputLayerNeurons[0].setBias(0.48);
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

  private boolean calibrateForInput(Feed calibrationFeed) {
    double[] output = ffLayers[0].activate(calibrationFeed.getInputs());
    for (int i = 1; i < ffLayers.length; i++) {
      output = ffLayers[i].activate(output);
    }
    if (Arrays.equals(calibrationFeed.getOutputs(), output)) {
      return true;
    }

    System.out.println("Expected output: " + Arrays.toString(calibrationFeed.getOutputs()));
    System.out.println("Actual output: " + Arrays.toString(output));
    return false;
  }

  public static void main(String[] args) {
    double p = 2;
    double result = 1 + Math.sin(Math.PI * p / 4);
    System.out.println(result);
  }
}
