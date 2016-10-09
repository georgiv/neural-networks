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
import joro.nn.impl.core.TransferFunctionType;
import joro.nn.impl.core.TransferFunctionFactory;
import joro.nn.impl.utils.Calculator;

public final class PerceptionLearningRule implements LearningRule {
  public static final double INITIAL_WEIGHT = 0.7;
  public static final double INITIAL_BIAS = 1.0;

  private Layer ffLayer;
  private BasicNeuron[] neurons;
  private List<Feed> calibrationFeed;

  public PerceptionLearningRule(Layer ffLayer, List<Feed> calibrationFeed) {
    this.ffLayer = ffLayer;
    this.calibrationFeed = calibrationFeed;
  }

  @Override
  public void converge() {
    neurons = initNeurons();
    ffLayer.adjust(neurons);

    boolean isConverged = false;
    int successfulIterationsCounter = 0;
    int stepsCounter = 0;
    while (!isConverged) {
      for (int i = 0; i < calibrationFeed.size(); i++) {
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
      }
    }
    System.out.println("Converted in: " + stepsCounter + " steps.");
    for (int i = 0; i < neurons.length; i++) {
      System.out.println(neurons[i]);
    }
  }

  private BasicNeuron[] initNeurons() {
    if (calibrationFeed.size() < 1) {
      throw new IllegalArgumentException("The calibration tests cannot be zero. There should be at least one test for each output class.");
    }

    int inputCount = calibrationFeed.get(0).getInput().length;
    int outputCount = calibrationFeed.get(0).getOutput().length;

    boolean containsNegative = false;
    boolean containsZero = false;

    for (int i = 0; i < calibrationFeed.size(); i++) {
      double[] input = calibrationFeed.get(i).getInput();
      if (input.length != inputCount) {
        throw new IllegalArgumentException("All input feed entries should contain the same number of elements.\n" + 
                                           "The input feed entry on row " + i + " contains " + input.length + " elements, " + 
                                            "which differs from the previous input feed entries " + inputCount + ".");
      }

      double[] output = calibrationFeed.get(i).getOutput();
      if (output.length != outputCount) {
        throw new IllegalArgumentException("All output feed entries should contain the same number of elements.\n" + 
                                           "The output feed entry on row " + i + " contains " + output.length + " elements, " + 
                                            "which differs from the previous output feed entries " + outputCount + ".");
      }

      if (!containsNegative) {
        containsNegative = DoubleStream.of(output).anyMatch(value -> value == -1);
      }
      if (!containsZero) {
        containsZero = DoubleStream.of(output).anyMatch(value -> value == 0);
      }

      if (containsNegative && containsZero) {
        throw new IllegalArgumentException("The output values cannot contain both -1 and 0.\n" + 
                                           "They should contain only:\n" + 
                                           " - 0 and 1 for hard limit transfer function (hardlim)\n" + 
                                           " - -1 and 1 for symetric hard limit transfer function (hardlims).");
      }
    }

    DoubleUnaryOperator transferFunction = containsZero ? 
        TransferFunctionFactory.getInstance().getTransferFunction(TransferFunctionType.HARD_LIMIT) : 
        TransferFunctionFactory.getInstance().getTransferFunction(TransferFunctionType.SYMMETRICAL_HARD_LIMIT);
    BasicNeuron[] neurons = Stream.generate(() -> new BasicNeuron(transferFunction)).limit(outputCount).toArray(BasicNeuron[]::new);

    Stream.of(neurons).forEach(neuron -> { neuron.setBias(INITIAL_BIAS);
                                           double[] weights = new double[inputCount];
                                           Arrays.fill(weights, INITIAL_WEIGHT);
                                           neuron.setWeights(weights); });
    return neurons;
  }

  private boolean calibrateForInput(Feed calibrationFeed) {
    double[] layerOutput = ffLayer.activate(calibrationFeed.getInput());
    if (Arrays.equals(calibrationFeed.getOutput(), layerOutput)) {
      return true;
    }

    for (int i = 0; i < calibrationFeed.getOutput().length; i++) {
      double error = calibrationFeed.getOutput()[i] - layerOutput[i];
      if (error != 0) {
        BasicNeuron neuron = neurons[i];
        neuron.setBias(neuron.getBias() + error);
        if (error > 0) {
          neuron.setWeights(Calculator.addDoubleArrays(neuron.getWeights(), calibrationFeed.getInput()));
        } else {
          neuron.setWeights(Calculator.subtractDoubleArrays(neuron.getWeights(), calibrationFeed.getInput()));
        }
      }
    }
    return false;
  }
}
