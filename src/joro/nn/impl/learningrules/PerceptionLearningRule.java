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
  private List<Feed> calibrationFeed;
  private BasicNeuron[] neurons;

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

    // Test area
    System.out.println("Converted in: " + stepsCounter + " steps.");
    for (int i = 0; i < neurons.length; i++) {
      System.out.println(neurons[i]);
    }
    // End of test are

  }

  private BasicNeuron[] initNeurons() {
    int inputsCount = calibrationFeed.get(0).getInputs().length;
    int outputsCount = calibrationFeed.get(0).getOutputs().length;

    boolean containsNegative = false;
    boolean containsZero = false;

    for (int i = 0; i < calibrationFeed.size(); i++) {
      double[] output = calibrationFeed.get(i).getOutputs();
      if (!containsNegative) {
        containsNegative = DoubleStream.of(output).anyMatch(value -> value == -1);
      }
      if (!containsZero) {
        containsZero = DoubleStream.of(output).anyMatch(value -> value == 0);
      }

      if (containsNegative && containsZero) {
        throw new IllegalArgumentException("The output values cannot have both -1 and 0 elements.\n" + 
                                           "They should contain only:\n" + 
                                           " - 1 and 0 elements for hard limit transfer function (hardlim)\n" + 
                                           " - 1 and -1 for symetric hard limit transfer function (hardlims).");
      }
    }

    DoubleUnaryOperator transferFunction = containsZero ? 
        TransferFunctionFactory.getInstance().getTransferFunction(TransferFunctionType.HARD_LIMIT) : 
        TransferFunctionFactory.getInstance().getTransferFunction(TransferFunctionType.SYMMETRICAL_HARD_LIMIT);

    BasicNeuron[] neurons = Stream.generate(() -> new BasicNeuron(transferFunction)).limit(outputsCount).toArray(BasicNeuron[]::new);

    Stream.of(neurons).forEach(neuron -> { neuron.setBias(INITIAL_BIAS);
                                           double[] weights = new double[inputsCount];
                                           Arrays.fill(weights, INITIAL_WEIGHT);
                                           neuron.setWeights(weights); });
    return neurons;
  }

  private boolean calibrateForInput(Feed calibrationFeed) {
    double[] layerOutput = ffLayer.activate(calibrationFeed.getInputs());
    if (Arrays.equals(calibrationFeed.getOutputs(), layerOutput)) {
      return true;
    }

    for (int i = 0; i < calibrationFeed.getOutputs().length; i++) {
      double error = calibrationFeed.getOutputs()[i] - layerOutput[i];
      if (error != 0) {
        BasicNeuron neuron = neurons[i];
        neuron.setBias(neuron.getBias() + error);
        if (error > 0) {
          neuron.setWeights(Calculator.addDoubleArrays(neuron.getWeights(), calibrationFeed.getInputs()));
        } else {
          neuron.setWeights(Calculator.subtractDoubleArrays(neuron.getWeights(), calibrationFeed.getInputs()));
        }
      }
    }
    return false;
  }
}
