package joro.nn.impl.core;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;

import joro.nn.api.Neuron;
import joro.nn.impl.utils.Calculator;

public final class BasicNeuron implements Neuron {
  private double[] weights;
  private double bias;
  private DoubleUnaryOperator transferFunction;

  public BasicNeuron(DoubleUnaryOperator transferFunction) {
    this.transferFunction = transferFunction;
  }

  public double[] getWeights() {
    return weights;
  }

  public void setWeights(double[] weights) {
    if (weights.length < 1) {
      throw new IllegalArgumentException("There should be at least one weight value.");
    }

    this.weights = weights;
  }

  public double getBias() {
    return bias;
  }

  public void setBias(double bias) {
    this.bias = bias;
  }

  @Override
  public final double applyTransferFunction(double... inputs) {
    if (inputs.length != weights.length) {
      throw new IllegalArgumentException("The input values count should be the same as the weight values count.\n" + 
                                         "Inputs: " + Arrays.toString(inputs) + "\n" + 
                                         "Weights: " + Arrays.toString(weights));
    }

    double result = transferFunction.applyAsDouble(produceNetInput(inputs));
    return Calculator.roundDouble(result);
  }

  @Override
  public String toString() {
    return "BasicNeuron -> weights: " + Arrays.toString(weights) + ", bias: " + bias;
  }

  private double produceNetInput(double... inputs) {
    double[] product = Calculator.multiplyDoubleArrays(inputs, weights);
    return Calculator.roundDouble(bias + DoubleStream.of(product).sum());
  }
}
