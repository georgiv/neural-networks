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
    this.weights = weights;
  }

  public double getBias() {
    return bias;
  }

  public void setBias(double bias) {
    this.bias = bias;
  }

  @Override
  public final double applyTransferFunction(double... input) {
    double result = transferFunction.applyAsDouble(produceNetInput(input));
    return Calculator.roundDouble(result);
  }

  @Override
  public String toString() {
    return "BasicNeuron -> weights: " + Arrays.toString(weights) + ", bias: " + bias;
  }

  private double produceNetInput(double... input) {
    double[] product = Calculator.multiplyDoubleArrays(input, weights);
    return Calculator.roundDouble(bias + DoubleStream.of(product).sum());
  }
}
