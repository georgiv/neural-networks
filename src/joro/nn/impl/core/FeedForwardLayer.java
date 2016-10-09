package joro.nn.impl.core;

import java.util.stream.Stream;

import joro.nn.api.Layer;
import joro.nn.api.Neuron;

public final class FeedForwardLayer implements Layer {
  private Neuron[] neurons;

  @Override
  public void adjust(Neuron... neurons) {
    this.neurons = neurons;
  }

  @Override
  public double[] activate(double... input) {
    return Stream.of(neurons).mapToDouble(neuron -> neuron.applyTransferFunction(input)).toArray();
  }
}
