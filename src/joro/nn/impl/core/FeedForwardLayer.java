package joro.nn.impl.core;

import java.util.stream.Stream;

import joro.nn.api.Layer;
import joro.nn.api.Neuron;

public final class FeedForwardLayer implements Layer {
  private Neuron[] neurons;

  @Override
  public void adjust(Neuron... neurons) {
    if (neurons.length < 1) {
      throw new IllegalArgumentException("There should be at least one neuron in a layer.");
    }

    this.neurons = neurons;
  }

  @Override
  public double[] activate(double... inputs) {
    if (inputs.length < 1) {
      throw new IllegalArgumentException("There should be at least one input value.");
    }

    return Stream.of(neurons).mapToDouble(neuron -> neuron.applyTransferFunction(inputs)).toArray();
  }
}
