package joro.nn.impl.core;

import java.util.Arrays;

public final class Feed {
  private double[] inputs;
  private double[] outputs;

  public double[] getInputs() {
    return inputs;
  }

  public void setInputs(double[] inputs) {
    if (inputs.length < 1) {
      throw new IllegalArgumentException("There should be at least one input value.");
    }

    this.inputs = inputs;
  }

  public double[] getOutputs() {
    return outputs;
  }

  public void setOutputs(double[] outputs) {
    if (outputs == null || outputs.length < 1) {
      throw new IllegalArgumentException("There should be at least one output value.");
    }

    this.outputs = outputs;
  }

  @Override
  public String toString() {
    return "Input: " + Arrays.toString(inputs) + " -> Output: " + Arrays.toString(outputs);
  }
}
