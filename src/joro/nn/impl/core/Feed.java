package joro.nn.impl.core;

import java.util.Arrays;

public final class Feed {
  private double[] input;
  private double[] output;

  public Feed(int inputCount, int outputCount) {
    if((inputCount <= 0) || (outputCount <= 0)) {
      throw new IllegalArgumentException("Both input and output should contain at least one element.\n" +  
                                         "Input count: " + inputCount + 
                                         "Output count: " + outputCount);
    }

    input = new double[inputCount];
    output = new double[outputCount];
  }

  public double[] getInput() {
    return input;
  }

  public void setInput(double[] input) {
    this.input = input;
  }

  public double[] getOutput() {
    return output;
  }

  public void setOutput(double[] output) {
    this.output = output;
  }

  @Override
  public String toString() {
    return "Input: " + Arrays.toString(input) + " -> Output: " + Arrays.toString(output);
  }
}
