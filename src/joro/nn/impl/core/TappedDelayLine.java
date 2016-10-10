package joro.nn.impl.core;

import joro.nn.api.Delay;

public final class TappedDelayLine implements Delay {
  private int timeSteps;
  private double[] inputs;

  public TappedDelayLine(int timeSteps) {
    if (timeSteps < 1) {
      throw new IllegalArgumentException("The time steps value for delay should be at least 1.\n" + 
                                         "Time steps: " + timeSteps);
    }

    this.timeSteps = timeSteps;
    inputs = new double[timeSteps];
  }

  @Override
  public double[] delayInput(double input) {
    double[] newInputs = new double[timeSteps];
    newInputs[0] = input;
    if (timeSteps > 1) {
      System.arraycopy(inputs, 0, newInputs, 1, newInputs.length - 1);
    }
    inputs = newInputs;
    return inputs;
  }
}
