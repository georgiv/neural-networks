package joro.nn.impl.core;

import java.util.Arrays;

import joro.nn.api.Delay;
import joro.nn.api.Layer;
import joro.nn.api.LearningRule;
import joro.nn.api.Network;

public abstract class SingleLayerNetwork implements Network {
  protected Layer ffLayer;
  protected LearningRule learningRule;
  protected Delay delay;

  public SingleLayerNetwork() {
    ffLayer = new FeedForwardLayer();
  }

  @Override
  public final void learn() {
    learningRule.converge();
  }

  @Override
  public final double[] process(double... inputs) {
    if (inputs.length < 1) {
      throw new IllegalArgumentException("There should be at least one input value.");
    }

    if (delay != null) {
      if (inputs.length != 1) {
        throw new IllegalArgumentException("When using a delay component, the neural network can accept only one value at a time.\n" + 
                                           "Inputs: " + Arrays.toString(inputs));
      }

      inputs = delay.delayInput(inputs[0]);
    }
    return ffLayer.activate(inputs);
  }
}
