package joro.nn.impl.core;

import joro.nn.api.Layer;
import joro.nn.api.LearningRule;
import joro.nn.api.Network;

public abstract class MultiLayerNetwork implements Network {
  protected Layer[] ffLayers;
  protected LearningRule learningRule;

  public MultiLayerNetwork() {
    ffLayers = new FeedForwardLayer[2];
    for (int i = 0; i < ffLayers.length; i++) {
      ffLayers[i] = new FeedForwardLayer();
    }
  }

  @Override
  public void learn() {
    learningRule.converge();
  }

  @Override
  public double[] process(double... inputs) {
    if (inputs.length < 1) {
      throw new IllegalArgumentException("There should be at least one input value.");
    }

    double[] result = ffLayers[0].activate(inputs);
    for (int i = 1; i < ffLayers.length; i++) {
      result = ffLayers[i].activate(result);
    }
    return result;
  }
}
