package joro.nn.impl.core;

import joro.nn.api.Layer;
import joro.nn.api.LearningRule;
import joro.nn.api.Network;

public abstract class SingleLayerNetwork implements Network {
  protected Layer ffLayer;
  protected LearningRule learningRule;

  public SingleLayerNetwork() {
    ffLayer = new FeedForwardLayer();
  }

  @Override
  public final void learn() {
    learningRule.converge();
  }

  @Override
  public double[] process(double... input) {
    return ffLayer.activate(input);
  }
}
