package joro.nn.impl.core;

import joro.nn.api.Layer;
import joro.nn.api.LearningRule;
import joro.nn.api.Network;

public abstract class MultiLayerNetwork implements Network {
  protected Layer[] ffLayers;
  protected LearningRule learningRule;

  @Override
  public void learn() {
    // TODO Auto-generated method stub
    
  }

  @Override
  public double[] process(double... input) {
    // TODO Auto-generated method stub
    return null;
  }
}
