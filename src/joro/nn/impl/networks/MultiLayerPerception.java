package joro.nn.impl.networks;

import java.util.List;

import joro.nn.impl.core.Feed;
import joro.nn.impl.core.MultiLayerNetwork;
import joro.nn.impl.learningrules.backpropagation.LMBP;

public final class MultiLayerPerception extends MultiLayerNetwork {
  public MultiLayerPerception(List<Feed> calibrationFeed) {
    learningRule = new LMBP(ffLayers, calibrationFeed);
  }
}
