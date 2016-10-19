package joro.nn.impl.networks;

import java.util.List;

import joro.nn.impl.core.Feed;
import joro.nn.impl.core.MultiLayerNetwork;
import joro.nn.impl.learningrules.BackpropagationLearningRule;

public final class MultiLayerPerception extends MultiLayerNetwork {
  public MultiLayerPerception(List<Feed> calibrationFeed) {
    learningRule = new BackpropagationLearningRule(ffLayers, calibrationFeed);
  }
}
