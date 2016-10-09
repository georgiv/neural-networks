package joro.nn.impl.networks;

import java.util.List;

import joro.nn.impl.core.Feed;
import joro.nn.impl.core.NetworkType;
import joro.nn.impl.core.SingleLayerNetwork;
import joro.nn.impl.learningrules.HebbLearningRule;

public final class LinearAssociator extends SingleLayerNetwork {
  public LinearAssociator(List<Feed> calibrationFeed) {
    learningRule = new HebbLearningRule(ffLayer, calibrationFeed, NetworkType.LINEAR_ASSOCIATOR);
  }
}
