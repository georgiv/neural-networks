package joro.nn.impl.networks;

import java.util.List;

import joro.nn.impl.core.Feed;
import joro.nn.impl.core.SingleLayerNetwork;
import joro.nn.impl.learningrules.LMSLearningRule;

public final class ADALINE extends SingleLayerNetwork {
  public ADALINE(List<Feed> calibrationFeed) {
    learningRule = new LMSLearningRule(ffLayer, calibrationFeed);
  }
}
