package joro.nn.impl.networks;

import java.util.List;

import joro.nn.impl.core.Feed;
import joro.nn.impl.core.SingleLayerNetwork;
import joro.nn.impl.core.TappedDelayLine;
import joro.nn.impl.learningrules.LMSLearningRule;

public final class AdaptiveFilter extends SingleLayerNetwork {
  public AdaptiveFilter(List<Feed> calibrationFeed) {
    learningRule = new LMSLearningRule(ffLayer, calibrationFeed);
    delay = new TappedDelayLine(calibrationFeed.get(0).getInputs().length);
  }
}
