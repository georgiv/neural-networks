package joro.nn.impl.networks;

import java.util.List;

import joro.nn.api.Delay;
import joro.nn.impl.core.Feed;
import joro.nn.impl.core.SingleLayerNetwork;
import joro.nn.impl.core.TappedDelayLine;
import joro.nn.impl.learningrules.LMSLearningRule;

public final class AdaptiveFilter extends SingleLayerNetwork {
  private Delay delay;

  public AdaptiveFilter(List<Feed> calibrationFeed) {
    learningRule = new LMSLearningRule(ffLayer, calibrationFeed);
    delay = new TappedDelayLine(calibrationFeed.get(0).getInput().length);
    
  }

  @Override
  public double[] process(double... input) {
    double[] inputs = delay.delayInput(input[0]);
    return ffLayer.activate(inputs);
  }
}
