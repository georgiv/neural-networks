package joro.nn.impl.networks;

import java.util.List;

import joro.nn.impl.core.Feed;
import joro.nn.impl.core.LearningRuleType;
import joro.nn.impl.core.NetworkType;
import joro.nn.impl.core.SingleLayerNetwork;
import joro.nn.impl.learningrules.HebbLearningRule;
import joro.nn.impl.learningrules.PerceptionLearningRule;

public final class Perception extends SingleLayerNetwork {
  public Perception(List<Feed> calibrationFeed) {
    this(calibrationFeed, LearningRuleType.PERCEPTION);
  }

  public Perception(List<Feed> calibrationFeed, LearningRuleType learningRuleType) {
    if (learningRuleType == LearningRuleType.PERCEPTION) {
      learningRule = new PerceptionLearningRule(ffLayer, calibrationFeed);
    } else if (learningRuleType == LearningRuleType.HEBB) {
      learningRule = new HebbLearningRule(ffLayer, calibrationFeed, NetworkType.PERCEPTION);
    }
  }
}
