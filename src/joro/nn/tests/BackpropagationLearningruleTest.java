package joro.nn.tests;

import java.util.List;

import joro.nn.api.Network;
import joro.nn.impl.core.Feed;
import joro.nn.impl.networks.MultiLayerPerception;
import joro.nn.impl.utils.CalibrationFeedGenerator;

public class BackpropagationLearningruleTest {
  public static void main(String[] args) {
    List<Feed> tests_unknown01_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/backpropagation/01_unknown_problem.txt");
    Network myNN_unknown01_problem = new MultiLayerPerception(tests_unknown01_problem);
    myNN_unknown01_problem.learn();
  }
}
