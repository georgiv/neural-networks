package joro.nn.tests;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import joro.nn.api.Network;
import joro.nn.impl.core.Feed;
import joro.nn.impl.networks.Perception;
import joro.nn.impl.utils.CalibrationFeedGenerator;

public class PerceptionLearningRuleTest {

  @Test
  public void testAll() {
    testSingleProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/01_and_problem.txt"));
    System.out.println("-------------------------------------------");
    testSingleProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/02_fruits_problem.txt"));
    System.out.println("-------------------------------------------");
    testSingleProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/03_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSingleProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/04_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSingleProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/05_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSingleProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/06_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSingleProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/07_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSingleProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/08_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSingleProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/09_unknown_problem.txt"));
  }

  private void testSingleProblem(List<Feed> tests) {
    Network nn = new Perception(tests);
    nn.learn();

    tests.stream().forEach(test -> { double[] inputs = test.getInputs();
                                     double[] outputs = test.getOutputs();
                                     double[] result = nn.process(inputs);
                                     assertTrue(Arrays.equals(outputs, result));
                                     System.out.println("inputs: " + Arrays.toString(inputs) + " --> outputs: " + Arrays.toString(outputs)); });
  }
}
