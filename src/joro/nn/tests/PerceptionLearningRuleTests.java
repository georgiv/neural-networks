package joro.nn.tests;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import joro.nn.api.Network;
import joro.nn.impl.core.Feed;
import joro.nn.impl.networks.Perception;
import joro.nn.impl.utils.CalibrationFeedGenerator;

public class PerceptionLearningRuleTests {

  @Test
  public void testAll() {
    List<Feed> tests01 = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/01_and_problem.txt");
    testSingleProblem(tests01);

    System.out.println("-------------------------------------------");

    List<Feed> tests02 = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/02_fruits_problem.txt");
    testSingleProblem(tests02);

    System.out.println("-------------------------------------------");

    List<Feed> tests03 = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/03_unknown_problem.txt");
    testSingleProblem(tests03);

    System.out.println("-------------------------------------------");

    List<Feed> tests04 = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/04_unknown_problem.txt");
    testSingleProblem(tests04);

    System.out.println("-------------------------------------------");

    List<Feed> tests05 = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/05_unknown_problem.txt");
    testSingleProblem(tests05);

    System.out.println("-------------------------------------------");

    List<Feed> tests06 = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/06_unknown_problem.txt");
    testSingleProblem(tests06);

    System.out.println("-------------------------------------------");

    List<Feed> tests07 = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/07_unknown_problem.txt");
    testSingleProblem(tests07);

    System.out.println("-------------------------------------------");

    List<Feed> tests08 = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/08_unknown_problem.txt");
    testSingleProblem(tests08);

    System.out.println("-------------------------------------------");

    List<Feed> tests09 = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/09_unknown_problem.txt");
    testSingleProblem(tests09);
  }

  private void testSingleProblem(List<Feed> tests) {
    Network nn = new Perception(tests);
    nn.learn();

    tests.stream().forEach(test -> { double[] inputs = test.getInputs();
                                     double[] outputs = test.getOutputs();
                                     double[] result = nn.process(test.getInputs());
                                     assertTrue(Arrays.equals(outputs, result));
                                     System.out.println("inputs: " + Arrays.toString(inputs) + " --> outputs: " + Arrays.toString(outputs)); });
  }
}
