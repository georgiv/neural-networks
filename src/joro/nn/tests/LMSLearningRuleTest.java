package joro.nn.tests;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import joro.nn.api.Network;
import joro.nn.impl.core.Feed;
import joro.nn.impl.networks.ADALINE;
import joro.nn.impl.networks.AdaptiveFilter;
import joro.nn.impl.utils.CalibrationFeedGenerator;

public class LMSLearningRuleTest {

  @Test
  public void testAll() {
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/01_fruits_problem.txt"), 0.001);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/02_unknown_problem.txt"), 0.67);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/03_unknown_problem.txt"), 0);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/04_unknown_problem.txt"), 0.67);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/05_letters_problem.txt"), 0.001);
    System.out.println("-------------------------------------------");
    testSingleAdaptiveFilterProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/06_unknown_problem.txt"), 0.03);
    System.out.println("-------------------------------------------");
    testSingleAdaptiveFilterProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/07_unknown_problem.txt"), 0.001);
    System.out.println("-------------------------------------------");
    testSingleAdaptiveFilterProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/07_unknown_problem.txt"), 0.001);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/08_lines_problem.txt"), 0); // NOT LINEARLY SEPARABLE?
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/09_unknown_problem.txt"), 0);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/10_unknown_problem.txt"), 0);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/11_unknown_problem.txt"), 0.09);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/12_unknown_problem.txt"), 0.001);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/13_unknown_problem.txt"), 0.001);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/14_unknown_problem.txt"), 0.001);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/15_unknown_problem.txt"), 0);
    System.out.println("-------------------------------------------");
    testSingleAdalineProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/16_unknown_problem.txt"), 0.004);
  }

  private void testSingleAdalineProblem(List<Feed> tests, double acceptableError) {
    Network nn = new ADALINE(tests);
    nn.learn();

    tests.stream().forEach(test -> { double[] inputs = test.getInputs();
                                     double[] outputs = test.getOutputs();
                                     double[] result = nn.process(inputs);
                                     assertTrue(compareDoubleArraysWithAcceptableError(outputs, result, acceptableError));
                                     System.out.println("inputs: " + Arrays.toString(inputs) + " --> outputs: " + Arrays.toString(outputs)); });
  }

  private void testSingleAdaptiveFilterProblem(List<Feed> tests, double acceptableError) {
    Network nn = new AdaptiveFilter(tests);
    nn.learn();

    tests.stream().forEach(test -> { double[] inputs = new double[] { test.getInputs()[0] };
                                     double[] outputs = test.getOutputs();
                                     double[] result = nn.process(inputs);
                                     assertTrue(compareDoubleArraysWithAcceptableError(outputs, result, acceptableError));
                                     System.out.println("inputs: " + Arrays.toString(inputs) + " --> outputs: " + Arrays.toString(outputs)); });
  }

  private boolean compareDoubleArraysWithAcceptableError(double[] target, double[] result, double acceptableError) {
    for (int i = 0; i < result.length; i++) {
      if (!(Math.abs(target[i] - result[i]) <= acceptableError)) {
        return false;
      }
    }
    return true;
  }

  public static void main(String[] args) {
    List<Feed> tests_unknown16_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/16_unknown_problem.txt");
    Network myNN_unknown16_problem = new ADALINE(tests_unknown16_problem);
    myNN_unknown16_problem.learn();
  
    double[] input = new double[] { 2, 4 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown16_problem.process(input)));
  
    input = new double[] { 4, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown16_problem.process(input)));

    input = new double[] { -2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown16_problem.process(input)));

    System.out.println("-------------------------------------------");
  }
}
