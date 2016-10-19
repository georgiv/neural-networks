package joro.nn.tests;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import joro.nn.api.Network;
import joro.nn.impl.core.Feed;
import joro.nn.impl.core.LearningRuleType;
import joro.nn.impl.networks.LinearAssociator;
import joro.nn.impl.networks.Perception;
import joro.nn.impl.utils.CalibrationFeedGenerator;

public class HebbLearningRuleTest {

  @Test
  public void testAll() {
    testSingleLinearAssociatorProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/01_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSingleLinearAssociatorProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/02_fruits_problem.txt"));
    System.out.println("-------------------------------------------");
    testSinglePatternRecognitionProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/03_digits_problem.txt"), 5);
    System.out.println("-------------------------------------------");
    testSingleLinearAssociatorProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/04_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSinglePatternRecognitionProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/05_tetris_problem.txt"), 2);
    System.out.println("-------------------------------------------");
    testSingleLinearAssociatorProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/06_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSinglePerceptionProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/07_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
    testSinglePerceptionProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/08_unknown_problem.txt"));
    System.out.println("-------------------------------------------");
//    testSingleLinearAssociatorProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/09_unknown_problem.txt"));
//    System.out.println("-------------------------------------------");
//    testSingleLinearAssociatorProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/10_unknown_problem.txt"));
//    System.out.println("-------------------------------------------");
    testSinglePatternRecognitionProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/11_digits2_problem.txt"), 5);
    System.out.println("-------------------------------------------");
    testSinglePatternRecognitionProblem(CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/12_tetris2_problem.txt"), 2);
  }

  private void testSingleLinearAssociatorProblem(List<Feed> tests) {
    Network nn = new LinearAssociator(tests);
    testSingleProblem(nn, tests);
  }

  private void testSinglePerceptionProblem(List<Feed> tests) {
    Network nn = new Perception(tests, LearningRuleType.HEBB);
    testSingleProblem(nn, tests);
  }

  private void testSingleProblem(Network nn, List<Feed> tests) {
    nn.learn();

    tests.stream().forEach(test -> { double[] inputs = test.getInputs();
                                     double[] outputs = test.getOutputs();
                                     double[] result = nn.process(inputs);
                                     assertTrue(Arrays.equals(outputs, result));
                                     System.out.println("inputs: " + Arrays.toString(inputs) + " --> outputs: " + Arrays.toString(outputs)); });
  }

  private void testSinglePatternRecognitionProblem(List<Feed> tests, int lineSeparator) {
    Network nn = new LinearAssociator(tests);
    nn.learn();

    tests.stream().forEach(test -> { double[] outputs = test.getOutputs();
                                     double[] result = nn.process(test.getInputs());
                                     assertTrue(Arrays.equals(outputs, result));
                                     printPattern(result, lineSeparator); });
  }

  private static void printPattern(double[] pattern, int lineSeparator) {
    for (int i = 0; i < pattern.length; i++) {
      if (pattern[i] == 1) {
        System.out.print(pattern[i]);
      } else {
        System.out.print("---");
      }
      if ((i + 1) % lineSeparator == 0) {
        System.out.println();
      }
    }
    System.out.println();
  }

  public static void main(String[] args) {
    List<Feed> tests_digits_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/03_digits_problem.txt");
    Network myNN_digits_problem = new LinearAssociator(tests_digits_problem);
    myNN_digits_problem.learn();

    double[] result = myNN_digits_problem.process(-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1 , -1, 1, -1, 1, 1, 1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(-1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1);
    printPattern(result, 5);

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1);
    printPattern(result, 5);

    System.out.println("-------------------------------------------");

    List<Feed> tests_tetris_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/05_tetris_problem.txt");
    Network myNN_tetris_problem = new LinearAssociator(tests_tetris_problem);
    myNN_tetris_problem.learn();

    result = myNN_tetris_problem.process(1, 1, 1, -1, -1, -1);
    printPattern(result, 2);

    result = myNN_tetris_problem.process(-1, 1, 1, 1, 1, -1);
    printPattern(result, 2);

    result = myNN_tetris_problem.process(1, 1, 1, 1, 1, -1);
    printPattern(result, 2);

    System.out.println("-------------------------------------------");

    List<Feed> tests_digits2_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/11_digits2_problem.txt");
    Network myNN_digits2_problem = new LinearAssociator(tests_digits2_problem);
    myNN_digits2_problem.learn();
  
    result = myNN_digits2_problem.process(0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0 , 0, 1, 0, 1, 1, 1, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0);
    printPattern(result, 5);
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1);
    printPattern(result, 5);
  
    System.out.println("-------------------------------------------");
    List<Feed> tests_tetris2_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/12_tetris2_problem.txt");
    Network myNN_tetris2_problem = new LinearAssociator(tests_tetris2_problem);
    myNN_tetris2_problem.learn();
  
    result = myNN_tetris2_problem.process(1, 1, 1, 0, 0, 0);
    printPattern(result, 2);
  
    result = myNN_tetris2_problem.process(0, 1, 1, 1, 1, 0);
    printPattern(result, 2);
  
    result = myNN_tetris2_problem.process(1, 1, 1, 1, 1, 0);
    printPattern(result, 2);
  }
}
