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
    
  }

  private void testSingleAdalineProblem(List<Feed> tests, double acceptableError) {
    Network nn = new ADALINE(tests);
    testSingleProblem(nn, tests, acceptableError);
  }

  private void testSingleAdaptiveFilterProblem(List<Feed> tests, double acceptableError) {
    Network nn = new AdaptiveFilter(tests);
    testSingleProblem(nn, tests, acceptableError);
  }

  private void testSingleProblem(Network nn, List<Feed> tests, double acceptableError) {
    nn.learn();

    tests.stream().forEach(test -> { double[] inputs = test.getInputs();
                                     double[] outputs = test.getOutputs();
                                     double[] result = nn.process(test.getInputs());
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
//    List<Feed> tests_fruits_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/01_fruits_problem.txt");
//    Network myNN_fruits_problem = new ADALINE(tests_fruits_problem);
//    myNN_fruits_problem.learn();
//
//    double[] input = new double[] { 1, -1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_fruits_problem.process(input)));
//
//    input = new double[] { 1, 1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_fruits_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown02_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/02_unknown_problem.txt");
//    Network myNN_unknown02_problem = new ADALINE(tests_unknown02_problem);
//    myNN_unknown02_problem.learn();
//
//    double[] input = new double[] { 1, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown02_problem.process(input)));
//
//    input = new double[] { -1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown02_problem.process(input)));
//
//    input = new double[] { 2, 2 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown02_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown03_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/03_unknown_problem.txt");
//    Network myNN_unknown03_problem = new ADALINE(tests_unknown03_problem);
//    myNN_unknown03_problem.learn();
//  
//    double[] input = new double[] { 1, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown03_problem.process(input)));
//  
//    input = new double[] { 1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown03_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
    List<Feed> tests_unknown04_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/04_unknown_problem.txt");
    Network myNN_unknown04_problem = new ADALINE(tests_unknown04_problem);
    myNN_unknown04_problem.learn();
  
    double[] input = new double[] { 1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));
  
    input = new double[] { 1, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));
  
    input = new double[] { 2, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { 2, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { -1, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { -2, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { -1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { -2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_letters_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/05_letters_problem.txt");
//    Network myNN_letters_problem = new ADALINE(tests_letters_problem);
//    myNN_letters_problem.learn();
//
//    input = new double[] { 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_letters_problem.process(input)));
//
//    input = new double[] { -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_letters_problem.process(input)));
//
//    input = new double[] { 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_letters_problem.process(input)));
//
//    input = new double[] { -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_letters_problem.process(input)));
//
//    input = new double[] { 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_letters_problem.process(input)));
//
//    input = new double[] { -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_letters_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown06_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/06_unknown_problem.txt");
//    Network myNN_unknown06_problem = new AdaptiveFilter(tests_unknown06_problem);
//    myNN_unknown06_problem.learn();
//
//    input = new double[] { 0, 0, 0 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));
//
//    input = new double[] { 5, 0, 0 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));
//
//    input = new double[] { -4, 5, 0 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));
//
//    input = new double[] { 0, -4, 5 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));
//
//    input = new double[] { 0, 0, -4 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown07_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/07_unknown_problem.txt");
//    Network myNN_unknown07_problem = new AdaptiveFilter(tests_unknown07_problem);
//    myNN_unknown07_problem.learn();
//
//    input = new double[] { 0, 0, 0 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));
//
//    input = new double[] { 1, 0, 0 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));
//
//    input = new double[] { 1, 1, 0 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));
//
//    input = new double[] { 2, 1, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));
//
//    input = new double[] { 0, 2, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));
//
//    input = new double[] { 0, 0, 2 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//  
//    // NOT LINEARLY SEPARABLE?
//    List<Feed> tests_lines_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/08_lines_problem.txt");
//    Network myNN_lines_problem = new ADALINE(tests_lines_problem);
//    myNN_lines_problem.learn();
//  
//    input = new double[] { 1, 1, -1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_lines_problem.process(input)));
//  
//    input = new double[] { -1, -1, 1, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_lines_problem.process(input)));
//  
//    input = new double[] { 1, -1, 1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_lines_problem.process(input)));
//  
//    input = new double[] { -1, 1, -1, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_lines_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown09_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/09_unknown_problem.txt");
//    Network myNN_unknown09_problem = new ADALINE(tests_unknown09_problem);
//    myNN_unknown09_problem.learn();
//  
//    input = new double[] { 1, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown09_problem.process(input)));
//  
//    input = new double[] { -1, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown09_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown10_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/10_unknown_problem.txt");
//    Network myNN_unknown10_problem = new ADALINE(tests_unknown10_problem);
//    myNN_unknown10_problem.learn();
//  
//    input = new double[] { 1, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown10_problem.process(input)));
//  
//    input = new double[] { -1, 2 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown10_problem.process(input)));
//
//    input = new double[] { 0, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown10_problem.process(input)));
//  
//    input = new double[] { -4, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown10_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown11_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/11_unknown_problem.txt");
//    Network myNN_unknown11_problem = new ADALINE(tests_unknown11_problem);
//    myNN_unknown11_problem.learn();
//  
//    input = new double[] { 3, 6 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown11_problem.process(input)));
//  
//    input = new double[] { 6, 3 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown11_problem.process(input)));
//
//    input = new double[] { -6, 3 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown11_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown12_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/12_unknown_problem.txt");
//    Network myNN_unknown12_problem = new ADALINE(tests_unknown12_problem);
//    myNN_unknown12_problem.learn();
//  
//    input = new double[] { 1, 2 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown12_problem.process(input)));
//  
//    input = new double[] { -2, 1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown12_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown13_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/13_unknown_problem.txt");
//    Network myNN_unknown13_problem = new ADALINE(tests_unknown13_problem);
//    myNN_unknown13_problem.learn();
//  
//    input = new double[] { 4, 2 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown13_problem.process(input)));
//  
//    input = new double[] { 2, -4 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown13_problem.process(input)));
//
//    input = new double[] { -4, 4 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown13_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown14_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/14_unknown_problem.txt");
//    Network myNN_unknown14_problem = new ADALINE(tests_unknown14_problem);
//    myNN_unknown14_problem.learn();
//  
//    input = new double[] { 4, 2 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown14_problem.process(input)));
//  
//    input = new double[] { 2, -4 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown14_problem.process(input)));
//
//    input = new double[] { -4, 4 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown14_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown15_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/15_unknown_problem.txt");
//    Network myNN_unknown15_problem = new ADALINE(tests_unknown15_problem);
//    myNN_unknown15_problem.learn();
//  
//    input = new double[] { -1, 2 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown15_problem.process(input)));
//  
//    input = new double[] { 2, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown15_problem.process(input)));
//
//    input = new double[] { 0, -1 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown15_problem.process(input)));
//
//    input = new double[] { -1, 0 };
//    System.out.println(Arrays.toString(input) + "k --> " + Arrays.toString(myNN_unknown15_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
//
//    List<Feed> tests_unknown16_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/lms/16_unknown_problem.txt");
//    Network myNN_unknown16_problem = new ADALINE(tests_unknown16_problem);
//    myNN_unknown16_problem.learn();
//  
//    input = new double[] { 2, 4 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown16_problem.process(input)));
//  
//    input = new double[] { 4, 2 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown16_problem.process(input)));
//
//    input = new double[] { -2, -2 };
//    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown16_problem.process(input)));
//
//    System.out.println("-------------------------------------------");
  }
}
