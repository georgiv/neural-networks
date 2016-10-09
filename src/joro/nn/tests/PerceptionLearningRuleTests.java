package joro.nn.tests;

import java.util.Arrays;
import java.util.List;

import joro.nn.api.Network;
import joro.nn.impl.core.Feed;
import joro.nn.impl.networks.Perception;
import joro.nn.impl.utils.CalibrationFeedGenerator;

public class PerceptionLearningRuleTests {
  public static void main(String[] args) {
    List<Feed> tests_and_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/01_and_problem.txt");
    Network myNN_and_problem = new Perception(tests_and_problem);
    myNN_and_problem.learn();

    double[] input = new double[] { 0, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_and_problem.process(input)));

    input = new double[] { 0, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_and_problem.process(input)));

    input = new double[] { 1, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_and_problem.process(input)));

    input = new double[] { 1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_and_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_fruits_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/02_fruits_problem.txt");
    Network myNN_fruits_problem = new Perception(tests_fruits_problem);
    myNN_fruits_problem.learn();

    input = new double[] { 1, -1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_fruits_problem.process(input)));

    input = new double[] { 1, 1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_fruits_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown03_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/03_unknown_problem.txt");
    Network myNN_unknown03_problem = new Perception(tests_unknown03_problem);
    myNN_unknown03_problem.learn();

    input = new double[] { 0, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown03_problem.process(input)));

    input = new double[] { 1, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown03_problem.process(input)));

    input = new double[] { 0, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown03_problem.process(input)));

    input = new double[] { 2, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown03_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown04_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/04_unknown_problem.txt");
    Network myNN_unknown04_problem = new Perception(tests_unknown04_problem);
    myNN_unknown04_problem.learn();

    input = new double[] { -2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { -2, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { -2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { 0, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { 0, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { 2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { 2, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { 2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown05_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/05_unknown_problem.txt");
    Network myNN_unknown05_problem = new Perception(tests_unknown05_problem);
    myNN_unknown05_problem.learn();

    input = new double[] { -2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown05_problem.process(input)));

    input = new double[] { -2, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown05_problem.process(input)));

    input = new double[] { -2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown05_problem.process(input)));

    input = new double[] { 0, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown05_problem.process(input)));

    input = new double[] { 0, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown05_problem.process(input)));

    input = new double[] { 2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown05_problem.process(input)));

    input = new double[] { 2, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown05_problem.process(input)));

    input = new double[] { 2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown05_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown06_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/06_unknown_problem.txt");
    Network myNN_unknown06_problem = new Perception(tests_unknown06_problem);
    myNN_unknown06_problem.learn();

    input = new double[] { -2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { -2, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { -2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { 0, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { 0, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { 2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { 2, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { 2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown07_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/07_unknown_problem.txt");
    Network myNN_unknown07_problem = new Perception(tests_unknown07_problem);
    myNN_unknown07_problem.learn();

    input = new double[] { 1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { 1, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { 2, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { 2, 0 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { -1, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { -2, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { -1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { -2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown08_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/08_unknown_problem.txt");
    Network myNN_unknown08_problem = new Perception(tests_unknown08_problem);
    myNN_unknown08_problem.learn();

    input = new double[] { 2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown08_problem.process(input)));

    input = new double[] { 1, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown08_problem.process(input)));

    input = new double[] { -2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown08_problem.process(input)));

    input = new double[] { -1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown08_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown09_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/perception/09_unknown_problem.txt");
    Network myNN_unknown09_problem = new Perception(tests_unknown09_problem);
    myNN_unknown09_problem.learn();

    input = new double[] { 1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown09_problem.process(input)));

    input = new double[] { -1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown09_problem.process(input)));

    input = new double[] { 2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown09_problem.process(input)));
  }
}
