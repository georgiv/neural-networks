package joro.nn.tests;

import java.util.Arrays;
import java.util.List;

import joro.nn.api.Network;
import joro.nn.impl.core.Feed;
import joro.nn.impl.core.LearningRuleType;
import joro.nn.impl.networks.LinearAssociator;
import joro.nn.impl.networks.Perception;
import joro.nn.impl.utils.CalibrationFeedGenerator;

public class HebbLearningRuleTests {
  public static void main(String[] args) {
    List<Feed> tests_unknown01_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/01_unknown_problem.txt");
    Network myNN_unknown01_problem = new LinearAssociator(tests_unknown01_problem);
    myNN_unknown01_problem.learn();

    double[] input = new double[] { 0.5, -0.5, 0.5, -0.5 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown01_problem.process(input)));

    input = new double[] { 0.5, 0.5, -0.5, -0.5 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown01_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_fruits_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/02_fruits_problem.txt");
    Network myNN_fruits_problem = new LinearAssociator(tests_fruits_problem);
    myNN_fruits_problem.learn();

    input = new double[] { 1, -1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_fruits_problem.process(input)));

    input = new double[] { 1, 1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_fruits_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_digits_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/03_digits_problem.txt");
    Network myNN_digits_problem = new LinearAssociator(tests_digits_problem);
    myNN_digits_problem.learn();

    double[] result = myNN_digits_problem.process(-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1 , -1, 1, -1, 1, 1, 1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(-1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_digits_problem.process(-1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown04_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/04_unknown_problem.txt");
    Network myNN_unknown04_problem = new LinearAssociator(tests_unknown04_problem);
    myNN_unknown04_problem.learn();

    input = new double[] { 1, -1, 1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    input = new double[] { 1, 1, -1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown04_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_tetris_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/05_tetris_problem.txt");
    Network myNN_tetris_problem = new LinearAssociator(tests_tetris_problem);
    myNN_tetris_problem.learn();

    result = myNN_tetris_problem.process(1, 1, 1, -1, -1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("---");
      }
      if ((i + 1) % 2 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_tetris_problem.process(-1, 1, 1, 1, 1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("---");
      }
      if ((i + 1) % 2 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    result = myNN_tetris_problem.process(1, 1, 1, 1, 1, -1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("---");
      }
      if ((i + 1) % 2 == 0) {
        System.out.println();
      }
    }
    System.out.println();

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown06_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/06_unknown_problem.txt");
    Network myNN_unknown06_problem = new LinearAssociator(tests_unknown06_problem);
    myNN_unknown06_problem.learn();

    input = new double[] { 1, 1, -1, -1 ,1, 1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { 1, 1, 1, -1, 1, -1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { -1, 1, -1, 1, 1, -1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    input = new double[] { -1, 1, -1, -1, 1, -1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown06_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown07_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/07_unknown_problem.txt");
    Network myNN_unknown07_problem = new Perception(tests_unknown07_problem, LearningRuleType.HEBB);
    myNN_unknown07_problem.learn();

    input = new double[] { 1, -1, 1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { 1, 1, -1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { -1, -1, -1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    input = new double[] { 1, -1, 1, -1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown07_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown08_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/08_unknown_problem.txt");
    Network myNN_unknown08_problem = new Perception(tests_unknown08_problem, LearningRuleType.HEBB);
    myNN_unknown08_problem.learn();

    input = new double[] { 1, 1 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown08_problem.process(input)));

    input = new double[] { 2, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown08_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown09_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/09_unknown_problem.txt");
    Network myNN_unknown09_problem = new LinearAssociator(tests_unknown09_problem);
    myNN_unknown09_problem.learn();

    input = new double[] { 2, 4 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown09_problem.process(input)));

    input = new double[] { 4, 2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown09_problem.process(input)));

    input = new double[] { -2, -2 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown09_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_unknown10_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/10_unknown_problem.txt");
    Network myNN_unknown10_problem = new LinearAssociator(tests_unknown10_problem);
    myNN_unknown10_problem.learn();

    input = new double[] { 3, 6 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown10_problem.process(input)));

    input = new double[] { 6, 3 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown10_problem.process(input)));

    input = new double[] { -6, 3 };
    System.out.println(Arrays.toString(input) + " --> " + Arrays.toString(myNN_unknown10_problem.process(input)));

    System.out.println("-------------------------------------------");

    List<Feed> tests_digits2_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/11_digits2_problem.txt");
    Network myNN_digits2_problem = new LinearAssociator(tests_digits2_problem);
    myNN_digits2_problem.learn();
  
    result = myNN_digits2_problem.process(0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0 , 0, 1, 0, 1, 1, 1, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_digits2_problem.process(0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("   ");
      }
      if ((i + 1) % 5 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    System.out.println("-------------------------------------------");
    List<Feed> tests_tetris2_problem = CalibrationFeedGenerator.generate("./learning_rule_test_inputs/hebb/12_tetris2_problem.txt");
    Network myNN_tetris2_problem = new LinearAssociator(tests_tetris2_problem);
    myNN_tetris2_problem.learn();
  
    result = myNN_tetris2_problem.process(1, 1, 1, 0, 0, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("---");
      }
      if ((i + 1) % 2 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_tetris2_problem.process(0, 1, 1, 1, 1, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("---");
      }
      if ((i + 1) % 2 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  
    result = myNN_tetris2_problem.process(1, 1, 1, 1, 1, 0);
    for (int i = 0; i < result.length; i++) {
      if (result[i] == 1) {
        System.out.print(result[i]);
      } else {
        System.out.print("---");
      }
      if ((i + 1) % 2 == 0) {
        System.out.println();
      }
    }
    System.out.println();
  }
}
