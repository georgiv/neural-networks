package joro.nn.tests;

import Jama.Matrix;
import joro.nn.impl.utils.Calculator;

public class MatrixDeternminantOptimization {
  public static void main(String[] args) {
    double[][] arr = new double[][] {
      new double[] { 1.0002, 0.0, 0.0, -1.0002, 0.6668, 0.0, -0.6668, 0.0, 0.6668, 0.0, -0.3334, -0.3334, 0.6668, 0.0, -0.3334, -0.3334, 0.0 },
      new double[] { 0.0, 1.0002, 1.0002, 0.0, -0.3334, 0.3334, -0.3334, -1.0002, -0.3334, 0.3334, 0.0, -0.6668, -0.3334, 0.3334, 0.0, -0.6668, 1.0002 },
      new double[] { 0.0, 1.0002, 1.0002, 0.0, -0.3334, 0.3334, -0.3334, -1.0002, -0.3334, 0.3334, 0.0, -0.6668, -0.3334, 0.3334, 0.0, -0.6668, 1.0002 },
      new double[] { -1.0002, 0.0, 0.0, 1.0002, -0.6668, 0.0, 0.6668, 0.0, -0.6668, 0.0, 0.3334, 0.3334, -0.6668, 0.0, 0.3334, 0.3334, 0.0 },
      new double[] { 0.6668, -0.3334, -0.3334, -0.6668, 1.0002, -0.3334, -0.3334, 0.3334, 1.0002, -0.3334, 0.0, 0.0, 1.0002, -0.3334, 0.0, 0.0, -0.3334 },
      new double[] { 0.0, 0.3334, 0.3334, 0.0, -0.3334, 1.0002, -0.3334, -0.3334, -0.3334, 0.3334, -0.6668, 0.0, -0.3334, 0.3334, -0.6668, 0.0, 0.3334 },
      new double[] { -0.6668, -0.3334, -0.3334, 0.6668, -0.3334, -0.3334, 1.0002, 0.3334, -0.3334, -0.3334, 0.0, 0.0, -0.3334, -0.3334, 0.0, 0.0, -0.3334 },
      new double[] { 0.0, -1.0002, -1.0002, 0.0, 0.3334, -0.3334, 0.3334, 1.0002, 0.3334, -0.3334, 0.0, 0.6668, 0.3334, -0.3334, 0.0, 0.6668, -1.0002 },
      new double[] { 0.6668, -0.3334, -0.3334, -0.6668, 1.0002, -0.3334, -0.3334, 0.3334, 1.0002, -0.3334, 0.0, 0.0, 1.0002, -0.3334, 0.0, 0.0, -0.3334 },
      new double[] { 0.0, 0.3334, 0.3334, 0.0, -0.3334, 0.3334, -0.3334, -0.3334, -0.3334, 1.0002, 0.0, 0.0, -0.3334, 1.0002, 0.0, 0.0, 0.3334 },
      new double[] { -0.3334, 0.0, 0.0, 0.3334, 0.0, -0.6668, 0.0, 0.0, 0.0, 0.0, 1.0002, 0.3334, 0.0, 0.0, 1.0002, 0.3334, 0.0 },
      new double[] { -0.3334, -0.6668, -0.6668, 0.3334, 0.0, 0.0, 0.0, 0.6668, 0.0, 0.0, 0.3334, 1.0002, 0.0, 0.0, 0.3334, 1.0002, -0.6668 },
      new double[] { 0.6668, -0.3334, -0.3334, -0.6668, 1.0002, -0.3334, -0.3334, 0.3334, 1.0002, -0.3334, 0.0, 0.0, 1.0002, -0.3334, 0.0, 0.0, -0.3334 },
      new double[] { 0.0, 0.3334, 0.3334, 0.0, -0.3334, 0.3334, -0.3334, -0.3334, -0.3334, 1.0002, 0.0, 0.0, -0.3334, 1.0002, 0.0, 0.0, 0.3334 },
      new double[] { -0.3334, 0.0, 0.0, 0.3334, 0.0, -0.6668, 0.0, 0.0, 0.0, 0.0, 1.0002, 0.3334, 0.0, 0.0, 1.0002, 0.3334, 0.0 },
      new double[] { -0.3334, -0.6668, -0.6668, 0.3334, 0.0, 0.0, 0.0, 0.6668, 0.0, 0.0, 0.3334, 1.0002, 0.0, 0.0, 0.3334, 1.0002, -0.6668 },
      new double[] { 0.0, 1.0002, 1.0002, 0.0, -0.3334, 0.3334, -0.3334, -1.0002, -0.3334, 0.3334, 0.0, -0.6668, -0.3334, 0.3334, 0.0, -0.6668, 1.0002 }
    };

//    arr = new double[][] {
//      new double[] { 6.111, 1.111, 1.111 },
//      new double[] { 4.111, -2.111, 5.111 },
//      new double[] { 2.111, 8.111, 7.111 }
//    };
//
//    // Calculate determinant with own method
//    double determinant = Calculator.calculateMatrixDeterminant(arr);
//    System.out.println("Own determinant: " + determinant);

    // Calculate determinant with Jama
    Matrix matrix = new Matrix(arr);
    double jamaDet = matrix.det();
    System.out.println("Jama determinant: " + jamaDet);
  }
}