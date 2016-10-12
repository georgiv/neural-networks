package joro.nn.impl.utils;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import Jama.Matrix;

public final class Calculator {
  public static final int DEFAULT_SCALE = 4;

  public static double roundDouble(double value) {
    return roundDouble(value, DEFAULT_SCALE);
  }

  public static double roundDouble(double value, int scale) {
    return new BigDecimal(value).setScale(scale, RoundingMode.HALF_UP).doubleValue();
  }

  public static double[] addDoubleArrays(double[] addend, double[] augend) {
    if (addend.length != augend.length) {
      throw new IllegalArgumentException("Both arrays should have the same length.\n" + 
                                         "Addend length: " + addend.length + "\n" + 
                                         "Augend length: " + augend.length);
    }

    BigDecimal[] bdAddend = toBigDecimalArray(addend);
    BigDecimal[] bdAugend = toBigDecimalArray(augend);

    double[] result = new double[addend.length];
    for (int i = 0; i < result.length; i++) {
      result[i] = bdAddend[i].add(bdAugend[i]).doubleValue();
    }
    return result;
  }

  public static double[] subtractDoubleArrays(double[] minuend, double[] subtrahend) {
    if (minuend.length != subtrahend.length) {
      throw new IllegalArgumentException("Both arrays should have the same length.\n" + 
                                         "Minuend length: " + minuend.length + "\n" + 
                                         "Subtrahend length: " + subtrahend.length);
    }

    BigDecimal[] bdMinuend = toBigDecimalArray(minuend);
    BigDecimal[] bdSubtrahend = toBigDecimalArray(subtrahend);

    double[] result = new double[minuend.length];
    for (int i = 0; i < result.length; i++) {
      result[i] = bdMinuend[i].subtract(bdSubtrahend[i]).doubleValue();
    }
    return result;
  }

  public static double[] multiplyDoubleArrays(double[] multiplicand, double[] multiplier) {
    if (multiplicand.length != multiplier.length) {
      throw new IllegalArgumentException("Both arrays should have the same length.\n" + 
                                         "Multiplicand length: " + multiplicand.length + "\n" + 
                                         "Multiplier length: " + multiplier.length);
    }

    BigDecimal[] bdMultiplicand = toBigDecimalArray(multiplicand);
    BigDecimal[] bdMultiplier = toBigDecimalArray(multiplier);

    double[] result = new double[multiplicand.length];
    for (int i = 0; i < result.length; i++) {
      result[i] = bdMultiplicand[i].multiply(bdMultiplier[i]).setScale(DEFAULT_SCALE, RoundingMode.HALF_UP).doubleValue();
    }
    return result;
  }

  public static double getVectorMagnitude(double[] vector) {
    if (vector.length < 1) {
      throw new IllegalArgumentException("A vector should have at least one element.\n" + 
                                         "Vector length: " + vector.length);
    }

    BigDecimal[] bdVector = toBigDecimalArray(vector);

    BigDecimal length = new BigDecimal(0);
    for (int i = 0; i < bdVector.length; i++) {
      length = length.add(bdVector[i].pow(2));
    }
    return roundDouble(Math.sqrt(length.setScale(DEFAULT_SCALE, RoundingMode.HALF_UP).doubleValue()));
  }

  public static double[] normalizeVector(double[] vector) {
    if (vector.length < 1) {
      throw new IllegalArgumentException("A vector should have at least one element.\n" + 
                                         "Vector length: " + vector.length);
    }

    double vectorMagnitude = getVectorMagnitude(vector);
    if (vectorMagnitude == 0) {
      return new double[vector.length];
    }
    if (vectorMagnitude == 1) {
      return vector;
    }

    BigDecimal[] bdVector = toBigDecimalArray(vector);
    BigDecimal bdVectorMagnitude = new BigDecimal(vectorMagnitude);

    double[] result = new double[vector.length];
    for (int i = 0; i < result.length; i++) {
      result[i] = bdVector[i].divide(bdVectorMagnitude, DEFAULT_SCALE, RoundingMode.HALF_UP).doubleValue();
    }
    return result;
  }

  public static double[][] transposeMatrix(double[][] matrix) {
    if ((matrix.length < 1) || (matrix[0].length < 1)) {
      throw new IllegalArgumentException("A matrix should have at least one row and one column.");
    }

    double[][] result = new double[matrix[0].length][matrix.length];
    for (int i = 0; i < result.length; i++) {
      for (int j = 0; j < result[0].length; j++) {
        result[i][j] = matrix[j][i];
      }
    }
    return result;
  }

  public static double[][] addMatrices(double[][] addend, double[][] augend) {
    if ((addend.length < 1) || (addend[0].length < 1) || (augend.length < 1) || (augend[0].length < 1)) {
      throw new IllegalArgumentException("Both matrices should have at least one row and one column.");
    }

    if (addend.length != augend.length) {
      throw new IllegalArgumentException("Both matrices should have the same rows count.\n" + 
                                         "Addend length: " + addend.length + "\n" +  
                                         "Augend length: " + augend.length);
    }

    int columns = addend[0].length;

    double[][] result = new double[addend.length][];
    for (int i = 0; i < result.length; i++) {
      if (addend[i].length != columns) {
        throw new IllegalArgumentException("All rows of both matrices should have the same elements count.\n" + 
                                           "Row " + i + " of addend contains " + addend[i].length + " elements, while the previous contain " + columns + " elements.");
      }
      if (augend[i].length != columns) {
        throw new IllegalArgumentException("Invalid input - all elements of both matrices should be the same size.\n" + 
                                           "Row " + i + " of augend contains " + augend[i].length + " elements, while the previous contain " + columns + " elements.");
      }

      result[i] = addDoubleArrays(addend[i], augend[i]);
    }
    return result;
  }

  public static double[][] subtractMatrices(double[][] minuend, double[][] subtrahend) {
    if ((minuend.length < 1) || (minuend[0].length < 1) || (subtrahend.length < 1) || (subtrahend[0].length < 1)) {
      throw new IllegalArgumentException("Both matrices should have at least one row and one column.");
    }

    if (minuend.length != subtrahend.length) {
      throw new IllegalArgumentException("Both matrices should have the same rows count.\n" + 
                                         "Minuend length: " + minuend.length + "\n" +  
                                         "Subtrahend length: " + subtrahend.length);
    }

    int columns = minuend[0].length;

    double[][] result = new double[minuend.length][];
    for (int i = 0; i < result.length; i++) {
      if (minuend[i].length != columns) {
        throw new IllegalArgumentException("All rows of both matrices should have the same elements count.\n" + 
                                           "Row " + i + " of minuend contains " + minuend[i].length + " elements, while the previous contain " + columns + " elements.");
      }
      if (subtrahend[i].length != columns) {
        throw new IllegalArgumentException("All rows of both matrices should have the same elements count.\n" + 
                                           "Row " + i + " of subtrahend contains " + subtrahend[i].length + " elements, while the previous contain " + columns + " elements.");
      }

      result[i] = subtractDoubleArrays(minuend[i], subtrahend[i]);
    }
    return result;
  }

  public static double[][] multiplyMatrix(double[][] multiplicand, double multiplier) {
    if ((multiplicand.length < 1) || (multiplicand[0].length < 1)) {
      throw new IllegalArgumentException("A matrix should have at least one row and one column.");
    }

    BigDecimal[][] bdMultiplicand = toBigDecimalMatrix(multiplicand);
    BigDecimal bdMultiplier = new BigDecimal(multiplier);

    int columns = multiplicand[0].length;

    double[][] result = new double[bdMultiplicand.length][];
    for (int i = 0; i < result.length; i++) {
      if (multiplicand[i].length != columns) {
        throw new IllegalArgumentException("All rows of a matrix should have the same elements count.\n" + 
                                           "Row " + i + " of matrix contains " + multiplicand[i].length + " elements, while the previous contain " + columns + " elements.");
      }

      result[i] = new double[bdMultiplicand[i].length];
      for (int j = 0; j < bdMultiplicand[i].length; j++) {
        result[i][j] = bdMultiplicand[i][j].multiply(bdMultiplier).setScale(DEFAULT_SCALE, RoundingMode.HALF_UP).doubleValue();
      }
    }
    return result;
  }

  public static double[][] multiplyMatrices(double[][] multiplicand, double[][] multiplier) {
    if ((multiplicand.length < 1) || (multiplicand[0].length < 1) || (multiplier.length < 1) || (multiplier[0].length < 1)) {
      throw new IllegalArgumentException("Both matrices should have at least one row and one column.");
    }

    double[][] transposedMultiplicand = transposeMatrix(multiplicand);
    if (transposedMultiplicand.length != multiplier.length) {
      throw new IllegalArgumentException("The multiplicand columns count should be the same as the multiplier rows count.\n" + 
                                         "Multiplicand columns: " + transposedMultiplicand.length + "\n" +  
                                         "Multiplier columns: " + multiplier.length);
    }

    double[][] transposedMultiplier = transposeMatrix(multiplier);

    double[][] result = new double[multiplicand.length][transposedMultiplier.length];
    for (int i = 0; i < result.length; i++) {
      for (int j = 0; j < transposedMultiplier.length; j++) {
        double[] product = multiplyDoubleArrays(multiplicand[i], transposedMultiplier[j]);
        result[i][j] = roundDouble(DoubleStream.of(product).sum());
      }
    }
    return result;
  }

  public static double[][] getIdentityMatrix(int length) {
    if (length < 2) {
      throw new IllegalArgumentException("The size of an identity matrix should be at least 2.\n" + 
                                         "Size: " + length + ".");
    }

    double[][] result = new double[length][length];
    for (int i = 0; i < result.length; i++) {
      for (int j = 0; j < result[i].length; j++) {
        if (i == j) {
          result[i][j] = 1;
        }
      }
    }
    return result;
  }

  public static boolean isSquareMatrix(double[][] matrix) {
    if (matrix.length < 2) {
      return false;
    }

    for (int i = 0; i < matrix.length; i++) {
      if (matrix.length != matrix[i].length) {
        return false;
      }
    }
    return true;
  }

  public static double calculateMatrixDeterminant(double[][] matrix) {
    if(!isSquareMatrix(matrix)) {
      throw new IllegalArgumentException("Determinant can be calculated only for square matrix with minimum size 2.");
    }

//    if (matrix.length == 2) {
//      BigDecimal a = new BigDecimal(matrix[0][0]);
//      BigDecimal b = new BigDecimal(matrix[0][1]);
//      BigDecimal c = new BigDecimal(matrix[1][0]);
//      BigDecimal d = new BigDecimal(matrix[1][1]);
//      return a.multiply(d).subtract(b.multiply(c)).setScale(DEFAULT_SCALE, RoundingMode.HALF_UP).doubleValue();
//    }
//
//    BigDecimal multiplicatorA = new BigDecimal(1);
//
//    BigDecimal result = new BigDecimal(0);
//    for (int i = 0; i < matrix[0].length; i++) {
//      BigDecimal multiplicatorB = new BigDecimal(matrix[0][i]);
//
//      double[][] subMatrix = getSubSquareMatrix(matrix, 0, i);
//      double submatrixDeterminant = calculateMatrixDeterminant(subMatrix);
//      BigDecimal multiplicatorC = new BigDecimal(submatrixDeterminant);
//
//      result = result.add(multiplicatorA.multiply(multiplicatorB).multiply(multiplicatorC).setScale(DEFAULT_SCALE, RoundingMode.HALF_UP));
//
//      multiplicatorA = multiplicatorA.multiply(new BigDecimal(-1));
//    }
//    return result.setScale(DEFAULT_SCALE, RoundingMode.HALF_UP).doubleValue();

    return roundDouble(new Matrix(matrix).det());
    
  }

  public static double[][] getInverseMatrix(double[][] matrix) {
    if (!isSquareMatrix(matrix)) {
      throw new IllegalArgumentException("Inverse can be calculated only for square matrix with minimum size 2.");
    }

    double determinant = calculateMatrixDeterminant(matrix);
    if (determinant == 0) {
      throw new IllegalArgumentException("Inverse does not exists, because the matrix is singular.");
    }

    double scalar = new BigDecimal(1).divide(new BigDecimal(determinant), 10, RoundingMode.HALF_UP).doubleValue();

    if (matrix.length == 2) {
      double[][] modified = new double[2][2];
      modified[0][0] = roundDouble(matrix[1][1]);
      modified[1][1] = roundDouble(matrix[0][0]);

      BigDecimal multiplier = new BigDecimal(-1);
      modified[0][1] = multiplier.multiply(new BigDecimal(matrix[0][1])).setScale(DEFAULT_SCALE, RoundingMode.HALF_UP).doubleValue();
      modified[1][0] = multiplier.multiply(new BigDecimal(matrix[1][0])).setScale(DEFAULT_SCALE, RoundingMode.HALF_UP).doubleValue();

      return multiplyMatrix(modified, scalar);
    }

    double[][] result = calculateMatrixOfMinors(matrix);
    result = calculateMatrixOfCofactors(result);
    result = transposeMatrix(result);
    result = multiplyMatrix(result, scalar);
    return result;
  }

  public static double[][] getLeftPseudoinverseMatrix(double[][] matrix) {
    double[][] result = multiplyMatrices(transposeMatrix(matrix), matrix);
    result = getInverseMatrix(result);
    result = multiplyMatrices(result, transposeMatrix(matrix));
    return result;
  }

  public static double[][] getRightPseudoinverseMatrix(double[][] matrix) {
    double[][] result = multiplyMatrices(matrix, transposeMatrix(matrix));
    result = getInverseMatrix(result);
    result = multiplyMatrices(transposeMatrix(matrix), result);
    return result;
  }

  private static BigDecimal[] toBigDecimalArray(double[] values) {
    return DoubleStream.of(values).mapToObj(d -> new BigDecimal(d).setScale(DEFAULT_SCALE, RoundingMode.HALF_UP)).toArray(BigDecimal[]::new);
  }

  private static BigDecimal[][] toBigDecimalMatrix(double[][] matrix) {
    return Stream.of(matrix).map(a -> toBigDecimalArray(a)).toArray(BigDecimal[][]::new);
  }

  private static double[][] getSubSquareMatrix(double[][] matrix, int row, int column) {
    double[][] result = new double[matrix.length - 1][matrix.length - 1];

    int resultRow = 0;
    for (int i = 0; i < matrix.length; i++) {
      if (i == row) {
        continue;
      }

      int resultColumn = 0;
      for (int j = 0; j < matrix[i].length; j++) {
        if (j == column) {
          continue;
        }

        result[resultRow][resultColumn] = matrix[i][j];
        resultColumn++;
      }
      resultRow++;
    }
    return result;
  }

  private static double[][] calculateMatrixOfMinors(double[][] matrix) {
    double[][] result = new double[matrix.length][matrix.length];
    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[i].length; j++) {
        double[][] subMatrix = getSubSquareMatrix(matrix, i, j);
        double subMatrixeterminant = calculateMatrixDeterminant(subMatrix);
        result[i][j] = subMatrixeterminant;
      }
    }
    return result;
  }

  private static double[][] calculateMatrixOfCofactors(double[][] matrix) {
    double[][] result = new double[matrix.length][matrix.length];
    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[i].length; j++) {
        BigDecimal base = new BigDecimal(-1);
        BigDecimal multiplicatorA = base.pow(i + j);
        BigDecimal multiplicatorB = new BigDecimal(matrix[i][j]);
        result[i][j] = multiplicatorA.multiply(multiplicatorB).setScale(DEFAULT_SCALE, RoundingMode.HALF_UP).doubleValue();
      }
    }
    return result;
  }
}
