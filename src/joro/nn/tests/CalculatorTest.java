package joro.nn.tests;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import joro.nn.impl.utils.Calculator;

public final class CalculatorTest {

  @Test
  public void roundDoubleTest() {
    double value = Calculator.roundDouble(0);
    assertTrue(0 == value);

    value = Calculator.roundDouble(-0.0);
    assertTrue(Double.doubleToRawLongBits(0.0) == Double.doubleToRawLongBits(value));

    value = Calculator.roundDouble(1);
    assertTrue(1 == value);

    value = Calculator.roundDouble(1.123456);
    assertTrue(1.1235 == value);
    assertFalse(1.123 == value);
    assertFalse(1.1234 == value);

    value = Calculator.roundDouble(-9.987654);
    assertTrue(-9.9877 == value);
    assertFalse(-9.987 == value);
    assertFalse(-9.9876 == value);
  }

  @Test
  public void addDoubleArraysSuccessfulTest() {
    double[] addend = new double[] { 1, 1, 1 };
    double[] augend = new double[] { 2, 2, 2 };
    double[] target = new double[] { 3, 3, 3 };
    assertTrue(Arrays.equals(target, Calculator.addDoubleArrays(addend, augend)));

    addend = new double[] { 0, 0, 0 };
    augend = new double[] { 1, 1, 1 };
    target = new double[] { 1, 1, 1 };
    assertTrue(Arrays.equals(target, Calculator.addDoubleArrays(addend, augend)));

    addend = new double[] { -1, -1, -1 };
    augend = new double[] { 2, 2, 2 };
    target = new double[] { 1, 1, 1 };
    assertTrue(Arrays.equals(target, Calculator.addDoubleArrays(addend, augend)));

    addend = new double[] { -1, -1, -1 };
    augend = new double[] { -2, -2, -2 };
    target = new double[] { -3, -3, -3 };
    assertTrue(Arrays.equals(target, Calculator.addDoubleArrays(addend, augend)));

    addend = new double[] { 1, 2, 3 };
    augend = new double[] { 4, 5, 6 };
    target = new double[] { 5, 7, 9 };
    assertTrue(Arrays.equals(target, Calculator.addDoubleArrays(addend, augend)));

    addend = new double[] { 1.5, 2.3, 3.7 };
    augend = new double[] { 4.1, 5.3, 6.9 };
    target = new double[] { 5.6, 7.6, 10.6 };
    assertTrue(Arrays.equals(target, Calculator.addDoubleArrays(addend, augend)));
  }

  @Test(expected = IllegalArgumentException.class)
  public void addDoubleArraysExceptionTest() {
    double[] addend = new double[] { 1, 1 };
    double[] augend = new double[] { 2, 2, 2 };
    Calculator.addDoubleArrays(addend, augend);
  }

  @Test
  public void subtractDoubleArraysSuccessfulTest() {
    double[] minuend = new double[] { 1, 1, 1 };
    double[] subtrahend = new double[] { 2, 2, 2 };
    double[] target = new double[] { -1, -1, -1 };
    assertTrue(Arrays.equals(target, Calculator.subtractDoubleArrays(minuend, subtrahend)));

    minuend = new double[] { 0, 0, 0 };
    subtrahend = new double[] { 1, 1, 1 };
    target = new double[] { -1, -1, -1 };
    assertTrue(Arrays.equals(target, Calculator.subtractDoubleArrays(minuend, subtrahend)));

    minuend = new double[] { -1, -1, -1 };
    subtrahend = new double[] { 2, 2, 2 };
    target = new double[] { -3, -3, -3 };
    assertTrue(Arrays.equals(target, Calculator.subtractDoubleArrays(minuend, subtrahend)));

    minuend = new double[] { -1, -1, -1 };
    subtrahend = new double[] { -2, -2, -2 };
    target = new double[] { 1, 1, 1 };
    assertTrue(Arrays.equals(target, Calculator.subtractDoubleArrays(minuend, subtrahend)));

    minuend = new double[] { 1, 2, 3 };
    subtrahend = new double[] { 4, 5, 6 };
    target = new double[] { -3, -3, -3 };
    assertTrue(Arrays.equals(target, Calculator.subtractDoubleArrays(minuend, subtrahend)));

    minuend = new double[] { 1.5, 2.3, 3.7 };
    subtrahend = new double[] { 4.1, 5.3, 6.9 };
    target = new double[] { -2.6, -3, -3.2 };
    assertTrue(Arrays.equals(target, Calculator.subtractDoubleArrays(minuend, subtrahend)));
  }

  @Test(expected = IllegalArgumentException.class)
  public void subtractDoubleArraysExceptionTest() {
    double[] minuend = new double[] { 1, 1 };
    double[] subtrahend = new double[] { 2, 2, 2 };
    Calculator.subtractDoubleArrays(minuend, subtrahend);
  }

  @Test
  public void multiplyDoubleArraysSuccessfulTest() {
    double[] multiplicand = new double[] { 1, 1, 1 };
    double[] multiplier = new double[] { 2, 2, 2 };
    double[] target = new double[] { 2, 2, 2 };
    assertTrue(Arrays.equals(target, Calculator.multiplyDoubleArrays(multiplicand, multiplier)));

    multiplicand = new double[] { 0, 0, 0 };
    multiplier = new double[] { 1, 1, 1 };
    target = new double[] { 0, 0, 0 };
    assertTrue(Arrays.equals(target, Calculator.multiplyDoubleArrays(multiplicand, multiplier)));

    multiplicand = new double[] { 0, 0, 0 };
    multiplier = new double[] { -1, -1, -1 };
    target = new double[] { 0, 0, 0 };
    assertTrue(Arrays.equals(target, Calculator.multiplyDoubleArrays(multiplicand, multiplier)));

    multiplicand = new double[] { -1, -1, -1 };
    multiplier = new double[] { 2, 2, 2 };
    target = new double[] { -2, -2, -2 };
    assertTrue(Arrays.equals(target, Calculator.multiplyDoubleArrays(multiplicand, multiplier)));

    multiplicand = new double[] { -1, -1, -1 };
    multiplier = new double[] { -2, -2, -2 };
    target = new double[] { 2, 2, 2 };
    assertTrue(Arrays.equals(target, Calculator.multiplyDoubleArrays(multiplicand, multiplier)));

    multiplicand = new double[] { 1, 2, 3 };
    multiplier = new double[] { 4, 5, 6 };
    target = new double[] { 4, 10, 18 };
    assertTrue(Arrays.equals(target, Calculator.multiplyDoubleArrays(multiplicand, multiplier)));

    multiplicand = new double[] { 1.753, 2.563, 3.4 };
    multiplier = new double[] { 4.851, 5.15, 6.8524 };
    target = new double[] { 8.5038, 13.1995, 23.2982 };
    assertTrue(Arrays.equals(target, Calculator.multiplyDoubleArrays(multiplicand, multiplier)));
  }

  @Test(expected = IllegalArgumentException.class)
  public void multiplyDoubleArraysExceptionTest() {
    double[] multiplicand = new double[] { 1, 1 };
    double[] multiplier = new double[] { 2, 2, 2 };
    Calculator.multiplyDoubleArrays(multiplicand, multiplier);
  }

  @Test
  public void getVectorMagnitudeSuccessfulTest() {
    double[] vector1D = new double[] { 2 };
    double target = 2;
    assertTrue(target == Calculator.getVectorMagnitude(vector1D));

    vector1D = new double[] { -2 };
    target = 2;
    assertTrue(target == Calculator.getVectorMagnitude(vector1D));

    vector1D = new double[] { -5.985467 };
    target = 5.9855;
    assertTrue(target == Calculator.getVectorMagnitude(vector1D));

    vector1D = new double[] { 0 };
    target = 0;
    assertTrue(target == Calculator.getVectorMagnitude(vector1D));

    double[] vector2D = new double[] { 0, 1 };
    target = 1;
    assertTrue(target == Calculator.getVectorMagnitude(vector2D));

    vector2D = new double[] { 0, -1 };
    target = 1;
    assertTrue(target == Calculator.getVectorMagnitude(vector2D));

    vector2D = new double[] { -4, 7 };
    target = 8.0623;
    assertTrue(target == Calculator.getVectorMagnitude(vector2D));

    vector2D = new double[] { 0, 0 };
    target = 0;
    assertTrue(target == Calculator.getVectorMagnitude(vector2D));

    double[] vector3D = new double[] { 0, 0, 1 };
    target = 1;
    assertTrue(target == Calculator.getVectorMagnitude(vector3D));

    vector3D = new double[] { 0, 0, -1 };
    target = 1;
    assertTrue(target == Calculator.getVectorMagnitude(vector3D));

    vector3D = new double[] { 4, 3, 8 };
    target = 9.434;
    assertTrue(target == Calculator.getVectorMagnitude(vector3D));

    vector3D = new double[] { 0, 0, 0 };
    target = 0;
    assertTrue(target == Calculator.getVectorMagnitude(vector3D));
  }

  @Test(expected = IllegalArgumentException.class)
  public void getVectorMagnitudeExceptionTest() {
    double[] vector = new double[0];
    Calculator.getVectorMagnitude(vector);
  }

  @Test
  public void normalizeVectorSuccessfulTest() {
    double[] unitVector = new double[] { 0, 1, 0 };
    assertTrue(Arrays.equals(unitVector, Calculator.normalizeVector(unitVector)));

    unitVector = new double[] { 0.5, -0.5, 0.5, -0.5 };
    assertTrue(Arrays.equals(unitVector, Calculator.normalizeVector(unitVector)));

    double[] zeroVector = new double[] { 0, 0, 0 };
    assertTrue(Arrays.equals(new double[] { 0, 0, 0 }, Calculator.normalizeVector(zeroVector)));

    double[] vector = new double[] { 1, -1, -1 };
    double[] target = new double[] { 0.5773, -0.5773, -0.5773 };
    assertTrue(Arrays.equals(target, Calculator.normalizeVector(vector)));

    vector = new double[] { 1, 2, 3 };
    target = new double[] { 0.2673, 0.5345, 0.8018 };
    assertTrue(Arrays.equals(target, Calculator.normalizeVector(vector)));
  }

  @Test(expected = IllegalArgumentException.class)
  public void normalizeVectorExceptionTest() {
    double[] vector = new double[0];
    Calculator.normalizeVector(vector);
  }

  @Test
  public void transposeMatrixSuccessfulTest() {
    double[][] matrix = new double[][] { 
      new double[] { 1, 2, 3 },
    };
    double[][] target = new double[][] { 
      new double[] { 1 },
      new double[] { 2 },
      new double[] { 3 }
    };
    double[][] transposed = Calculator.transposeMatrix(matrix);
    for (int i = 0; i < transposed.length; i++) {
      assertTrue(Arrays.equals(target[i], transposed[i]));
    }

    matrix = new double[][] { 
      new double[] { 1 },
      new double[] { 2 },
      new double[] { 3 }
    };
    target = new double[][] { 
      new double[] { 1, 2, 3 }
    };
    transposed = Calculator.transposeMatrix(matrix);
    for (int i = 0; i < transposed.length; i++) {
      assertTrue(Arrays.equals(target[i], transposed[i]));
    }

    matrix = new double[][] { 
      new double[] { 1, 2, 3 },
      new double[] { 1, 2, 3 }
    };
    target = new double[][] { 
      new double[] { 1, 1 },
      new double[] { 2, 2 },
      new double[] { 3, 3 }
    };
    transposed = Calculator.transposeMatrix(matrix);
    for (int i = 0; i < transposed.length; i++) {
      assertTrue(Arrays.equals(target[i], transposed[i]));
    }

    matrix = new double[][] { 
      new double[] { 1, 2, 3 },
      new double[] { 1, 2, 3 },
      new double[] { 1, 2, 3 }
    };
    target = new double[][] { 
      new double[] { 1, 1, 1 },
      new double[] { 2, 2, 2 },
      new double[] { 3, 3, 3 }
    };
    transposed = Calculator.transposeMatrix(matrix);
    for (int i = 0; i < transposed.length; i++) {
      assertTrue(Arrays.equals(target[i], transposed[i]));
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void transposeZeroRowsMatrixExceptionTest() {
    double[][] matrix = new double[0][1];
    Calculator.transposeMatrix(matrix);
  }

  @Test(expected = IllegalArgumentException.class)
  public void transposeZeroColumnsMatrixExceptionTest() {
    double[][] matrix = new double[1][0];
    Calculator.transposeMatrix(matrix);
  }

  @Test
  public void addMatricesSuccessfulTest() {
    double[][] addend = new double[][] {
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 }
    };
    double[][] audend = new double[][] {
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 }
    };
    double[][] target = new double[][] {
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 }
    };
    double[][] result = Calculator.addMatrices(addend, audend);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    addend = new double[][] {
      new double[] { -1, -1, -1 },
      new double[] { -1, -1, -1 },
      new double[] { -1, -1, -1 }
    };
    audend = new double[][] {
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 }
    };
    target = new double[][] {
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 }
    };
    result = Calculator.addMatrices(addend, audend);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    addend = new double[][] {
      new double[] { 1, 2, 3 },
      new double[] { 4, 5, 6 }
    };
    audend = new double[][] {
      new double[] { 1, 2, 3 },
      new double[] { 4, 5, 6 }
    };
    target = new double[][] {
      new double[] { 2, 4, 6 },
      new double[] { 8, 10, 12 }
    };
    result = Calculator.addMatrices(addend, audend);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    addend = new double[][] {
      new double[] { -1.18, 2.4, 3.1 },
      new double[] { 11.4, 36.946, -7.03 }
    };
    audend = new double[][] {
      new double[] { 2.41, -1.11, 7.432 },
      new double[] { -18.9, 1.2, 104.1111 }
    };
    target = new double[][] {
      new double[] { 1.23, 1.29, 10.532 },
      new double[] { -7.5, 38.146, 97.0811 }
    };
    result = Calculator.addMatrices(addend, audend);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void addMatricesDifferentRowsExceptionTest() {
    double[][] addend = new double[3][3];
    double[][] augend = new double[2][3];
    Calculator.addMatrices(addend, augend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void addMatricesDifferentColumnsExceptionTest() {
    double[][] addend = new double[3][3];
    double[][] augend = new double[][] {
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 },
      new double[] { 1, 1 }
    };
    Calculator.addMatrices(addend, augend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void addZeroRowsAMatricesExceptionTest() {
    double[][] addend = new double[0][1];
    double[][] augend = new double[1][1];
    Calculator.addMatrices(addend, augend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void addZeroColumnsAMatricesExceptionTest() {
    double[][] addend = new double[1][0];
    double[][] augend = new double[1][1];
    Calculator.addMatrices(addend, augend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void addZeroRowsBMatricesExceptionTest() {
    double[][] addend = new double[1][1];
    double[][] augend = new double[0][1];
    Calculator.addMatrices(addend, augend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void addZeroColumnsBMatricesExceptionTest() {
    double[][] addend = new double[1][1];
    double[][] augend = new double[1][0];
    Calculator.addMatrices(addend, augend);
  }

  @Test
  public void subtractMatricesSuccessfulTest() {
    double[][] minuend = new double[][] {
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 }
    };
    double[][] subtrahend = new double[][] {
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 }
    };
    double[][] target = new double[][] {
      new double[] { -1, -1, -1 },
      new double[] { -1, -1, -1 },
      new double[] { -1, -1, -1 }
    };
    double[][] result = Calculator.subtractMatrices(minuend, subtrahend);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    minuend = new double[][] {
      new double[] { -1, -1, -1 },
      new double[] { -1, -1, -1 },
      new double[] { -1, -1, -1 }
    };
    subtrahend = new double[][] {
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 }
    };
    target = new double[][] {
      new double[] { -2, -2, -2 },
      new double[] { -2, -2, -2 },
      new double[] { -2, -2, -2 }
    };
    result = Calculator.subtractMatrices(minuend, subtrahend);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    minuend = new double[][] {
      new double[] { 1, 2, 3 },
      new double[] { 4, 5, 6 }
    };
    subtrahend = new double[][] {
      new double[] { 1, 2, 3 },
      new double[] { 4, 5, 6 }
    };
    target = new double[][] {
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 }
    };
    result = Calculator.subtractMatrices(minuend, subtrahend);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    minuend = new double[][] {
      new double[] { -1.18, 2.4, 3.1 },
      new double[] { 11.4, 36.946, -7.03 }
    };
    subtrahend = new double[][] {
      new double[] { 2.41, -1.11, 7.432 },
      new double[] { -18.9, 1.2, 104.1111 }
    };

    target = new double[][] {
      new double[] { -3.59, 3.51, -4.332 },
      new double[] { 30.3, 35.746, -111.1411 }
    };
    result = Calculator.subtractMatrices(minuend, subtrahend);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void subtractMatricesDifferentRowsExceptionTest() {
    double[][] minuend = new double[3][3];
    double[][] subtrahend = new double[2][3];
    Calculator.subtractMatrices(minuend, subtrahend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void subtractMatricesDifferentColumnsExceptionTest() {
    double[][] minuend = new double[3][3];
    double[][] subtrahend = new double[][] {
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 },
      new double[] { 1, 1 }
    };
    Calculator.subtractMatrices(minuend, subtrahend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void subtractZeroRowsAMatricesExceptionTest() {
    double[][] minuend = new double[0][1];
    double[][] subtrahend = new double[1][1];
    Calculator.subtractMatrices(minuend, subtrahend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void subtractZeroColumnsAMatricesExceptionTest() {
    double[][] minuend = new double[1][0];
    double[][] subtrahend = new double[1][1];
    Calculator.subtractMatrices(minuend, subtrahend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void subtractZeroRowsBMatricesExceptionTest() {
    double[][] minuend = new double[1][1];
    double[][] subtrahend = new double[0][1];
    Calculator.subtractMatrices(minuend, subtrahend);
  }

  @Test(expected = IllegalArgumentException.class)
  public void subtractZeroColumnsBMatricesExceptionTest() {
    double[][] minuend = new double[1][1];
    double[][] subtrahend = new double[1][0];
    Calculator.subtractMatrices(minuend, subtrahend);
  }

  @Test
  public void multiplyMatrixSuccessfulTest() {
    double[][] multiplicand = new double[][] {
      new double[] { 1, 2, 3 }
    };
    double multiplier = 2;
    double[][] target = new double[][] {
      new double[] { 2, 4, 6 }
    };
    double[][] result = Calculator.multiplyMatrix(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    multiplicand = new double[][] {
      new double[] { 1 },
      new double[] { 2 },
      new double[] { 3 }
    };
    multiplier = 4;
    target = new double[][] {
      new double[] { 4 },
      new double[] { 8 },
      new double[] { 12 }
    };
    result = Calculator.multiplyMatrix(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    multiplicand = new double[][] {
      new double[] { 1, 2, 3 },
      new double[] { 2, 3, 4 },
      new double[] { 3, 4, 5 }
    };
    multiplier = 3;
    target = new double[][] {
      new double[] { 3, 6, 9 },
      new double[] { 6, 9, 12 },
      new double[] { 9, 12, 15 }
    };
    result = Calculator.multiplyMatrix(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    multiplicand = new double[][] {
      new double[] { 1, 2, 3 },
      new double[] { 2, 3, 4 },
      new double[] { 3, 4, 5 }
    };
    multiplier = 2.111111;
    target = new double[][] {
      new double[] { 2.1111, 4.2222, 6.3333 },
      new double[] { 4.2222, 6.3333, 8.4444 },
      new double[] { 6.3333, 8.4444, 10.5556 }
    };
    result = Calculator.multiplyMatrix(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    multiplicand = new double[][] {
      new double[] { 1, 2, 3 },
      new double[] { 2, 3, 4 },
      new double[] { 3, 4, 5 }
    };
    multiplier = -2.111111;
    target = new double[][] {
      new double[] { -2.1111, -4.2222, -6.3333 },
      new double[] { -4.2222, -6.3333, -8.4444 },
      new double[] { -6.3333, -8.4444, -10.5556 }
    };
    result = Calculator.multiplyMatrix(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    multiplicand = new double[][] {
      new double[] { -1, -2, -3 },
      new double[] { -2, -3, -4 },
      new double[] { 3, 4, 5 }
    };
    multiplier = -2.111111;
    target = new double[][] {
      new double[] { 2.1111, 4.2222, 6.3333 },
      new double[] { 4.2222, 6.3333, 8.4444 },
      new double[] { -6.3333, -8.4444, -10.5556 }
    };
    result = Calculator.multiplyMatrix(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    multiplicand = new double[][] {
      new double[] { -1, -2, -3 },
      new double[] { -2, -3, -4 },
      new double[] { 3, 4, 5 }
    };
    multiplier = 0;
    target = new double[][] {
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 }
    };
    result = Calculator.multiplyMatrix(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    multiplicand = new double[][] {
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 }
    };
    multiplier = -3;
    target = new double[][] {
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 },
      new double[] { 0, 0, 0 }
    };
    result = Calculator.multiplyMatrix(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void multiplyMatrixDifferentColumnsExceptionTest() {
    double[][] multiplicand = new double[][] {
      new double[] { 1, 1, 1 },
      new double[] { 1, 1, 1 },
      new double[] { 1, 1 }
    };
    Calculator.multiplyMatrix(multiplicand, 1);
  }

  @Test(expected = IllegalArgumentException.class)
  public void multiplyZeroRowsMatrixExceptionTest() {
    double[][] multiplicand = new double[0][1];
    Calculator.multiplyMatrix(multiplicand, 1);
  }

  @Test(expected = IllegalArgumentException.class)
  public void multiplyZeroColumnsMatrixExceptionTest() {
    double[][] multiplicand = new double[1][0];
    Calculator.multiplyMatrix(multiplicand, 1);
  }

  @Test
  public void multiplyMatricesSuccessfulTest() {
    double[][] multiplicand = new double[][] {
      new double[] { 1, 2, 3 },
      new double[] { 4, 5, 6 }
    };
    double[][] multiplier = new double[][] {
      new double[] { 7, 8 },
      new double[] { 9, 10 },
      new double[] { 11, 12 }
    };
    double[][] target = new double[][] {
      new double[] { 58, 64 },
      new double[] { 139, 154 }
    };
    double[][] result = Calculator.multiplyMatrices(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    multiplicand = new double[][] {
      new double[] { 3, 4, 2 }
    };
    multiplier = new double[][] {
      new double[] { 13, 9, 7, 15 },
      new double[] { 8, 7, 4, 6 },
      new double[] { 6, 4, 0, 3 }
    };
    target = new double[][] {
      new double[] { 83, 63, 37, 75 }
    };
    result = Calculator.multiplyMatrices(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    multiplicand = new double[][] {
      new double[] { -1, 3.431123, -9.222222 }
    };
    multiplier = new double[][] {
      new double[] { 0, 9 },
      new double[] { 0, 7 },
      new double[] { 0, 4 }
    };
    target = new double[][] {
      new double[] { 0, -21.8711 }
    };
    result = Calculator.multiplyMatrices(multiplicand, multiplier);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void multiplyMatricesExceptionTest() {
    double[][] multiplicand = new double[][] {
      new double[] { 1, 2, 3 },
      new double[] { 4, 5, 6 }
    };
    double[][] multiplier = new double[][] {
      new double[] { 7, 8 },
      new double[] { 9, 10 },
      new double[] { 11, 12 },
      new double[] { 16, 20 }
    };
    Calculator.multiplyMatrices(multiplicand, multiplier);
  }

  @Test(expected = IllegalArgumentException.class)
  public void multiplyZeroRowsAMatricesExceptionTest() {
    double[][] multiplicand = new double[0][1];
    double[][] multiplier = new double[1][1];
    Calculator.multiplyMatrices(multiplicand, multiplier);
  }

  @Test(expected = IllegalArgumentException.class)
  public void multiplyZeroColumnsAMatricesExceptionTest() {
    double[][] multiplicand = new double[1][0];
    double[][] multiplier = new double[1][1];
    Calculator.multiplyMatrices(multiplicand, multiplier);
  }

  @Test(expected = IllegalArgumentException.class)
  public void multiplyZeroRowsBMatricesExceptionTest() {
    double[][] multiplicand = new double[1][1];
    double[][] multiplier = new double[0][1];
    Calculator.multiplyMatrices(multiplicand, multiplier);
  }

  @Test(expected = IllegalArgumentException.class)
  public void multiplyZeroColumnsBMatricesExceptionTest() {
    double[][] multiplicand = new double[1][1];
    double[][] multiplier = new double[1][0];
    Calculator.multiplyMatrices(multiplicand, multiplier);
  }

  @Test
  public void createIdentityMatrixSuccessfulTest() {
    int size = 2;
    double[][] target = new double[][] {
      new double[] { 1, 0 },
      new double[] { 0, 1 }
    };
    double[][] result = Calculator.getIdentityMatrix(size);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    size = 3;
    target = new double[][] {
      new double[] { 1, 0, 0 },
      new double[] { 0, 1, 0 },
      new double[] { 0, 0, 1 }
    };
    result = Calculator.getIdentityMatrix(size);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    size = 4;
    target = new double[][] {
      new double[] { 1, 0, 0, 0 },
      new double[] { 0, 1, 0, 0 },
      new double[] { 0, 0, 1, 0 },
      new double[] { 0, 0, 0, 1 }
    };
    result = Calculator.getIdentityMatrix(size);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void createIdentityMatrixNegativeSizeExceptionTest() {
    Calculator.getIdentityMatrix(-1);
  }

  @Test(expected = IllegalArgumentException.class)
  public void createIdentityMatrixSizeZeroExceptionTest() {
    Calculator.getIdentityMatrix(0);
  }

  @Test(expected = IllegalArgumentException.class)
  public void createIdentityMatrixSizeOneExceptionTest() {
    Calculator.getIdentityMatrix(1);
  }

  @Test
  public void isSquareMatrixTest() {
    
    assertTrue(Calculator.isSquareMatrix(new double[2][2]));
    assertTrue(Calculator.isSquareMatrix(new double[3][3]));
    assertTrue(Calculator.isSquareMatrix(new double[100][100]));

    assertFalse(Calculator.isSquareMatrix(new double[0][0]));
    assertFalse(Calculator.isSquareMatrix(new double[1][1]));
    assertFalse(Calculator.isSquareMatrix(new double[0][1]));
    assertFalse(Calculator.isSquareMatrix(new double[1][2]));
  }

  @Test
  public void calculateMatrixDeterminantSuccessfulTest() {
    double[][] matrix = new double[][] {
      new double[] { 3, 8 },
      new double[] { 4, 6 }
    };
    double target = -14;
    assertTrue(target == Calculator.calculateMatrixDeterminant(matrix));

    matrix = new double[][] {
      new double[] { 6, 1, 1 },
      new double[] { 4, -2, 5 },
      new double[] { 2, 8, 7 }
    };
    target = -306;
    assertTrue(target == Calculator.calculateMatrixDeterminant(matrix));

    matrix = new double[][] {
      new double[] { 6.111, 1.111, 1.111 },
      new double[] { 4.111, -2.111, 5.111 },
      new double[] { 2.111, 8.111, 7.111 }
    };
    target = -323.5622;
    assertTrue(target == Calculator.calculateMatrixDeterminant(matrix));
  }

  @Test(expected = IllegalArgumentException.class)
  public void calculateZeroMatrixDeterminantExceptionTest() {
    Calculator.calculateMatrixDeterminant(new double[0][0]);
  }

  @Test(expected = IllegalArgumentException.class)
  public void calculateLengthOneMatrixDeterminantExceptionTest() {
    Calculator.calculateMatrixDeterminant(new double[1][1]);
  }

  @Test(expected = IllegalArgumentException.class)
  public void calculateNotSquareMatrixDeterminantExceptionTest() {
    Calculator.calculateMatrixDeterminant(new double[3][4]);
  }

  @Test
  public void getInverseMatrixSuccessfulTest() {
    double[][] matrix = new double[][] {
      new double[] { 4, 7 },
      new double[] { 2, 6 }
    };
    double[][] target = new double[][] {
      new double[] { 0.6, -0.7 },
      new double[] { -0.2, 0.4 }
    };
    double[][] result = Calculator.getInverseMatrix(matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    double[][] identityMatrix = Calculator.getIdentityMatrix(matrix.length);
    double[][] product = Calculator.multiplyMatrices(result, matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(identityMatrix[i], product[i]));
    }

    matrix = new double[][] {
      new double[] { 3, 3.5 },
      new double[] { 3.2, 3.6 }
    };
    target = new double[][] {
      new double[] { -9, 8.75 },
      new double[] { 8, -7.5 }
    };
    result = Calculator.getInverseMatrix(matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    identityMatrix = Calculator.getIdentityMatrix(matrix.length);
    product = Calculator.multiplyMatrices(result, matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(identityMatrix[i], product[i]));
    }

    matrix = new double[][] {
      new double[] { 3, 3.2 },
      new double[] { 3.5, 3.6 }
    };
    target = new double[][] {
      new double[] { -9, 8 },
      new double[] { 8.75, -7.5 }
    };
    result = Calculator.getInverseMatrix(matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    identityMatrix = Calculator.getIdentityMatrix(matrix.length);
    product = Calculator.multiplyMatrices(result, matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(identityMatrix[i], product[i]));
    }

    matrix = new double[][] {
      new double[] { 1, 3, 3 },
      new double[] { 1, 4, 3 },
      new double[] { 1, 3, 4 }
    };
    target = new double[][] {
      new double[] { 7, -3, -3 },
      new double[] { -1, 1, 0 },
      new double[] { -1, 0, 1 }
    };
    result = Calculator.getInverseMatrix(matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    identityMatrix = Calculator.getIdentityMatrix(matrix.length);
    product = Calculator.multiplyMatrices(result, matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(identityMatrix[i], product[i]));
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void getInverseMatrixZeroDeterminantExceptionTest() {
    double[][] matrix = new double[][] {
      new double[] { 3, 4 },
      new double[] { 6, 8 }
    };
    Calculator.getInverseMatrix(matrix);
  }

  @Test(expected = IllegalArgumentException.class)
  public void getInverseNotSquareMatrixExceptionTest() {
    Calculator.getInverseMatrix(new double[1][2]);
  }

  @Test
  public void getLeftPseudoinverseSuccessfulTest() {
    double[][] matrix = new double[][] {
      new double[] { 1, 1 },
      new double[] { -1, 1 },
      new double[] { -1, -1 }
    };
    double[][] target = new double[][] {
      new double[] { 0.25, -0.5, -0.25 },
      new double[] { 0.25, 0.5, -0.25 }
    };
    double[][] result = Calculator.getLeftPseudoinverseMatrix(matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    double[][] identityMatrix = Calculator.getIdentityMatrix(result.length);
    double[][] product = Calculator.multiplyMatrices(result, matrix);
    for (int i = 0; i < product.length; i++) {
      assertTrue(Arrays.equals(identityMatrix[i], product[i]));
    }
  }

  @Test
  public void getRightPseudoinverseMatrixTest() {
    double matrix[][] = new double[][] {
      new double[] { 1, 1, 1, 1 },
      new double[] { 5, 7, 7, 9 }
    };
    double[][] target = new double[][] {
      new double[] { 2, -0.25 },
      new double[] { 0.25, 0 },
      new double[] { 0.25, 0 },
      new double[] { -1.5, 0.25 }
    };
    double[][] result = Calculator.getRightPseudoinverseMatrix(matrix);
    for (int i = 0; i < result.length; i++) {
      assertTrue(Arrays.equals(target[i], result[i]));
    }

    double[][] identityMatrix = Calculator.getIdentityMatrix(matrix.length);
    double[][] product = Calculator.multiplyMatrices(matrix, result);
    for (int i = 0; i < product.length; i++) {
      assertTrue(Arrays.equals(identityMatrix[i], product[i]));
    }
  }
}
