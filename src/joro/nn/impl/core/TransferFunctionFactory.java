package joro.nn.impl.core;

import java.util.function.DoubleUnaryOperator;

public final class TransferFunctionFactory {
  private static TransferFunctionFactory INSTANCE = new TransferFunctionFactory();

  private TransferFunctionFactory() {
  }

  public static TransferFunctionFactory getInstance() {
    synchronized (TransferFunctionFactory.class) {
      if (INSTANCE == null) {
        INSTANCE = new TransferFunctionFactory();
      }
    }
    return INSTANCE;
  }

  public DoubleUnaryOperator getTransferFunction(TransferFunctionType type) {
    DoubleUnaryOperator transferFunction = null;

    switch (type) {
      case HARD_LIMIT: transferFunction = i -> i >= 0 ? 1 : 0; //hardlim
        break;
      case SYMMETRICAL_HARD_LIMIT: transferFunction = i -> i >= 0 ? 1 : -1; //hardlims
        break;
      case LINEAR: transferFunction = i -> i; //purelin
        break;
      case POSITIVE_LINEAR: transferFunction = i -> i >= 0 ? i : 0; //poslin
        break;
      case SATURATING_LINEAR: transferFunction = i -> i > 1 ? 1 : (i < 0 ? 0 : i); //satlin
        break;
      case SYMMETRIC_SATURATING_LINEAR: transferFunction = i -> i > 1 ? 1 : (i < -1 ? -1 : i); //satlins
        break;
      case LOG_SIGMOID: transferFunction = i -> 1 / (1 + Math.exp(-i)); //logsig
        break;
      case HYPERBOLIC_TANGENT_SIGMOID: transferFunction = i -> (Math.exp(i) - Math.exp(-i)) / (Math.exp(i) + Math.exp(-i)); //tansig
        break;
      case COMPETITIVE: transferFunction = null; // TODO: implement compet
        break;
      default: throw new IllegalArgumentException("Unknow transfer function type: " + type);
    }

    return transferFunction;
  }
}
