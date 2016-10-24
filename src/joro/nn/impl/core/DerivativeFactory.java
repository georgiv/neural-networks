package joro.nn.impl.core;

import java.util.function.DoubleUnaryOperator;

public final class DerivativeFactory {
  private static DerivativeFactory INSTANCE = new DerivativeFactory();

  private DerivativeFactory() {
  }

  public static DerivativeFactory getInstance() {
    synchronized (TransferFunctionFactory.class) {
      if (INSTANCE == null) {
        INSTANCE = new DerivativeFactory();
      }
    }
    return INSTANCE;
  }

  public DoubleUnaryOperator getTransferFunction(TransferFunctionType type) {
    DoubleUnaryOperator derivative = null;

    switch (type) {
      case HARD_LIMIT: derivative = null; //TODO: implement derivative for hardlim
        break;
      case SYMMETRICAL_HARD_LIMIT: derivative = null; //TODO: implement derivative for hardlims
        break;
      case LINEAR: derivative = i -> 1; //purelin
        break;
      case POSITIVE_LINEAR: derivative = null; //TODO: implement derivative for poslin
        break;
      case SATURATING_LINEAR: derivative = null; //TODO: implement derivative for satlin
        break;
      case SYMMETRIC_SATURATING_LINEAR: derivative = null; //TODO: implement derivative for satlins
        break;
      case LOG_SIGMOID: derivative = i -> (1 - i) * i; //logsig
        break;
      case HYPERBOLIC_TANGENT_SIGMOID: derivative = null; //TODO: implement derivative for tansig
        break;
      case COMPETITIVE: derivative = null; //TODO: implement derivative for compet
        break;
      default: throw new IllegalArgumentException("Unknow transfer function type: " + type);
    }

    return derivative;
  }
}
