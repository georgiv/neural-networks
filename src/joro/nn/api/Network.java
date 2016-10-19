package joro.nn.api;

public interface Network {
  void learn();
  double[] process(double... inputs);
}
