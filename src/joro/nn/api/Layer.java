package joro.nn.api;

public interface Layer {
  void adjust(Neuron... neurons);
  double[] activate(double... input);
}
