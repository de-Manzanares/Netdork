#ifndef NEURON_H_
#define NEURON_H_

#include <vector>

using uint = unsigned int;
class Neuron;
typedef std::vector<Neuron> Layer;

struct Synapse {
  double weight;
  double delta_weight;
};

class Neuron {
 public:
  Neuron(uint connection_count, uint my_index, double eta, double alpha);
  double &val() { return _val; }
  [[nodiscard]] double val() const { return _val; }
  void feed_forward(const Layer &previous_layer);
  void gradients_output(double target);
  void gradients_hidden(const Layer &next_layer);
  void update_connection(Layer *previous_layer) const;
  std::vector<Synapse> &synapse() { return _synapse; }
  [[nodiscard]] const std::vector<Synapse> &synapse() const { return _synapse; }

 private:
  static double transfer_function(double x);
  static double transfer_function_derivative(double x);
  static double randomWeight();
  [[nodiscard]] double sum_deltas_of_weights(const Layer &next_layer) const;
  double _eta;   ///< [0.0..1.0] overall net training rate
  double _alpha; ///< [0.0..n] multiplier of last weight change (momentum)
  double _val;  ///< the value this neruon feeds forward
  std::vector<Synapse> _synapse; ///< connections to next layer of neurons
  uint _my_index;                ///< this neuron's position in its layer
  double _gradient;
};

#endif // NEURON_H_
