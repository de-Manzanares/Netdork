#include "Neuron.h"
#include <cmath>
#include <random>

Neuron::Neuron(const uint connection_count, const uint my_index,
               const double eta = 0, const double alpha = 0)
    : _eta(eta), _alpha(alpha), _val(), _gradient() {
  for (uint i = 0; i < connection_count; ++i) {
    _synapse.emplace_back();
    _synapse.back().weight = randomWeight();
  }
  _my_index = my_index;
}

void Neuron::feed_forward(const Layer &previous_layer) {
  double sum{};
  for (const auto &n : previous_layer) {
    sum += n.val() * n._synapse[_my_index].weight;
  }
  _val = transfer_function(sum);
}

void Neuron::gradients_output(const double target) {
  _gradient = (target - _val) * transfer_function_derivative(_val);
}

void Neuron::gradients_hidden(const Layer &next_layer) {
  const double dow = sum_deltas_of_weights(next_layer);
  _gradient = dow * transfer_function_derivative(_val);
}

void Neuron::update_connection(Layer *previous_layer) const {
  // The weights to be updated are in the Connection container
  // in the neurons in the preceding layer

  for (auto &n : *previous_layer) {
    const double delta_weight_old = n._synapse[_my_index].delta_weight;

    const double delta_weight_new =
        // Individual input, magnified by the gradient and train rate:
        _eta * n.val() * _gradient
        // Also add momentum = a fraction of the previous delta weight;
        + _alpha * delta_weight_old;

    n._synapse[_my_index].delta_weight = delta_weight_new;
    n._synapse[_my_index].weight += delta_weight_new;
  }
}

double Neuron::transfer_function(const double x) { return std::tanh(x); }

double Neuron::transfer_function_derivative(const double x) {
  return std::pow(1 / std::cosh(x), 2);
}

double Neuron::sum_deltas_of_weights(const Layer &next_layer) const {
  double sum{};
  // Sum our contributions of the errors at the nodes we feed.
  for (uint n = 0; n < next_layer.size() - 1; ++n) {
    sum += _synapse[n].weight * next_layer[n]._gradient;
  }
  return sum;
}

double Neuron::randomWeight() {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution dist(0.0, 1.0);
  return dist(mt);
}
