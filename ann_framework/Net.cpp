#include "Net.h"
#include "Neuron.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

double Net::m_recentAverageSmoothingFactor =
    1000.0; // Number of training samples to average over

Net::Net(const std::vector<uint> &topology, const double eta = 0.0,
         const double alpha = 0.0)
    : _error(), _topology(topology), m_recentAverageError() {
  const uint layer_count = topology.size();
  for (uint l = 0; l < layer_count; ++l) {
    _net.emplace_back();
    const uint connection_count =
        l == topology.size() - 1 ? 0 : topology[l + 1];

    // We have a new layer, now fill it with neurons, and
    // add a bias neuron in each layer.
    for (uint n = 0; n <= topology[l]; ++n) {
      _net.back().emplace_back(connection_count, n, eta, alpha);
#ifdef DEBUG_PRINT
      std::cout << "Made a Neuron!" << std::endl;
#endif
    }
    // Force the bias node's output to 1.0 (it was the last neuron pushed in
    // this layer):
    _net.back().back().val() = 1.0;
  }
}

Net::Net(const std::string &load_file, const double eta, const double alpha)
    : _error(), m_recentAverageError() {
  std::ifstream file(load_file);
  if (!file.is_open()) {
    throw std::runtime_error("file not found " + load_file);
  }
  std::string s;
  std::string val;
  std::vector<double> weight;

  if (std::getline(file, s)) { // read topology
    std::vector<uint> topology;
    std::istringstream iss(s);
    while (std::getline(iss, val, ',')) {
      if (!val.empty()) {
        topology.push_back(static_cast<uint>(std::stoul(val)));
      }
    }
    _topology = topology;
  }
  const uint layer_count = _topology.size();
  for (uint l = 0; l < layer_count; ++l) { // make layers
    _net.emplace_back();
    const uint connection_count =
        l == _topology.size() - 1 ? 0 : _topology[l + 1];

    // We have a new layer, now fill it with neurons, and
    // add a bias neuron in each layer.
    for (uint n = 0; n <= _topology[l]; ++n) {
      _net.back().emplace_back(connection_count, n, eta, alpha); // lth layer
      weight.clear();
      if (std::getline(file, s)) { // read in weights
        std::istringstream iss(s);
        while (std::getline(iss, val, ',')) {
          if (!val.empty()) {
            weight.push_back(std::stod(val));
          }
        }
      }
      for (auto i = 0; i < _net[l][n].synapse().size(); ++i) {
        _net[l][n].synapse()[i].weight = weight[i];
      }
    }
    // Force the bias node's output to 1.0 (it was the last neuron pushed in
    // this layer):
    _net.back().back().val() = 1.0;
  }
}

void Net::feed_forward(const std::vector<double> &input) {
  assert(input.size() == _net[0].size() - 1);
  // Assign (latch) the input values into the input neurons
  for (uint i = 0; i < input.size(); ++i) {
    _net[0][i].val() = input[i];
  }
  // forward propagate
  for (uint l = 1; l < _net.size(); ++l) {
    for (uint n = 0; n < _net[l].size() - 1; ++n) {
      _net[l][n].feed_forward(_net[l - 1]);
    }
  }
}

void Net::back_prop(const std::vector<double> &target) {
  // Calculate overall net error (RMS of output neuron errors)

  Layer &output_layer = _net.back();
  _error = 0.0;

  // RMS
  for (uint n = 0; n < output_layer.size() - 1; ++n) {
    _error += std::pow(target[n] - output_layer[n].val(), 2);
  }
  _error /= static_cast<double>(output_layer.size() - 1);
  _error = sqrt(_error); // RMS

  // Implement a recent average measurement
  // m_recentAverageError =
  //     (m_recentAverageError * m_recentAverageSmoothingFactor + _error) /
  //     (m_recentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradients
  for (uint n = 0; n < output_layer.size() - 1; ++n) {
    output_layer[n].gradients_output(target[n]);
  }
  // Calculate hidden layer gradients
  for (uint l = _net.size() - 2; l > 0; --l) {
    for (uint n = 0; n < _net[l].size(); ++n) {
      _net[l][n].gradients_hidden(_net[l + 1]);
    }
  }
  // For all layers from outputs to first hidden layer,
  // update connection weights
  for (uint l = _net.size() - 1; l > 0; --l) {
    for (uint n = 0; n < _net[l].size() - 1; ++n) {
      _net[l][n].update_connection(&_net[l - 1]);
    }
  }
}

void Net::get_result(std::vector<double> *result) const {
  result->clear();
  const auto output = _net.back();
  for (uint n = 0; n < output.size() - 1; ++n) {
    result->push_back(output[n].val());
  }
}

void Net::save(const std::string &save_file) {
  std::ofstream file;
  file.open(save_file, std::ofstream::out);
  for (const auto &i : _topology) {
    file << i << ',';
  }
  file << '\n';
  for (const auto &l : _net) {
    for (const auto &n : l) {
      for (const auto &[weight, delta_weight] : n.synapse()) {
        file << weight << ',';
      }
      file << '\n';
    }
  }
  file.flush();
  file.close();
}
