#ifndef NET_H_
#define NET_H_

#include "Neuron.h"

#include <string>
#include <vector>

using uint = unsigned int;
class Neuron;
typedef std::vector<Neuron> Layer;

class Net {
 public:
  explicit Net(const std::string &load_file, double eta, double alpha);
  explicit Net(const std::vector<uint> &topology, double eta, double alpha);
  void feed_forward(const std::vector<double> &input);
  void back_prop(const std::vector<double> &target);
  void get_result(std::vector<double> *result) const;
  // [[nodiscard]] const double &avg_error() const { return m_recentAverageError; }
  [[nodiscard]] const double &error() const { return _error; }
  void save(const std::string &save_file);
  void load(const std::string &load_file);

 private:
  std::vector<Layer> _net; ///< list of layers _net[layer][neuron]
  double _error;
  std::vector<uint> _topology;
  double m_recentAverageError;
  static double m_recentAverageSmoothingFactor;
};

#endif // NET_H_
