#include "Net.h"
#include "Neuron.h"

#include <catch2/catch_all.hpp>
#include <random>

std::vector<uint> random_index(const uint size) {
  std::vector<uint> rand;
  rand.reserve(size);
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<uint> dist(0, 3);
  for (auto i = 0; i < size; ++i) {
    rand.push_back(dist(mt));
  }
  return rand;
}

TEST_CASE("xorNet-test") {
  Net xorNet(std::string{"../../data/xorNet.csv"}, 0, 0);
  const std::vector<std::vector<double>> input{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const std::vector<double> target{-1, 1, 1, -1};
  std::vector<double> result;
  for (std::vector<uint> rand =
           random_index(static_cast<uint>(std::pow(10, 3)));
       const auto &r : rand) {
    xorNet.feed_forward(input[r]);
    xorNet.get_result(&result);
    CHECK_THAT(result[0], Catch::Matchers::WithinRel(target[r], 0.1));
  }
}
