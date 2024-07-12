#include "Net.h"
#include "mnist.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>

#define N_START_NEW
#define Y_SMALL_SESSION

int main() {
  std::string TRAINING_IMAGES_DIR = "../../mnist/mnist_pgm/training/";
  std::string LOAD_CONFIG_FILE = "../../mnist/mnist5.csv"; // load config
  std::string SAVE_CONFIG_FILE = "../../mnist/mnist6.csv"; // save name

#ifdef Y_START_NEW // create a new net
  std::vector mnistNet{Net(std::vector<uint>{784, 64, 10}, 0.006000, 0.5)};
#endif
#ifdef N_START_NEW // continue training an existing net
  std::vector mnistNet{Net(std::string{LOAD_CONFIG_FILE}, 0.0000900, 0.5)};
#endif

  std::vector<double> error{};   // average error per 1000 images
  std::vector<double> s_error{}; // average error over the session
  std::vector<double> input;     // (28x28 pgm)
  std::vector<double> target;    // target results, for training
  uint image_count{};            // track the number of images we have used

  std::cout << '[';
  for (uint net = 0; net < mnistNet.size(); ++net) {
    std::vector<std::vector<uint>> dirs{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
    std::vector<std::pair<uint, uint>> data;
    std::random_device rd;
    std::mt19937 mt(rd());

    for (uint i = 0; i < 10; ++i) { // find all the training files
      for (uint j = 1; j < 60'001; ++j) {
        std::string inFile = TRAINING_IMAGES_DIR + std::to_string(i) + "/" +
                             std::to_string(j) + ".pgm";
        if (std::ifstream ifs(inFile); ifs.is_open()) {
          dirs[i].push_back(j);
        }
      }
    }
    for (uint dir = 0; dir < dirs.size(); ++dir) { // list files
      for (const auto &file : dirs[dir]) {
        data.emplace_back(dir, file);
      }
    }
    std::ranges::shuffle(data, mt); // shuffle file order

    for (const auto &[dir, file] : data) { // for every image in the list
      std::string inFile = TRAINING_IMAGES_DIR + std::to_string(dir) + "/" +
                           std::to_string(file) + ".pgm";

      read_pixels(&inFile, &input, image_count); // set the input
      set_target(dir, target);                   // set the target value
      mnistNet[net].feed_forward(input);         // input to net
      mnistNet[net].back_prop(target);           // correct the net
      error.push_back(mnistNet[net].error());
      if (image_count > 0 && image_count % 1000 == 0) { // track error rate
        auto val = std::accumulate(error.begin(), error.end(), 0.0) /
                   static_cast<double>(error.size());
        error.clear();
        std::cout << val << ',';
#ifdef Y_SMALL_SESSION
        s_error.push_back(val);
#endif
      }
    }
    mnistNet[net].save(SAVE_CONFIG_FILE); // save ann config

#ifdef N_SMALL_SESSION
    if (net < 8) {
      mnistNet.emplace_back(std::string{LOAD_CONFIG_FILE}, 0.00001, 0.5);
    }
#endif
  }
  std::cout << ']';
#ifdef Y_SMALL_SESSION
  std::cout << "\nSession error: "
            << std::accumulate(error.begin(), error.end(), 0.0) /
                   static_cast<double>(error.size());
#endif
  std::cout << '\n';
  std::cout << "Image count : " << image_count << '\n';

  return EXIT_SUCCESS;
}
