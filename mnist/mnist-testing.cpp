#include "Net.h"
#include "mnist.hpp"

#include <algorithm>
#include <catch2/internal/catch_to_string.hpp>
#include <fstream>
#include <iostream>
#include <random>

int main() {
  std::string TEST_IMAGES_DIR = "../../mnist/mnist_pgm/testing/";
  std::string LOAD_CONFIG_FILE = "../../mnist/mnist5.csv"; // config to load
  std::string MISTAKES_FILE = "../../mnist/mistakes.txt"; // track errors

  Net mnistNet(std::string{LOAD_CONFIG_FILE}, 0, 0);

  std::vector<double> input;  // (28x28 pgm)
  std::vector<double> target; // target results
  std::vector<double> result; // ann output

  double correct{};                 // count correct responses
  double incorrect{};               // count incorrect responses
  std::ofstream ofs(MISTAKES_FILE); // track mistakes

  std::vector<std::vector<uint>> dirs{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
  std::vector<std::pair<uint, uint>> data;
  std::random_device rd;
  std::mt19937 mt(rd());

  for (uint i = 0; i < 10; ++i) { // find all the files
    for (uint j = 1; j < 60'001; ++j) {
      std::string inFile = TEST_IMAGES_DIR + std::to_string(i) + "/" +
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

  for (const auto &[directory, file] : data) { // for every file in the list
    std::string image_file = TEST_IMAGES_DIR + std::to_string(directory) + "/" +
                             std::to_string(file) + ".pgm";

    read_pixels(&image_file, &input); // set the input
    set_target(directory, target);    // set the target value
    mnistNet.feed_forward(input);     // input to net
    mnistNet.get_result(&result);     // get output from net

    if (auto ann_response = translate(result),
        correct_answer = translate(target);
        ann_response == correct_answer) { // check correctness
      correct++;
    } else {
      incorrect++;
      std::string mistake = std::to_string(correct_answer) + "\t" +
                            std::to_string(ann_response) + "\t" + image_file +
                            "\n";
      std::cout << mistake; // display on console
      ofs << mistake;       // write to mistake file
    }
  }
  std::cout << "Score: " << correct / (correct + incorrect);
  return 0;
}
