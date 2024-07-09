#include "../include/Net.h"

#include <fstream>
#include <iostream>
#include <sstream>

#define VERBOSE_Y

int main() {

  std::vector xorNet{Net(std::vector<uint>{2, 4, 4, 1}, 0.1, 0.5)};

  std::vector<double> input;
  std::vector<double> target;
  std::vector<double> result;

#ifdef VERBOSE_Y
  std::cout << '[';
#endif

  for (auto i = 0; i < xorNet.size(); ++i) {
    std::string s;
    std::ifstream ifs("../data/xor.txt");
    if (!ifs.is_open()) {
      std::cerr << "FILE NOT FOUND!\n";
    }
    while (std::getline(ifs, s)) {
      std::istringstream iss(s);
      double i0, i1, t0;
      input.clear();
      target.clear();
      iss >> i0 >> i1 >> t0;
      input.push_back(i0);
      input.push_back(i1);
      target.push_back(t0);

      xorNet[i].feed_forward(input);
      xorNet[i].back_prop(target);
      xorNet[i].get_result(&result);
#ifdef VERBOSE_Y
      std::cout << xorNet[i].error() << ',';
#endif
    }
    ifs.close();
    xorNet[i].save("../data/xorNet.csv");

    switch (i) {
    case 0:
    case 1:
      xorNet.emplace_back(std::string{"../data/xorNet.csv"}, 0.1, 0.5);
      break;
    default:;
    }
  }

#ifdef VERBOSE_Y
  std::cout << ']' << '\n';
  std::cout << xorNet.back().avg_error() << std::endl;
#endif

  return 0;
}
