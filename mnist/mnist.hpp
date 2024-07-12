#ifndef MNIST_MNIST_HPP_
#define MNIST_MNIST_HPP_

#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Reads pixel values from a pgm into the input vector.
 * @param inFile The file to read from.
 * @param input To be mutated: a vector of 728 integers [0,255]. The pixel
 * values of the portable gray map (pgm) file.
 * @param image_count To be mutated: track the number ofimages we have trained
 * with.
 */
inline void read_pixels(const std::string *inFile, std::vector<double> *input,
                        uint *image_count) {
  if (std::ifstream ifs(*inFile); ifs.is_open()) {
    std::string s;

    getline(ifs, s); // stop the party if the file isn't a pgm
    assert(s == "P2");
    getline(ifs, s);
    assert(s == "28 28");
    getline(ifs, s);
    assert(s == "255");
    *image_count += 1;

    input->clear();
    while (getline(ifs, s)) { // read in values
      std::istringstream iss(s);
      double g;
      while (iss >> g) {
        input->push_back(g);
      }
    }
  } else {
    throw std::runtime_error("file not found " + *inFile);
  }
}

/**
 * Reads pixel values from a pgm into the input vector.
 * @param inFile The file to read from.
 * @param input To be mutated: a vector of 728 integers [0,255]. The pixel
 * values of the portable gray map (pgm) file.
 */
inline void read_pixels(const std::string *inFile, std::vector<double> *input) {
  uint dummy_image_count = 0;
  read_pixels(inFile, input, &dummy_image_count);
}

/**
 * The target is represented by a high, all others are low.
 * For example: \n
 * 0: {1, -1, -1, ... , -1} \n
 * 1: {-1, 1, -1, ... , -1} \n
 * 2: {-1, -1, 1, ... , -1} \n
 * @param directory Image fils have been categorized into directories. All 0's
 * are in the "0" directory, all 1's are in the "1" directory, etc.
 * @param target To be mutated. Represents the target value.
 */
inline void set_target(const uint directory, std::vector<double> *target) {
  *target = std::vector<double>(10, -1);
  target->at(directory) = 1;
}

/**
 * The number is represented by a high, all others are low.
 * For example: \n
 * 0: {1, -1, -1, ... , -1} \n
 * 1: {-1, 1, -1, ... , -1} \n
 * 2: {-1, -1, 1, ... , -1} \n
 * @tparam T type
 * @param v vector to translate
 * @return A number 0-9
 */
template <class T> uint translate(std::vector<T> v) {
  auto max = -10;
  auto index = 0;
  for (int i = 0; i < v.size(); ++i) {
    if (v[i] > max) {
      max = v[i];
      index = i;
    }
  }
  return index;
}

#endif // MNIST_MNIST_HPP_
