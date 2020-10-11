#include <core/naive_bayes_trainer.h>
#include <iostream>
#include <vector>

namespace naivebayes {

std::istream &operator>>(std::istream &in, NaiveBayesTrainer &trainer) {
  std::string line;
  while (getline(in, line)) {
    size_t image_size = line.size();
    std::vector<std::vector<char>> image_pixels(image_size);

    image_pixels.at(0) = std::vector<char>(line.begin(), line.end());
    for (size_t i = 1; i < image_size; i++) {
      getline(in, line);
      image_pixels.at(i) = std::vector<char>(line.begin(), line.end());
    }

    TrainingImage image(image_pixels);
    trainer.AddTrainingImage(image);
  }

  return in;
}

NaiveBayesTrainer::NaiveBayesTrainer() {
}

void NaiveBayesTrainer::AddTrainingImage(TrainingImage image) {
  training_images_.push_back(image);
}

size_t NaiveBayesTrainer::GetImageDatasetSize() {
  return training_images_.size();
}

}  // namespace naivebayes