#pragma once
#include <core/training_image.h>

#include <string>
#include <unordered_map>

#include "image_dataset.h"
#include "label_dataset.h"

namespace naivebayes {

class NaiveBayesTrainer {
 public:
  NaiveBayesTrainer(ImageDataset images, LabelDataset labels);

  void Train();
  size_t GetImageDatasetSize();
  double GetProbabilityClassEquals(size_t c);
  double GetProbabilityPixelEqualsGivenClass(size_t row, size_t col,
                                             bool is_shaded,
                                             size_t class_given);

 private:
  std::vector<TrainingImage> training_images_;
  size_t training_image_size;
  std::unordered_map<size_t, double> probability_class_equals_;
  std::unordered_map<
      size_t,
      std::unordered_map<
          size_t, std::unordered_map<bool, std::unordered_map<size_t, double>>>>
      probability_pixel_equals_given_class_;

  void ComputeProbabilitiesClassEquals(size_t laplace_k);
  void ComputeProbabilitiesPixelEqualsGivenClass(size_t laplace_k);
};

}  // namespace naivebayes