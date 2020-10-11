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

 private:
  std::vector<TrainingImage> training_images_;
  std::unordered_map<size_t, double> probability_class_equals_;

  void ComputeProbabilitesClassEquals();
};

}  // namespace naivebayes