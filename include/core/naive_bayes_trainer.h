#pragma once
#include <core/training_image.h>

#include <string>

#include "image_dataset.h"
#include "label_dataset.h"

namespace naivebayes {

class NaiveBayesTrainer {
 public:
  NaiveBayesTrainer(ImageDataset images, LabelDataset labels);

  size_t GetImageDatasetSize();

 private:
  std::vector<TrainingImage> training_images_;
};

}  // namespace naivebayes