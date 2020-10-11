#include <core/naive_bayes_trainer.h>

#include <vector>

namespace naivebayes {

NaiveBayesTrainer::NaiveBayesTrainer(ImageDataset images, LabelDataset labels) {
  if (images.GetDatasetSize() == labels.GetDatasetSize()) {
    for (size_t i = 0; i < images.GetDatasetSize(); i++) {
      training_images_.push_back(
          TrainingImage(images.GetImage(i), labels.GetLabel(i)));
    }
  }
}

size_t NaiveBayesTrainer::GetImageDatasetSize() {
  return training_images_.size();
}

}  // namespace naivebayes