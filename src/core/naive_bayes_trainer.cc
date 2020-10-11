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

void NaiveBayesTrainer::Train() {
  ComputeProbabilitesClassEquals();
}

size_t NaiveBayesTrainer::GetImageDatasetSize() {
  return training_images_.size();
}

void NaiveBayesTrainer::ComputeProbabilitesClassEquals() {
  std::unordered_map<size_t, size_t> class_counts;
  for (TrainingImage image: training_images_) {
    class_counts[image.GetLabel()]++;
  }

  size_t laplace_k = 1;
  size_t laplace_v = class_counts.size();
  size_t num_images = training_images_.size();
  for (auto const &count: class_counts) {
    probability_class_equals_[count.first] = (double) (laplace_k + count.second)/(laplace_v * laplace_k + num_images);
  }
}

double NaiveBayesTrainer::GetProbabilityClassEquals(size_t c) {
  return probability_class_equals_[c];
}

}  // namespace naivebayes