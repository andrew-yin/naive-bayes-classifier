#include <core/naive_bayes_trainer.h>

#include <vector>

namespace naivebayes {

NaiveBayesTrainer::NaiveBayesTrainer(const ImageDataset &images,
                                     const LabelDataset &labels) {
  if (images.GetDatasetSize() == labels.GetDatasetSize()) {
    for (size_t i = 0; i < images.GetDatasetSize(); i++) {
      training_images_.push_back(
          TrainingImage(images.GetImage(i), labels.GetLabel(i)));
    }
  }
  training_image_size = training_images_.at(0).GetImageSize();
}

void NaiveBayesTrainer::Train() {
  size_t laplace_k = 1;
  ComputeProbabilitiesClassEquals(laplace_k);
  ComputeProbabilitiesPixelEqualsGivenClass(laplace_k);
}

size_t NaiveBayesTrainer::GetImageDatasetSize() const {
  return training_images_.size();
}

void NaiveBayesTrainer::ComputeProbabilitiesClassEquals(
    const size_t &laplace_k) {
  std::unordered_map<size_t, size_t> class_counts;
  for (TrainingImage image : training_images_) {
    class_counts[image.GetLabel()]++;
  }

  size_t laplace_v = class_counts.size();
  size_t num_images = training_images_.size();
  for (auto const &count : class_counts) {
    probability_class_equals_[count.first] =
        (double)(laplace_k + count.second) /
        (laplace_v * laplace_k + num_images);
  }
}

void NaiveBayesTrainer::ComputeProbabilitiesPixelEqualsGivenClass(
    const size_t &laplace_k) {
  std::unordered_map<size_t, size_t> class_counts;
  for (TrainingImage image : training_images_) {
    class_counts[image.GetLabel()]++;
  }

  std::unordered_map<
      size_t,
      std::unordered_map<
          size_t, std::unordered_map<bool, std::unordered_map<size_t, size_t>>>>
      pixel_equals_given_class_counts;
  for (auto const &i : class_counts) {
    size_t c = i.first;

    for (TrainingImage &image : training_images_) {
      if (image.GetLabel() == c) {
        for (size_t row = 0; row < training_image_size; row++) {
          for (size_t col = 0; col < training_image_size; col++) {
            if (image.IsShaded(row, col)) {
              pixel_equals_given_class_counts[row][col][true][c]++;
              pixel_equals_given_class_counts[row][col][false][c] += 0;
            } else {
              pixel_equals_given_class_counts[row][col][false][c]++;
              pixel_equals_given_class_counts[row][col][true][c] += 0;
            }
          }
        }
      }
    }
  }

  size_t laplace_v = 2;
  for (auto const &i : pixel_equals_given_class_counts) {
    size_t row = i.first;
    for (auto const &j : i.second) {
      size_t col = j.first;
      for (auto const &k : j.second) {
        bool is_shaded = k.first;
        for (auto const &l : k.second) {
          size_t c = l.first;
          double prob = (double)(laplace_k + l.second) /
                        (laplace_k * laplace_v + class_counts[c]);
          probability_pixel_equals_given_class_[row][col][is_shaded][c] = prob;
        }
      }
    }
  }
}

double NaiveBayesTrainer::GetProbabilityClassEquals(const size_t &c) const {
  return probability_class_equals_[c];
}

double NaiveBayesTrainer::GetProbabilityPixelEqualsGivenClass(
    const size_t &row, const size_t &col, const bool &is_shaded,
    const size_t &class_given) const {
  return probability_pixel_equals_given_class_[row][col][is_shaded]
                                              [class_given];
}

}  // namespace naivebayes