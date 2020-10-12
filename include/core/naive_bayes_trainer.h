#pragma once
#include <core/training_image.h>

#include <string>
#include <unordered_map>

#include "image_dataset.h"
#include "label_dataset.h"

namespace naivebayes {

class NaiveBayesTrainer {
 public:
  NaiveBayesTrainer();
  NaiveBayesTrainer(const ImageDataset &images, const LabelDataset &labels);

  void Train();
  double GetProbabilityClassEquals(const size_t &c) const;
  double GetProbabilityPixelEqualsGivenClass(const size_t &row,
                                             const size_t &col,
                                             const bool &is_shaded,
                                             const size_t &class_given) const;
  size_t GetImageDatasetSize() const;

  friend std::istream &operator>>(std::istream &in, NaiveBayesTrainer &trainer);
  friend std::ostream &operator<<(std::ostream &out,
                                  NaiveBayesTrainer &trainer);

 private:
  std::vector<TrainingImage> training_images_;
  size_t training_image_size_;
  std::unordered_map<size_t, double> probability_class_equals_;
  std::unordered_map<
      size_t,
      std::unordered_map<
          size_t, std::unordered_map<bool, std::unordered_map<size_t, double>>>>
      probability_pixel_equals_given_class_;

  void ComputeProbabilitiesClassEquals(const size_t &laplace_k);
  void ComputeProbabilitiesPixelEqualsGivenClass(const size_t &laplace_k);
};

}  // namespace naivebayes