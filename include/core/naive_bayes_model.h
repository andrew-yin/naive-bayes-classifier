#pragma once
#include <core/training_image.h>

#include <string>
#include <unordered_map>

#include "image_dataset.h"
#include "label_dataset.h"

namespace naivebayes {

class NaiveBayesModel {
 public:
  NaiveBayesModel();
  NaiveBayesModel(const ImageDataset &images, const LabelDataset &labels);

  void Train();
  double GetClassProbability(const size_t &c) const;
  double GetPixelProbability(const size_t &row, const size_t &col,
                             const bool &is_shaded,
                             const size_t &class_given) const;

  friend std::istream &operator>>(std::istream &in, NaiveBayesModel &trainer);
  friend std::ostream &operator<<(std::ostream &out, NaiveBayesModel &trainer);

 private:
  std::vector<TrainingImage> training_images_;
  size_t training_image_size_;
  std::unordered_map<size_t, double> class_probabilities_;
  std::unordered_map<
      size_t,
      std::unordered_map<
          size_t, std::unordered_map<bool, std::unordered_map<size_t, double>>>>
      pixel_probabilities_;

  void DetermineClassProbabilities(const size_t &laplace_k);
  void DeterminePixelProbabilities(const size_t &laplace_k);
};

}  // namespace naivebayes