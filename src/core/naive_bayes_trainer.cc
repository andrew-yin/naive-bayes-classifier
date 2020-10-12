#include <core/naive_bayes_trainer.h>

#include <iostream>
#include <vector>

namespace naivebayes {

NaiveBayesTrainer::NaiveBayesTrainer() = default;

NaiveBayesTrainer::NaiveBayesTrainer(const ImageDataset &images,
                                     const LabelDataset &labels) {
  if (images.GetDatasetSize() == labels.GetDatasetSize()) {
    for (size_t i = 0; i < images.GetDatasetSize(); i++) {
      training_images_.push_back(
          TrainingImage(images.GetImage(i), labels.GetLabel(i)));
    }
  }
  training_image_size_ = training_images_.at(0).GetImageSize();
}

void NaiveBayesTrainer::Train() {
  size_t laplace_k = 1;
  ComputeProbabilitiesClassEquals(laplace_k);
  ComputeProbabilitiesPixelEqualsGivenClass(laplace_k);
}

std::ostream &operator<<(std::ostream &out, NaiveBayesTrainer &trainer) {
  out << trainer.training_image_size_ << std::endl;

  out << trainer.probability_class_equals_.size() << std::endl;
  for (auto const &class_probability : trainer.probability_class_equals_) {
    out << class_probability.first << std::endl;
    out << class_probability.second << std::endl;
  }

  size_t num_prob = trainer.probability_pixel_equals_given_class_.size() *
                    trainer.training_image_size_ *
                    trainer.training_image_size_ * 2 *
                    2;  // TODO: fix magic numbers
  out << num_prob << std::endl;
  for (auto const &i : trainer.probability_pixel_equals_given_class_) {
    size_t row = i.first;
    for (auto const &j : i.second) {
      size_t col = j.first;
      for (auto const &k : j.second) {
        bool is_shaded = k.first;
        for (auto const &l : k.second) {
          size_t c = l.first;
          out << row << std::endl;
          out << col << std::endl;
          out << is_shaded << std::endl;
          out << c << std::endl;
          out << trainer.probability_pixel_equals_given_class_[row][col]
                                                              [is_shaded][c]
              << std::endl;
        }
      }
    }
  }

  return out;
}

std::istream &operator>>(std::istream &in, NaiveBayesTrainer &trainer) {
  in >> trainer.training_image_size_;

  size_t num_classes;
  in >> num_classes;
  for (size_t i = 0; i < num_classes; i++) {
    size_t c;
    double prob;
    in >> c >> prob;
    trainer.probability_class_equals_[c] = prob;
  }

  size_t num_images;
  in >> num_images;
  for (size_t i = 0; i < num_images; i++) {
    size_t row, col, c;
    bool is_shaded;
    double prob;
    in >> row >> col >> is_shaded >> c >> prob;
    trainer.probability_pixel_equals_given_class_[row][col][is_shaded][c] =
        prob;
  }

  return in;
}

void NaiveBayesTrainer::ComputeProbabilitiesClassEquals(
    const size_t &laplace_k) {
  std::unordered_map<size_t, size_t> class_counts;
  for (TrainingImage &image : training_images_) {
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
  for (TrainingImage &image : training_images_) {
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
        for (size_t row = 0; row < training_image_size_; row++) {
          for (size_t col = 0; col < training_image_size_; col++) {
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
  return probability_class_equals_.at(c);
}

double NaiveBayesTrainer::GetProbabilityPixelEqualsGivenClass(
    const size_t &row, const size_t &col, const bool &is_shaded,
    const size_t &class_given) const {
  return probability_pixel_equals_given_class_.at(row).at(col).at(is_shaded).at(
      class_given);
}

size_t NaiveBayesTrainer::GetImageDatasetSize() const {
  return training_images_.size();
}

}  // namespace naivebayes