#include <core/naive_bayes_model.h>

#include <iostream>
#include <vector>

namespace naivebayes {

NaiveBayesModel::NaiveBayesModel() = default;

NaiveBayesModel::NaiveBayesModel(const ImageDataset &images,
                                 const LabelDataset &labels) {
  if (images.images_.size() == labels.labels_.size()) {
    for (size_t i = 0; i < images.images_.size(); i++) {
      training_images_.push_back(
          TrainingImage(images.images_[i], labels.labels_[i]));
    }
  }
  training_image_width_ = training_images_[0].image_size_;
}

void NaiveBayesModel::Train() {
  double laplace_k = 1.0;

  std::unordered_map<size_t, size_t> class_frequencies;
  for (TrainingImage &image : training_images_) {
    class_frequencies[image.label_]++;
  }

  CalculateClassProbabilities(laplace_k, class_frequencies);
  CalculatePixelProbabilities(laplace_k, class_frequencies);
}

std::ostream &operator<<(std::ostream &out, NaiveBayesModel &trainer) {
  out << trainer.training_image_width_ << std::endl;

  out << trainer.class_probabilities_.size() << std::endl;
  for (auto const &class_probability : trainer.class_probabilities_) {
    out << class_probability.first << std::endl;
    out << class_probability.second << std::endl;
  }

  out << trainer.num_pixel_probabilities << std::endl;
  for (auto const &i : trainer.pixel_probabilities_) {
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
          out << trainer.pixel_probabilities_[row][col][is_shaded][c]
              << std::endl;
        }
      }
    }
  }

  return out;
}

std::istream &operator>>(std::istream &in, NaiveBayesModel &trainer) {
  in >> trainer.training_image_width_;

  size_t num_class_probabilities;
  in >> num_class_probabilities;
  for (size_t i = 0; i < num_class_probabilities; i++) {
    size_t c;
    double prob;
    in >> c >> prob;
    trainer.class_probabilities_[c] = prob;
  }

  size_t num_pixel_probabilities;
  in >> num_pixel_probabilities;
  for (size_t i = 0; i < num_pixel_probabilities; i++) {
    size_t row, col, c;
    bool is_shaded;
    double prob;
    in >> row >> col >> is_shaded >> c >> prob;
    trainer.pixel_probabilities_[row][col][is_shaded][c] = prob;
  }

  return in;
}

void NaiveBayesModel::CalculateClassProbabilities(
    const double &laplace_k,
    const std::unordered_map<size_t, size_t> &class_frequencies) {
  size_t laplace_v = class_frequencies.size();
  size_t num_images = training_images_.size();
  for (auto const &count : class_frequencies) {
    class_probabilities_[count.first] = (laplace_k + count.second) /
                                        (laplace_v * laplace_k + num_images);
  }
}

void NaiveBayesModel::CalculatePixelProbabilities(
    const double &laplace_k,
    const std::unordered_map<size_t, size_t> &class_frequencies) {
  std::unordered_map<
      size_t,
      std::unordered_map<
          size_t, std::unordered_map<bool, std::unordered_map<size_t, size_t>>>>
      pixel_equals_given_class_counts;

  for (auto const &i : class_frequencies) {
    size_t c = i.first;
    for (TrainingImage &image : training_images_) {
      if (image.label_ == c) {
        for (size_t row = 0; row < training_image_width_; row++) {
          for (size_t col = 0; col < training_image_width_; col++) {
            if (image.IsShaded(row, col)) {
              // TODO: refactor?
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
          double prob = (laplace_k + l.second) /
                        (laplace_k * laplace_v + class_frequencies.at(c));
          pixel_probabilities_[row][col][is_shaded][c] = prob;
          num_pixel_probabilities++;
        }
      }
    }
  }
}

double NaiveBayesModel::GetClassProbability(const size_t &c) const {
  return class_probabilities_.at(c);
}

double NaiveBayesModel::GetPixelProbability(const size_t &row,
                                            const size_t &col,
                                            const bool &is_shaded,
                                            const size_t &class_given) const {
  return pixel_probabilities_.at(row).at(col).at(is_shaded).at(class_given);
}

}  // namespace naivebayes