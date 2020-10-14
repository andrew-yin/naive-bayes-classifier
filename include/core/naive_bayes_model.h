#pragma once
#include <core/training_image.h>

#include <string>
#include <unordered_map>

#include "image_dataset.h"
#include "label_dataset.h"

namespace naivebayes {

/**
 * Represents a naive bayes model for handwriting classification
 * Trains on data provided either from a dataset or from a loaded file
 */
class NaiveBayesModel {
 public:
  /**
   * Default constructor for when model is loaded from file
   */
  NaiveBayesModel();

  /**
   * Constructor for when model is loaded from a dataset
   *
   * @param images The dataset of training images
   * @param labels The dataset of training labels
   */
  NaiveBayesModel(const ImageDataset &images, const LabelDataset &labels);

  /**
   * Trains the model on the training dataset given
   *
   * @param laplace_k The k-value used for Laplace smoothing during training
   */
  void Train(const size_t &laplace_k);

  /**
   * Gets the probability a class is equal to 'c'
   *
   * @param c The class given whose probability is to be determined
   * @return The value of P(class = c)
   */
  double GetClassProbability(const size_t &c) const;

  /**
   * Gets the probability the pixel at (row, col) is/isn't shaded
   * given the class is equal to 'c'
   *
   * @param row The row the pixel is in
   * @param col The col the pixel is in
   * @param is_shaded True if the pixel must be shaded, false otherwise
   * @param class_given The given value of the class
   * @return The value of P(F(row, col) == is_shaded | class = c)
   */
  double GetPixelProbability(const size_t &row, const size_t &col,
                             const bool &is_shaded,
                             const size_t &class_given) const;

  /**
   * Operator overload of >> to load model from a file
   */
  friend std::istream &operator>>(std::istream &in, NaiveBayesModel &trainer);

  /**
   * Operator overload of << to save current model to a file
   */
  // TODO: make second param const
  friend std::ostream &operator<<(std::ostream &out, const NaiveBayesModel &trainer);

 private:
  /** Stores a vector of TrainingImages representing the training dataset */
  std::vector<TrainingImage> training_images_;

  /**
   * Key: 'c'
   * Value: P(class = c)
   */
  std::unordered_map<size_t, double> class_probabilities_;

  /**
   * Key(s): row, col, is_shaded, c
   * Value: P(F(row, col) == is_shaded | class = c)
   */
  // TODO: refactor maybe?
  std::unordered_map<
      size_t,
      std::unordered_map<
          size_t, std::unordered_map<bool, std::unordered_map<size_t, double>>>>
      pixel_probabilities_;

  /**
   * The pixel width of a training image
   */
  size_t training_image_width_;

  /**
   * Computes all possible values of P(class = c)
   *
   * @param laplace_k The Laplace smoothing k coefficient to be used
   * @param class_frequencies Key: the class of an image, Value: The frequency
   * that the class appears in the training set
   */
  void CalculateClassProbabilities(
      const double &laplace_k,
      const std::unordered_map<size_t, size_t> &class_frequencies);

  /**
   * Computes all possible values of
   * P(F(row, col) == is_shaded | class = c)
   *
   * @param laplace_k The Laplace smoothing k coefficient to be used
   * @param class_frequencies Key: the class of an image, Value: The frequency
   * that the class appears in the training set
   */
  void CalculatePixelProbabilities(
      const double &laplace_k,
      const std::unordered_map<size_t, size_t> &class_frequencies);
};

}  // namespace naivebayes