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
   */
  void Train();

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
   * @return The value of P(F(row, col) is/isn't shaded | class = c)
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

  /**
   * Determines all possible values of P(class = c)
   * @param laplace_k The Laplace smoothing k coefficient to be used
   */
  void DetermineClassProbabilities(const size_t &laplace_k);

  /**
   * Determines all possible values of
   * P(F(row, col) is/isn't shaded | class = c)
   * @param laplace_k The Laplace smoothing k coefficient to be used
   */
  void DeterminePixelProbabilities(const size_t &laplace_k);
};

}  // namespace naivebayes