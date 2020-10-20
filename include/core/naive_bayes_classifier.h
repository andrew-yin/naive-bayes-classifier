#pragma once

#include "naive_bayes_model.h"

namespace naivebayes {

class NaiveBayesClassifier {
 public:
  /**
   * Default constructor used for Cinder app
   */
  NaiveBayesClassifier();

  /**
   * Constructs a classifier from the model given
   *
   * @param model The model to be used for classification
   */
  NaiveBayesClassifier(const NaiveBayesModel& model);

  /**
   * Classifies the given image into a digit
   *
   * @param image The image to be classified
   * @return The digit the model predicts the image to be classified as
   */
  size_t Classify(const Image& image);

  /**
   * Returns the likelihood score of a digit on the following image
   *
   * @param digit The digit whose likelihood score is to be computed
   * @param image The image to be used
   * @return The likelihood score of the image for a digit
   */
  double GetLikelihoodScore(const size_t& digit, const Image& image);

  /**
   * Setter method used when classifier is initialized via default constructor
   *
   * @param model The model to be used for classification
   */
  void SetModel(const NaiveBayesModel& model);

 private:
  NaiveBayesModel model_;

  /**
   * Computes and returns a mapping from digit classes to their likelihoods
   *
   * @param image The image whose likelihood scores are to be calculate from
   * @return A map with keys as digits and values as their likelihood score
   */
  std::unordered_map<size_t, double> ComputeLikelihoodScores(
      const Image& image);
};

}  // namespace naivebayes