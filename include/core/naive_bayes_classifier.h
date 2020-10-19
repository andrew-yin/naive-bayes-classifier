#pragma once

#include "naive_bayes_model.h"

namespace naivebayes {

class NaiveBayesClassifier {
 public:
  NaiveBayesClassifier(const NaiveBayesModel& model, const Image& image);
  size_t Classify();
  double GetLikelihoodScore(size_t digit);
 private:
  NaiveBayesModel model_;
  Image image_;
  std::unordered_map<size_t, double> likelihood_scores_;

  void ComputeLikelihoodScores();
};

}  // namespace naivebayes