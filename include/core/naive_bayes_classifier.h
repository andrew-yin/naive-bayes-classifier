#pragma once

#include "naive_bayes_model.h"

namespace naivebayes {

class NaiveBayesClassifier {
 public:
  NaiveBayesClassifier(const NaiveBayesModel& model);
  size_t Classify(const Image& image);
  double GetLikelihoodScore(const size_t& digit, const Image& image);

 private:
  NaiveBayesModel model_;

  std::unordered_map<size_t, double> ComputeLikelihoodScores(
      const Image& image);
};

}  // namespace naivebayes