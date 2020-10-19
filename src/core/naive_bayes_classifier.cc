#include "core/naive_bayes_classifier.h"

#include <float.h>

namespace naivebayes {

NaiveBayesClassifier::NaiveBayesClassifier(const NaiveBayesModel &model,
                                           const Image &image)
    : model_(model), image_(image) {
}

size_t NaiveBayesClassifier::Classify() {
  ComputeLikelihoodScores();
  size_t most_likely_digit = -1;
  double highest_likelihood_score = -DBL_MAX;
  for (const auto &i : likelihood_scores_) {
    if (highest_likelihood_score < i.second) {
      most_likely_digit = i.first;
      highest_likelihood_score = i.second;
    }
  }
  return most_likely_digit;
}

double NaiveBayesClassifier::GetLikelihoodScore(size_t digit) {
  return likelihood_scores_[digit];
}

void NaiveBayesClassifier::ComputeLikelihoodScores() {
  for (size_t &digit: model_.GetClasses()) {
    double likelihood_score = log10(model_.GetClassProbability(digit));
    for (size_t row = 0; row < image_.size(); row++) {
      for (size_t col = 0; col < image_.size(); col++) {
        bool is_shaded = image_[row][col] != ' ';
        likelihood_score +=
            log10(model_.GetPixelProbability(row, col, is_shaded, digit));
      }
    }
    likelihood_scores_[digit] = likelihood_score;
  }
}

}  // namespace naivebayes