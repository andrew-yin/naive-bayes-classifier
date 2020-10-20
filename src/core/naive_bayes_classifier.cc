#include "core/naive_bayes_classifier.h"

#include <float.h>

namespace naivebayes {

NaiveBayesClassifier::NaiveBayesClassifier() = default;

NaiveBayesClassifier::NaiveBayesClassifier(const NaiveBayesModel& model)
    : model_(model) {
}

size_t NaiveBayesClassifier::Classify(const Image& image) {
  std::unordered_map<size_t, double> likelihood_scores =
      ComputeLikelihoodScores(image);
  size_t most_likely_digit = -1;
  double highest_likelihood_score = -DBL_MAX;
  for (const auto &i : likelihood_scores) {
    if (highest_likelihood_score < i.second) {
      most_likely_digit = i.first;
      highest_likelihood_score = i.second;
    }
  }
  return most_likely_digit;
}

void NaiveBayesClassifier::SetModel(const NaiveBayesModel& model) {
  model_ = model;
}

double NaiveBayesClassifier::GetLikelihoodScore(const size_t& digit,
                                                const Image& image) {
  std::unordered_map<size_t, double> likelihood_scores =
      ComputeLikelihoodScores(image);
  return likelihood_scores[digit];
}

std::unordered_map<size_t, double>
NaiveBayesClassifier::ComputeLikelihoodScores(const Image& image) {
  std::unordered_map<size_t, double> likelihood_scores;
  for (size_t& digit : model_.GetClasses()) {
    double likelihood_score = log10(model_.GetClassProbability(digit));
    for (size_t row = 0; row < image.size(); row++) {
      for (size_t col = 0; col < image.size(); col++) {
        bool is_shaded = image[row][col] != ' ';
        likelihood_score +=
            log10(model_.GetPixelProbability(row, col, is_shaded, digit));
      }
    }
    likelihood_scores[digit] = likelihood_score;
  }

  return likelihood_scores;
}

}  // namespace naivebayes