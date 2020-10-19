#include <core/naive_bayes_model.h>
#include <core/naive_bayes_classifier.h>

#include <catch2/catch.hpp>
#include <iostream>

using namespace naivebayes;

TEST_CASE("Classifier can correctly compute likelihood scores") {
  ImageDataset image_dataset;
  LabelDataset label_dataset;
  std::ifstream test_image_stream("tests/data/test_data_3x3/test_images");
  std::ifstream test_label_stream("tests/data/test_data_3x3/test_labels");

  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

  NaiveBayesModel model(image_dataset, label_dataset);
  model.Train(1.0);

  Image image = {{'#', '+', ' '}, {' ', '+', ' '}, {' ', '+', ' '}};

  NaiveBayesClassifier classifier(model, image);
  size_t digit_predicted = classifier.Classify();

  SECTION("Test likelihood score that class = 0") {
    REQUIRE(classifier.GetLikelihoodScore(0) == Approx(-3.788941));
  }
  SECTION("Test likelihood score that class = 1") {
    REQUIRE(classifier.GetLikelihoodScore(1) == Approx(-1.874571));
  }
  SECTION("Test that classifier outputs most likely digit based on scores") {
    REQUIRE(digit_predicted == 1);
  }
}