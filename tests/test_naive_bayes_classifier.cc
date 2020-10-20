#include <core/naive_bayes_classifier.h>
#include <core/naive_bayes_model.h>

#include <catch2/catch.hpp>
#include <iostream>

using namespace naivebayes;

TEST_CASE("Classifier can predict with >70% accuracy") {
  NaiveBayesModel model;
  std::ifstream load_file("data/mnistdata_save_model");
  load_file >> model;

  ImageDataset test_images;
  LabelDataset test_labels;
  std::ifstream image_stream("data/mnistdatavalidation/testimages");
  std::ifstream label_stream("data/mnistdatavalidation/testlabels");
  image_stream >> test_images;
  label_stream >> test_labels;

  size_t num_correct = 0;
  naivebayes::NaiveBayesClassifier classifier(model);
  for (size_t i = 0; i < test_images.images_.size(); i++) {
    if (classifier.Classify(test_images.images_[i]) == test_labels.labels_[i]) {
      num_correct++;
    }
  }
  double accuracy = (double)num_correct / test_images.images_.size() * 100.0;

  REQUIRE(accuracy > 70.0);
}

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

  NaiveBayesClassifier classifier(model);

  SECTION("Test likelihood score that class = 0") {
    REQUIRE(classifier.GetLikelihoodScore(0, image) == Approx(-3.788941));
  }
  SECTION("Test likelihood score that class = 1") {
    REQUIRE(classifier.GetLikelihoodScore(1, image) == Approx(-1.874571));
  }
  SECTION("Test that classifier outputs most likely digit based on scores") {
    REQUIRE(classifier.Classify(image) == 1);
  }
}

TEST_CASE("Classifier correctly computes on a different image size") {
  ImageDataset image_dataset;
  LabelDataset label_dataset;
  std::ifstream test_image_stream("tests/data/test_data_5x5/test_images");
  std::ifstream test_label_stream("tests/data/test_data_5x5/test_labels");

  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

  NaiveBayesModel model(image_dataset, label_dataset);
  model.Train(1.0);

  Image image = {{'#', '+', '+', ' ', ' '},
                 {' ', ' ', '#', ' ', ' '},
                 {' ', ' ', '#', ' ', ' '},
                 {' ', ' ', '+', ' ', ' '},
                 {' ', ' ', '#', ' ', ' '}};

  NaiveBayesClassifier classifier(model);

  SECTION("Test likelihood score that class = 0") {
    REQUIRE(classifier.GetLikelihoodScore(0, image) == Approx(-9.315671));
  }
  SECTION("Test likelihood score that class = 1") {
    REQUIRE(classifier.GetLikelihoodScore(1, image) == Approx(-3.697500));
  }
  SECTION("Test that classifier outputs most likely digit based on scores") {
    REQUIRE(classifier.Classify(image) == 1);
  }
}