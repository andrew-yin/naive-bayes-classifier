#include <core/naive_bayes_trainer.h>
#include <catch2/catch.hpp>
#include <iostream>

TEST_CASE("Input stream >> operator overloading") {
  naivebayes::ImageDataset image_dataset;
  naivebayes::LabelDataset label_dataset;

  std::ifstream training_images ("data/mnistdatatraining/trainingimages");
  std::ifstream training_labels ("data/mnistdatatraining/traininglabels");

  training_images >> image_dataset;
  training_labels >> label_dataset;

  naivebayes::NaiveBayesTrainer trainer(image_dataset, label_dataset);

  REQUIRE(trainer.GetImageDatasetSize() == 5000);
}

TEST_CASE("Trainer class computes P(class = c) successfully") {
  naivebayes::ImageDataset image_dataset;
  naivebayes::LabelDataset label_dataset;

  std::ifstream training_images ("data/test_data/test_images");
  std::ifstream training_labels ("data/test_data/test_labels");

  training_images >> image_dataset;
  training_labels >> label_dataset;

  naivebayes::NaiveBayesTrainer trainer(image_dataset, label_dataset);

  trainer.Train();

  SECTION("Test P(class = 0)") {
    REQUIRE(trainer.GetProbabilityClassEquals(0) == Approx(0.4));
  }
  SECTION("Test P(class = 1)") {
    REQUIRE(trainer.GetProbabilityClassEquals(1) == Approx(0.6));
  }
}

