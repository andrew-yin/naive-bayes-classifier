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