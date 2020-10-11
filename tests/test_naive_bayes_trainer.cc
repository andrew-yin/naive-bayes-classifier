#include <core/naive_bayes_trainer.h>
#include <fstream>
#include <string>
#include <catch2/catch.hpp>


TEST_CASE("Input stream >> operator overloading") {
  naivebayes::NaiveBayesTrainer trainer;
  std::ifstream training_images ("../data/mnistdatatraining/trainingimages");
  training_images >> trainer;
  REQUIRE(trainer.GetImageDatasetSize() == 5000);
}