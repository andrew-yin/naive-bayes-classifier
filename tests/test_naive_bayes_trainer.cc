#include <core/naive_bayes_trainer.h>

#include <catch2/catch.hpp>
#include <iostream>

naivebayes::ImageDataset image_dataset;
naivebayes::LabelDataset label_dataset;

std::ifstream training_images("data/test_data/test_images");
std::ifstream training_labels("data/test_data/test_labels");

//TODO: create more specific tests
TEST_CASE("Input stream >> operator overloading") {
  training_images >> image_dataset;
  training_labels >> label_dataset;

  naivebayes::NaiveBayesTrainer trainer(image_dataset, label_dataset);
  trainer.Train();

  REQUIRE(trainer.GetImageDatasetSize() == 3);
}

TEST_CASE("Trainer class computes P(class = c) successfully") {
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

TEST_CASE("Trainer class computes P(F_(i,j) = f | class = c) successfully") {
  training_images >> image_dataset;
  training_labels >> label_dataset;

  naivebayes::NaiveBayesTrainer trainer(image_dataset, label_dataset);
  trainer.Train();

  SECTION("Test P(F(0,0) = 0 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 0, 0, 0) ==
            Approx(0.333333));
  }
  SECTION("Test P(F(0,1) = 0 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 1, 0, 0) ==
            Approx(0.333333));
  }
  SECTION("Test P(F(0,2) = 0 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 2, 0, 0) ==
            Approx(0.333333));
  }
  SECTION("Test P(F(1,0) = 0 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 0, 0, 0) ==
            Approx(0.333333));
  }
  SECTION("Test P(F(1,1) = 0 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 1, 0, 0) ==
            Approx(0.666667));
  }
  SECTION("Test P(F(1,2) = 0 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 2, 0, 0) ==
            Approx(0.333333));
  }
  SECTION("Test P(F(2,0) = 0 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 0, 0, 0) ==
            Approx(0.333333));
  }
  SECTION("Test P(F(2,1) = 0 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 1, 0, 0) ==
            Approx(0.333333));
  }
  SECTION("Test P(F(2,2) = 0 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 2, 0, 0) ==
            Approx(0.333333));
  }

  SECTION("Test P(F(0,0) = 1 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 0, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(0,1) = 1 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 1, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(0,2) = 1 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 2, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(1,0) = 1 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 0, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(1,1) = 1 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 1, 1, 0) ==
        Approx(0.333333));
  }
  SECTION("Test P(F(1,2) = 1 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 2, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(2,0) = 1 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 0, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(2,1) = 1 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 1, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(2,2) = 1 | class = 0)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 2, 1, 0) ==
        Approx(0.666667));
  }

  SECTION("Test P(F(0,0) = 0 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 0, 0, 1) ==
        Approx(0.5));
  }
  SECTION("Test P(F(0,1) = 0 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 1, 0, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(0,2) = 0 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 2, 0, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(1,0) = 0 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 0, 0, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(1,1) = 0 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 1, 0, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(1,2) = 0 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 2, 0, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(2,0) = 0 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 0, 0, 1) ==
        Approx(0.5));
  }
  SECTION("Test P(F(2,1) = 0 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 1, 0, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(2,2) = 0 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 2, 0, 1) ==
        Approx(0.5));
  }


  SECTION("Test P(F(0,0) = 1 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 0, 1, 1) ==
        Approx(0.5));
  }
  SECTION("Test P(F(0,1) = 1 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 1, 1, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(0,2) = 1 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(0, 2, 1, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(1,0) = 1 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 0, 1, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(1,1) = 1 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 1, 1, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(1,2) = 1 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(1, 2, 1, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(2,0) = 1 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 0, 1, 1) ==
        Approx(0.5));
  }
  SECTION("Test P(F(2,1) = 1 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 1, 1, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(2,2) = 1 | class = 1)") {
    REQUIRE(trainer.GetProbabilityPixelEqualsGivenClass(2, 2, 1, 1) ==
        Approx(0.5));
  }
}

TEST_CASE("Trainer class can save and load model to/from file") {
  training_images >> image_dataset;
  training_labels >> label_dataset;

  naivebayes::NaiveBayesTrainer trainer(image_dataset, label_dataset);
  trainer.Train();

  std::string save_file_path = "data/test_save";
  std::ofstream save_file(save_file_path);

  save_file << trainer;

  naivebayes::NaiveBayesTrainer trainer2;
  std::string load_file_path = "data/test_save";
  std::ifstream load_file(load_file_path);
  load_file >> trainer2;

  SECTION("Test P(class = 0)") {
    REQUIRE(trainer.GetProbabilityClassEquals(0) == Approx(0.4));
  }
  SECTION("Test P(class = 1)") {
    REQUIRE(trainer.GetProbabilityClassEquals(1) == Approx(0.6));
  }

  SECTION("Test P(F(0,0) = 0 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 0, 0, 0) ==
        Approx(0.333333));
  }
  SECTION("Test P(F(0,1) = 0 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 1, 0, 0) ==
        Approx(0.333333));
  }
  SECTION("Test P(F(0,2) = 0 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 2, 0, 0) ==
        Approx(0.333333));
  }
  SECTION("Test P(F(1,0) = 0 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 0, 0, 0) ==
        Approx(0.333333));
  }
  SECTION("Test P(F(1,1) = 0 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 1, 0, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(1,2) = 0 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 2, 0, 0) ==
        Approx(0.333333));
  }
  SECTION("Test P(F(2,0) = 0 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 0, 0, 0) ==
        Approx(0.333333));
  }
  SECTION("Test P(F(2,1) = 0 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 1, 0, 0) ==
        Approx(0.333333));
  }
  SECTION("Test P(F(2,2) = 0 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 2, 0, 0) ==
        Approx(0.333333));
  }

  SECTION("Test P(F(0,0) = 1 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 0, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(0,1) = 1 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 1, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(0,2) = 1 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 2, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(1,0) = 1 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 0, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(1,1) = 1 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 1, 1, 0) ==
        Approx(0.333333));
  }
  SECTION("Test P(F(1,2) = 1 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 2, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(2,0) = 1 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 0, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(2,1) = 1 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 1, 1, 0) ==
        Approx(0.666667));
  }
  SECTION("Test P(F(2,2) = 1 | class = 0)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 2, 1, 0) ==
        Approx(0.666667));
  }

  SECTION("Test P(F(0,0) = 0 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 0, 0, 1) ==
        Approx(0.5));
  }
  SECTION("Test P(F(0,1) = 0 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 1, 0, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(0,2) = 0 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 2, 0, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(1,0) = 0 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 0, 0, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(1,1) = 0 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 1, 0, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(1,2) = 0 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 2, 0, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(2,0) = 0 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 0, 0, 1) ==
        Approx(0.5));
  }
  SECTION("Test P(F(2,1) = 0 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 1, 0, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(2,2) = 0 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 2, 0, 1) ==
        Approx(0.5));
  }
  SECTION("Test P(F(0,0) = 1 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 0, 1, 1) ==
        Approx(0.5));
  }
  SECTION("Test P(F(0,1) = 1 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 1, 1, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(0,2) = 1 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(0, 2, 1, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(1,0) = 1 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 0, 1, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(1,1) = 1 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 1, 1, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(1,2) = 1 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(1, 2, 1, 1) ==
        Approx(0.25));
  }
  SECTION("Test P(F(2,0) = 1 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 0, 1, 1) ==
        Approx(0.5));
  }
  SECTION("Test P(F(2,1) = 1 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 1, 1, 1) ==
        Approx(0.75));
  }
  SECTION("Test P(F(2,2) = 1 | class = 1)") {
    REQUIRE(trainer2.GetProbabilityPixelEqualsGivenClass(2, 2, 1, 1) ==
        Approx(0.5));
  }
}