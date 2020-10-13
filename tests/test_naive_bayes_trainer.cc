#include <core/naive_bayes_trainer.h>

#include <catch2/catch.hpp>
#include <iostream>

TEST_CASE("Input stream >> operator overloading") {
  naivebayes::ImageDataset image_dataset;
  naivebayes::LabelDataset label_dataset;

  std::ifstream test_image_stream("tests/data/test_data_3x3/test_images");
  std::ifstream test_label_stream("tests/data/test_data_3x3/test_labels");

  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

  naivebayes::Image image1 = {
      {'#', '+', ' '}, {' ', '+', ' '}, {'+', '+', '#'}};

  naivebayes::Image image2 = {
      {'#', '+', '#'}, {'+', ' ', '+'}, {'#', '#', '+'}};

  naivebayes::Image image3 = {
      {' ', '+', ' '}, {' ', '#', ' '}, {' ', '+', ' '}};

  std::vector<naivebayes::Image> images = {image1, image2, image3};
  std::vector<size_t> labels = {1, 0, 1};

  SECTION("Test that image contents are correct") {
    REQUIRE(images == image_dataset.images_);
  }
  SECTION("Test that label contents are correct") {
    REQUIRE(labels == label_dataset.labels_);
  }
}

TEST_CASE("Trainer class computes P(class = c) successfully") {
  naivebayes::ImageDataset image_dataset;
  naivebayes::LabelDataset label_dataset;

  std::ifstream test_image_stream("tests/data/test_data_3x3/test_images");
  std::ifstream test_label_stream("tests/data/test_data_3x3/test_labels");

  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

  naivebayes::NaiveBayesTrainer trainer(image_dataset, label_dataset);
  trainer.Train();

  SECTION("Test P(class = 0)") {
    REQUIRE(trainer.GetProbabilityClassEquals(0) == Approx(0.4));
  }
  SECTION("Test P(class = 1)") {
    REQUIRE(trainer.GetProbabilityClassEquals(1) == Approx(0.6));
  }
}

TEST_CASE("Image and Label dataset structs can handle dynamic image sizes") {
  std::ifstream test_image_stream("tests/data/test_data_5x5/test_images");
  std::ifstream test_label_stream("tests/data/test_data_5x5/test_labels");

  naivebayes::ImageDataset image_dataset;
  naivebayes::LabelDataset label_dataset;

  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

  naivebayes::Image image1 = {{'#', '+', '#', '+', '#'},
                              {'+', ' ', ' ', ' ', '+'},
                              {'+', ' ', ' ', ' ', '#'},
                              {'#', ' ', ' ', ' ', '#'},
                              {'+', '+', '+', '#', '#'}};

  naivebayes::Image image2 = {{' ', ' ', '+', ' ', ' '},
                              {' ', ' ', '#', ' ', ' '},
                              {' ', ' ', '#', ' ', ' '},
                              {' ', ' ', '+', ' ', ' '},
                              {' ', ' ', '#', ' ', ' '}};

  naivebayes::Image image3 = {{'+', '#', '+', ' ', ' '},
                              {' ', ' ', '#', ' ', ' '},
                              {' ', ' ', '+', ' ', ' '},
                              {' ', ' ', '#', ' ', ' '},
                              {' ', ' ', '#', ' ', ' '}};

  std::vector<naivebayes::Image> images = {image1, image2, image3};
  std::vector<size_t> labels = {0, 1, 1};

  SECTION("Test that image contents are correct") {
    REQUIRE(images == image_dataset.images_);
  }
  SECTION("Test that label contents are correct") {
    REQUIRE(labels == label_dataset.labels_);
  }
}

TEST_CASE("Trainer class computes P(F_(i,j) = f | class = c) successfully") {
  naivebayes::ImageDataset image_dataset;
  naivebayes::LabelDataset label_dataset;

  std::ifstream test_image_stream("tests/data/test_data_3x3/test_images");
  std::ifstream test_label_stream("tests/data/test_data_3x3/test_labels");

  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

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
  naivebayes::ImageDataset image_dataset;
  naivebayes::LabelDataset label_dataset;

  std::ifstream test_image_stream("tests/data/test_data_3x3/test_images");
  std::ifstream test_label_stream("tests/data/test_data_3x3/test_labels");

  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

  naivebayes::NaiveBayesTrainer trainer(image_dataset, label_dataset);
  trainer.Train();

  std::string save_file_path = "tests/data/test_save";
  std::ofstream save_file(save_file_path);

  save_file << trainer;

  naivebayes::NaiveBayesTrainer trainer2;
  std::string load_file_path = "tests/data/test_save";
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