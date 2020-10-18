#include <core/naive_bayes_model.h>

#include <catch2/catch.hpp>
#include <iostream>

using namespace naivebayes;

/**
 * Initializes and trains a model on the test data
 * @param model The model that is to be initialized and trained for testing
 */
void InitializeAndTrainModel(NaiveBayesModel& model) {
  ImageDataset image_dataset;
  LabelDataset label_dataset;
  std::ifstream test_image_stream("tests/data/test_data_3x3/test_images");
  std::ifstream test_label_stream("tests/data/test_data_3x3/test_labels");
  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

  /* Create a model based on the dataset and begin training */
  NaiveBayesModel test_model(image_dataset, label_dataset);
  model = test_model;
}

TEST_CASE("Input stream >> operator overloading") {
  ImageDataset image_dataset;
  LabelDataset label_dataset;

  /* Read training dataset */
  std::ifstream test_image_stream("tests/data/test_data_3x3/test_images");
  std::ifstream test_label_stream("tests/data/test_data_3x3/test_labels");
  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

  /* Create expected image and label database */
  Image image1 = {{'#', '+', ' '}, {' ', '+', ' '}, {'+', '+', '#'}};

  Image image2 = {{'#', '+', '#'}, {'+', ' ', '+'}, {'#', '#', '+'}};

  Image image3 = {{' ', '+', ' '}, {' ', '#', ' '}, {' ', '+', ' '}};

  std::vector<Image> images = {image1, image2, image3};
  std::vector<size_t> labels = {1, 0, 1};

  SECTION("Test that image contents are correct") {
    REQUIRE(images == image_dataset.images_);
  }
  SECTION("Test that label contents are correct") {
    REQUIRE(labels == label_dataset.labels_);
  }
}

TEST_CASE("Image and Label dataset structs can handle dynamic image sizes") {
  /* Same testing process as above, using 5x5 data instead of 3x3 */
  std::ifstream test_image_stream("tests/data/test_data_5x5/test_images");
  std::ifstream test_label_stream("tests/data/test_data_5x5/test_labels");

  ImageDataset image_dataset;
  LabelDataset label_dataset;

  test_image_stream >> image_dataset;
  test_label_stream >> label_dataset;

  Image image1 = {{'#', '+', '#', '+', '#'},
                  {'+', ' ', ' ', ' ', '+'},
                  {'+', ' ', ' ', ' ', '#'},
                  {'#', ' ', ' ', ' ', '#'},
                  {'+', '+', '+', '#', '#'}};

  Image image2 = {{' ', ' ', '+', ' ', ' '},
                  {' ', ' ', '#', ' ', ' '},
                  {' ', ' ', '#', ' ', ' '},
                  {' ', ' ', '+', ' ', ' '},
                  {' ', ' ', '#', ' ', ' '}};

  Image image3 = {{'+', '#', '+', ' ', ' '},
                  {' ', ' ', '#', ' ', ' '},
                  {' ', ' ', '+', ' ', ' '},
                  {' ', ' ', '#', ' ', ' '},
                  {' ', ' ', '#', ' ', ' '}};

  std::vector<Image> images = {image1, image2, image3};
  std::vector<size_t> labels = {0, 1, 1};

  SECTION("Test that image contents are correct") {
    REQUIRE(images == image_dataset.images_);
  }
  SECTION("Test that label contents are correct") {
    REQUIRE(labels == label_dataset.labels_);
  }
}

TEST_CASE("Model class computes P(class = c) successfully") {
  NaiveBayesModel model;
  InitializeAndTrainModel(model);

  double laplace_k = 1.0;
  model.Train(laplace_k);

  SECTION("Test P(class = 0)") {
    REQUIRE(model.GetClassProbability(0) == Approx(0.4));
  }
  SECTION("Test P(class = 1)") {
    REQUIRE(model.GetClassProbability(1) == Approx(0.6));
  }
}

TEST_CASE("Model class computes P(F_(i,j) = f | class = c) successfully") {
  NaiveBayesModel model;
  InitializeAndTrainModel(model);

  double laplace_k = 1.0;
  model.Train(laplace_k);

  /*
   * Tests for unshaded pixels for given class label '0'
   */
  SECTION("Test P(F(0,0) = 0 | class = 0)") {
    REQUIRE(model.GetPixelProbability(0, 0, 0, 0) == Approx(0.333333));
  }
  SECTION("Test P(F(0,1) = 0 | class = 0)") {
    REQUIRE(model.GetPixelProbability(0, 1, 0, 0) == Approx(0.333333));
  }
  SECTION("Test P(F(0,2) = 0 | class = 0)") {
    REQUIRE(model.GetPixelProbability(0, 2, 0, 0) == Approx(0.333333));
  }
  SECTION("Test P(F(1,0) = 0 | class = 0)") {
    REQUIRE(model.GetPixelProbability(1, 0, 0, 0) == Approx(0.333333));
  }
  SECTION("Test P(F(1,1) = 0 | class = 0)") {
    REQUIRE(model.GetPixelProbability(1, 1, 0, 0) == Approx(0.666667));
  }
  SECTION("Test P(F(1,2) = 0 | class = 0)") {
    REQUIRE(model.GetPixelProbability(1, 2, 0, 0) == Approx(0.333333));
  }
  SECTION("Test P(F(2,0) = 0 | class = 0)") {
    REQUIRE(model.GetPixelProbability(2, 0, 0, 0) == Approx(0.333333));
  }
  SECTION("Test P(F(2,1) = 0 | class = 0)") {
    REQUIRE(model.GetPixelProbability(2, 1, 0, 0) == Approx(0.333333));
  }
  SECTION("Test P(F(2,2) = 0 | class = 0)") {
    REQUIRE(model.GetPixelProbability(2, 2, 0, 0) == Approx(0.333333));
  }

  /*
   * Tests for shaded pixels given class label '0'
   */
  SECTION("Test P(F(0,0) = 1 | class = 0)") {
    REQUIRE(model.GetPixelProbability(0, 0, 1, 0) == Approx(0.666667));
  }
  SECTION("Test P(F(0,1) = 1 | class = 0)") {
    REQUIRE(model.GetPixelProbability(0, 1, 1, 0) == Approx(0.666667));
  }
  SECTION("Test P(F(0,2) = 1 | class = 0)") {
    REQUIRE(model.GetPixelProbability(0, 2, 1, 0) == Approx(0.666667));
  }
  SECTION("Test P(F(1,0) = 1 | class = 0)") {
    REQUIRE(model.GetPixelProbability(1, 0, 1, 0) == Approx(0.666667));
  }
  SECTION("Test P(F(1,1) = 1 | class = 0)") {
    REQUIRE(model.GetPixelProbability(1, 1, 1, 0) == Approx(0.333333));
  }
  SECTION("Test P(F(1,2) = 1 | class = 0)") {
    REQUIRE(model.GetPixelProbability(1, 2, 1, 0) == Approx(0.666667));
  }
  SECTION("Test P(F(2,0) = 1 | class = 0)") {
    REQUIRE(model.GetPixelProbability(2, 0, 1, 0) == Approx(0.666667));
  }
  SECTION("Test P(F(2,1) = 1 | class = 0)") {
    REQUIRE(model.GetPixelProbability(2, 1, 1, 0) == Approx(0.666667));
  }
  SECTION("Test P(F(2,2) = 1 | class = 0)") {
    REQUIRE(model.GetPixelProbability(2, 2, 1, 0) == Approx(0.666667));
  }

  /*
   * Tests for unshaded pixels given class label '1'
   */
  SECTION("Test P(F(0,0) = 0 | class = 1)") {
    REQUIRE(model.GetPixelProbability(0, 0, 0, 1) == Approx(0.5));
  }
  SECTION("Test P(F(0,1) = 0 | class = 1)") {
    REQUIRE(model.GetPixelProbability(0, 1, 0, 1) == Approx(0.25));
  }
  SECTION("Test P(F(0,2) = 0 | class = 1)") {
    REQUIRE(model.GetPixelProbability(0, 2, 0, 1) == Approx(0.75));
  }
  SECTION("Test P(F(1,0) = 0 | class = 1)") {
    REQUIRE(model.GetPixelProbability(1, 0, 0, 1) == Approx(0.75));
  }
  SECTION("Test P(F(1,1) = 0 | class = 1)") {
    REQUIRE(model.GetPixelProbability(1, 1, 0, 1) == Approx(0.25));
  }
  SECTION("Test P(F(1,2) = 0 | class = 1)") {
    REQUIRE(model.GetPixelProbability(1, 2, 0, 1) == Approx(0.75));
  }
  SECTION("Test P(F(2,0) = 0 | class = 1)") {
    REQUIRE(model.GetPixelProbability(2, 0, 0, 1) == Approx(0.5));
  }
  SECTION("Test P(F(2,1) = 0 | class = 1)") {
    REQUIRE(model.GetPixelProbability(2, 1, 0, 1) == Approx(0.25));
  }
  SECTION("Test P(F(2,2) = 0 | class = 1)") {
    REQUIRE(model.GetPixelProbability(2, 2, 0, 1) == Approx(0.5));
  }

  /*
   * Tests for shaded pixels given class label '1'
   */
  SECTION("Test P(F(0,0) = 1 | class = 1)") {
    REQUIRE(model.GetPixelProbability(0, 0, 1, 1) == Approx(0.5));
  }
  SECTION("Test P(F(0,1) = 1 | class = 1)") {
    REQUIRE(model.GetPixelProbability(0, 1, 1, 1) == Approx(0.75));
  }
  SECTION("Test P(F(0,2) = 1 | class = 1)") {
    REQUIRE(model.GetPixelProbability(0, 2, 1, 1) == Approx(0.25));
  }
  SECTION("Test P(F(1,0) = 1 | class = 1)") {
    REQUIRE(model.GetPixelProbability(1, 0, 1, 1) == Approx(0.25));
  }
  SECTION("Test P(F(1,1) = 1 | class = 1)") {
    REQUIRE(model.GetPixelProbability(1, 1, 1, 1) == Approx(0.75));
  }
  SECTION("Test P(F(1,2) = 1 | class = 1)") {
    REQUIRE(model.GetPixelProbability(1, 2, 1, 1) == Approx(0.25));
  }
  SECTION("Test P(F(2,0) = 1 | class = 1)") {
    REQUIRE(model.GetPixelProbability(2, 0, 1, 1) == Approx(0.5));
  }
  SECTION("Test P(F(2,1) = 1 | class = 1)") {
    REQUIRE(model.GetPixelProbability(2, 1, 1, 1) == Approx(0.75));
  }
  SECTION("Test P(F(2,2) = 1 | class = 1)") {
    REQUIRE(model.GetPixelProbability(2, 2, 1, 1) == Approx(0.5));
  }
}

TEST_CASE("Model class can save and load model to/from file") {
  /* Create a model and save to a file path */
  NaiveBayesModel saved_model;
  InitializeAndTrainModel(saved_model);

  double laplace_k = 1.0;
  saved_model.Train(laplace_k);

  std::string save_file_path = "tests/data/test_model_save";
  std::ofstream save_file(save_file_path);
  save_file << saved_model;

  /* Create another model and load its state from the previously saved model */
  NaiveBayesModel loaded_model;
  std::string load_file_path = "tests/data/test_model_save";
  std::ifstream load_file(load_file_path);
  load_file >> loaded_model;

  /*
   * Test that both models match for P(class = 0)
   */
  SECTION("Test P(class = 0)") {
    REQUIRE(loaded_model.GetClassProbability(0) ==
            Approx(saved_model.GetClassProbability(0)));
  }
  SECTION("Test P(class = 1)") {
    REQUIRE(loaded_model.GetClassProbability(1) ==
            Approx(saved_model.GetClassProbability(1)));
  }

  /*
   * Test that both models match for P(F(row, col) = 0 | class = 0)
   */
  SECTION("Test P(F(0,0) = 0 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 0, 0, 0) ==
            Approx(saved_model.GetPixelProbability(0, 0, 0, 0)));
  }
  SECTION("Test P(F(0,1) = 0 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 1, 0, 0) ==
            Approx(saved_model.GetPixelProbability(0, 1, 0, 0)));
  }
  SECTION("Test P(F(0,2) = 0 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 2, 0, 0) ==
            Approx(saved_model.GetPixelProbability(0, 2, 0, 0)));
  }
  SECTION("Test P(F(1,0) = 0 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 0, 0, 0) ==
            Approx(saved_model.GetPixelProbability(1, 0, 0, 0)));
  }
  SECTION("Test P(F(1,1) = 0 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 1, 0, 0) ==
            Approx(saved_model.GetPixelProbability(1, 1, 0, 0)));
  }
  SECTION("Test P(F(1,2) = 0 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 2, 0, 0) ==
            Approx(saved_model.GetPixelProbability(1, 2, 0, 0)));
  }
  SECTION("Test P(F(2,0) = 0 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 0, 0, 0) ==
            Approx(saved_model.GetPixelProbability(2, 0, 0, 0)));
  }
  SECTION("Test P(F(2,1) = 0 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 1, 0, 0) ==
            Approx(saved_model.GetPixelProbability(2, 1, 0, 0)));
  }
  SECTION("Test P(F(2,2) = 0 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 2, 0, 0) ==
            Approx(saved_model.GetPixelProbability(2, 2, 0, 0)));
  }

  /*
   * Test that both models match for P(F(row, col) = 1 | class = 0)
   */
  SECTION("Test P(F(0,0) = 1 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 0, 1, 0) ==
            Approx(saved_model.GetPixelProbability(0, 0, 1, 0)));
  }
  SECTION("Test P(F(0,1) = 1 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 1, 1, 0) ==
            Approx(saved_model.GetPixelProbability(0, 1, 1, 0)));
  }
  SECTION("Test P(F(0,2) = 1 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 2, 1, 0) ==
            Approx(saved_model.GetPixelProbability(0, 2, 1, 0)));
  }
  SECTION("Test P(F(1,0) = 1 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 0, 1, 0) ==
            Approx(saved_model.GetPixelProbability(1, 0, 1, 0)));
  }
  SECTION("Test P(F(1,1) = 1 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 1, 1, 0) ==
            Approx(saved_model.GetPixelProbability(1, 1, 1, 0)));
  }
  SECTION("Test P(F(1,2) = 1 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 2, 1, 0) ==
            Approx(saved_model.GetPixelProbability(1, 2, 1, 0)));
  }
  SECTION("Test P(F(2,0) = 1 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 0, 1, 0) ==
            Approx(saved_model.GetPixelProbability(2, 0, 1, 0)));
  }
  SECTION("Test P(F(2,1) = 1 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 1, 1, 0) ==
            Approx(saved_model.GetPixelProbability(2, 1, 1, 0)));
  }
  SECTION("Test P(F(2,2) = 1 | class = 0)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 2, 1, 0) ==
            Approx(saved_model.GetPixelProbability(2, 2, 1, 0)));
  }

  /*
   * Test that both models match for P(F(row, col) = 0 | class = 1)
   */
  SECTION("Test P(F(0,0) = 0 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 0, 0, 1) ==
            Approx(saved_model.GetPixelProbability(0, 0, 0, 1)));
  }
  SECTION("Test P(F(0,1) = 0 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 1, 0, 1) ==
            Approx(saved_model.GetPixelProbability(0, 1, 0, 1)));
  }
  SECTION("Test P(F(0,2) = 0 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 2, 0, 1) ==
            Approx(saved_model.GetPixelProbability(0, 2, 0, 1)));
  }
  SECTION("Test P(F(1,0) = 0 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 0, 0, 1) ==
            Approx(saved_model.GetPixelProbability(1, 0, 0, 1)));
  }
  SECTION("Test P(F(1,1) = 0 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 1, 0, 1) ==
            Approx(saved_model.GetPixelProbability(1, 1, 0, 1)));
  }
  SECTION("Test P(F(1,2) = 0 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 2, 0, 1) ==
            Approx(saved_model.GetPixelProbability(1, 2, 0, 1)));
  }
  SECTION("Test P(F(2,0) = 0 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 0, 0, 1) ==
            Approx(saved_model.GetPixelProbability(2, 0, 0, 1)));
  }
  SECTION("Test P(F(2,1) = 0 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 1, 0, 1) ==
            Approx(saved_model.GetPixelProbability(2, 1, 0, 1)));
  }
  SECTION("Test P(F(2,2) = 0 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 2, 0, 1) ==
            Approx(saved_model.GetPixelProbability(2, 2, 0, 1)));
  }

  /*
   * Test that both models match for P(F(row, col) = 1 | class = 1)
   */
  SECTION("Test P(F(0,0) = 1 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 0, 1, 1) ==
            Approx(saved_model.GetPixelProbability(0, 0, 1, 1)));
  }
  SECTION("Test P(F(0,1) = 1 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 1, 1, 1) ==
            Approx(saved_model.GetPixelProbability(0, 1, 1, 1)));
  }
  SECTION("Test P(F(0,2) = 1 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(0, 2, 1, 1) ==
            Approx(saved_model.GetPixelProbability(0, 2, 1, 1)));
  }
  SECTION("Test P(F(1,0) = 1 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 0, 1, 1) ==
            Approx(saved_model.GetPixelProbability(1, 0, 1, 1)));
  }
  SECTION("Test P(F(1,1) = 1 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 1, 1, 1) ==
            Approx(saved_model.GetPixelProbability(1, 1, 1, 1)));
  }
  SECTION("Test P(F(1,2) = 1 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(1, 2, 1, 1) ==
            Approx(saved_model.GetPixelProbability(1, 2, 1, 1)));
  }
  SECTION("Test P(F(2,0) = 1 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 0, 1, 1) ==
            Approx(saved_model.GetPixelProbability(2, 0, 1, 1)));
  }
  SECTION("Test P(F(2,1) = 1 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 1, 1, 1) ==
            Approx(saved_model.GetPixelProbability(2, 1, 1, 1)));
  }
  SECTION("Test P(F(2,2) = 1 | class = 1)") {
    REQUIRE(loaded_model.GetPixelProbability(2, 2, 1, 1) ==
            Approx(saved_model.GetPixelProbability(2, 2, 1, 1)));
  }
}

TEST_CASE(
    "Model throws exception if file path DNE/blank rather than give undefined"
    "behavior") {
  SECTION("Valid data format but images/labels do not match") {
    ImageDataset image_dataset;
    LabelDataset label_dataset;

    /* Specify data streams for two different training sets */
    std::ifstream test_image_stream("tests/data/test_data_3x3/test_images");
    std::ifstream test_label_stream("data/mnistdatatraining/traininglabels");

    test_image_stream >> image_dataset;
    test_label_stream >> label_dataset;

    REQUIRE_THROWS_AS(NaiveBayesModel(image_dataset, label_dataset),
                      std::invalid_argument);
  }

  SECTION("Valid data format but file paths do not exist") {
    ImageDataset image_dataset;
    LabelDataset label_dataset;

    std::ifstream test_image_stream("random");
    std::ifstream test_label_stream("stuff");

    test_image_stream >> image_dataset;
    test_label_stream >> label_dataset;

    REQUIRE_THROWS_AS(NaiveBayesModel(image_dataset, label_dataset),
                      std::invalid_argument);
  }
}
