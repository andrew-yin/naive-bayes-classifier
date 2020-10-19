#include <core/naive_bayes_classifier.h>
#include <core/naive_bayes_model.h>
#include <gflags/gflags.h>

#include <iostream>

DEFINE_string(training_images, "null",
              "The file path containing a training image dataset");
DEFINE_string(training_labels, "null",
              "The file path containing a training label dataset");
DEFINE_string(save, "null", "The file path to save the model to");
DEFINE_string(load, "null", "The file path to load the model from");
DEFINE_string(test_images, "null",
              "The file path containing images to test the model's accuracy");
DEFINE_string(test_labels, "null",
              "The file path containing labels to test the model's accuracy");

/** Determines if the model needs to be trained based on CLI args */
bool DetermineIsTraining();

/* Create a model based on whether it needs to be trained/loaded/saved */
naivebayes::NaiveBayesModel GenerateModel(bool training, bool loading,
                                          bool saving);

bool DetermineIsTesting();
int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  /* True if the model should be trained from datasets, false otherwise */
  bool is_training = DetermineIsTraining();
  /* True if the model should be loaded from file, false otherwise */
  bool is_loading = FLAGS_load != "null";
  /* True if the model should save it state to a file, false otherwise */
  bool is_saving = FLAGS_save != "null";
  bool is_testing = DetermineIsTesting();

  if (is_training && is_loading) {
    throw std::invalid_argument(
        "Model cannot be both loaded from a file and trained with a "
        "dataset. Please try again.");
  }

  if (is_training && !is_saving) {
    std::cerr << "Warning: You are training a model from a dataset but are not "
                 "saving it."
              << std::endl;
  }

  naivebayes::NaiveBayesModel model =
      GenerateModel(is_training, is_loading, is_saving);

  if (is_testing) {
    std::cout << "Testing model..." << std::endl;
    naivebayes::ImageDataset test_images;
    naivebayes::LabelDataset test_labels;

    std::ifstream image_stream(FLAGS_test_images);
    std::ifstream label_stream(FLAGS_test_labels);
    image_stream >> test_images;
    label_stream >> test_labels;

    size_t image_dataset_size = test_images.images_.size();
    size_t label_dataset_size = test_labels.labels_.size();
    if (image_dataset_size != label_dataset_size) {
      throw std::invalid_argument(
          "Test image and label datasets given do not align. Please try "
          "again.");
    } else if (image_dataset_size == 0 || label_dataset_size == 0) {
      throw std::invalid_argument(
          "Test files specified are either blank or DNE."
          "Please try again.");
    }

    size_t num_correct = 0;
    for (size_t i = 0; i < test_images.images_.size(); i++) {
      naivebayes::NaiveBayesClassifier classifier(model,
                                                  test_images.images_[i]);
      if (classifier.Classify() == test_labels.labels_[i]) {
        num_correct++;
      }
    }

    double accuracy = (double)num_correct / test_images.images_.size() * 100.0;
    std::cout << "Your model determined the digit correctly " << accuracy
              << "% of the time." << std::endl;
  }

  return 0;
}

naivebayes::NaiveBayesModel GenerateModel(bool is_training, bool is_loading,
                                          bool is_saving) {
  if (is_training) {
    std::cout << "Loading training images from \"" << FLAGS_training_images
              << "\"..." << std::endl;
    naivebayes::ImageDataset image_dataset;
    std::ifstream image_stream(FLAGS_training_images);
    image_stream >> image_dataset;

    std::cout << "Loading training labels from \"" << FLAGS_training_labels
              << "\"..." << std::endl;
    naivebayes::LabelDataset label_dataset;
    std::ifstream label_stream(FLAGS_training_labels);
    label_stream >> label_dataset;

    naivebayes::NaiveBayesModel model(image_dataset, label_dataset);
    std::cout << "Loaded training dataset successfully." << std::endl;

    std::cout << "Training model..." << std::endl;
    double laplace_k = 1.0;
    model.Train(laplace_k);
    std::cout << "Training completed successfully." << std::endl;

    if (is_saving) {
      std::cout << "Saving model to \"" << FLAGS_save << "\"..." << std::endl;
      std::ofstream save_file(FLAGS_save);
      save_file << model;
      std::cout << "Model saved successfully." << std::endl;
    }

    return model;
  } else if (is_loading) {
    std::cout << "Loading model from \"" << FLAGS_load << "\"..." << std::endl;
    naivebayes::NaiveBayesModel model;
    std::ifstream load_file(FLAGS_load);
    load_file >> model;
    std::cout << "Model loaded successfully." << std::endl;

    if (is_saving) {
      std::cout << "Saving model to \"" << FLAGS_save << "\"..." << std::endl;
      std::ofstream save_file(FLAGS_save);
      save_file << model;
      std::cout << "Model saved successfully." << std::endl;
    }

    return model;
  } else {
    throw std::invalid_argument(
        "Model cannot be generated. Input data not specified.");
  }
}

bool DetermineIsTraining() {
  if (FLAGS_training_images == "null") {
    if (FLAGS_training_labels != "null") {
      throw std::invalid_argument(
          "Model must be trained using both images and labels, but you "
          "have only provided a path to labels. Please try again.");
    } else {
      return false;
    }
  } else {
    if (FLAGS_training_labels == "null") {
      throw std::invalid_argument(
          "Model must be trained using both images and labels, but you "
          "have only provided a path to images. Please try again.");
    } else {
      return true;
    }
  }
}

bool DetermineIsTesting() {
  if (FLAGS_test_images == "null") {
    if (FLAGS_test_labels != "null") {
      throw std::invalid_argument(
          "Model must be tested using both images and labels, but you "
          "have only provided a path to labels. Please try again.");
    } else {
      return false;
    }
  } else {
    if (FLAGS_test_labels == "null") {
      throw std::invalid_argument(
          "Model must be tested using both images and labels, but you "
          "have only provided a path to images. Please try again.");
    } else {
      return true;
    }
  }
}