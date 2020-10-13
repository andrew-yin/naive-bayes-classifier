#include <core/naive_bayes_model.h>
#include <gflags/gflags.h>

#include <iostream>

bool DetermineIsTraining();
bool DetermineIsLoading();
bool DetermineIsSaving();
naivebayes::NaiveBayesModel GenerateModel(bool const &is_training,
                                          bool const &is_loading,
                                          bool const &is_saving);

DEFINE_string(training_image, "null",
              "The file path containing a training image dataset");
DEFINE_string(training_label, "null",
              "The file path containing a training label dataset");
DEFINE_string(save, "null", "The file path to save the model to");
DEFINE_string(load, "null", "The file path to load the model from");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  bool is_training = DetermineIsTraining();
  bool is_loading = DetermineIsLoading();
  bool is_saving = DetermineIsSaving();

  if (is_training && is_loading) {
    throw std::invalid_argument(
        "Error: Model cannot both be loaded from a file and trained with a "
        "dataset. Please try again.");
  }

  naivebayes::NaiveBayesModel model =
      GenerateModel(is_training, is_loading, is_saving);

  return 0;
}

naivebayes::NaiveBayesModel GenerateModel(bool const &is_training,
                                          bool const &is_loading,
                                          bool const &is_saving) {
  if (is_training) {
    naivebayes::ImageDataset image_dataset;
    naivebayes::LabelDataset label_dataset;

    std::ifstream image_stream(FLAGS_training_image);
    std::ifstream label_stream(FLAGS_training_label);
    image_stream >> image_dataset;
    label_stream >> label_dataset;

    naivebayes::NaiveBayesModel model(image_dataset, label_dataset);
    model.Train();

    if (is_saving) {
      std::ofstream save_file(FLAGS_save);
      save_file << model;
    }

    return model;

  } else if (is_loading) {
    naivebayes::NaiveBayesModel model;

    std::ifstream load_file(FLAGS_load);
    load_file >> model;

    if (is_saving) {
      std::ofstream save_file(FLAGS_save);
      save_file << model;
    }

    return model;

  } else {
    throw std::invalid_argument(
        "Error: Model cannot be generated. Input data not found.");
  }
}

bool DetermineIsSaving() {
  if (FLAGS_save == "null") {
    return false;
  } else {
    return true;
  }
}
bool DetermineIsLoading() {
  if (FLAGS_load == "null") {
    return false;
  } else {
    return true;
  }
}
bool DetermineIsTraining() {
  if (FLAGS_training_image == "null") {
    if (FLAGS_training_label != "null") {
      throw std::invalid_argument(
          "Error: Model must be trained using both images and labels, but you "
          "have only provided a path to labels. Please try again.");
    } else {
      return false;
    }
  } else {
    if (FLAGS_training_label == "null") {
      throw std::invalid_argument(
          "Error: Model must be trained using both images and labels, but you "
          "have only provided a path to images. Please try again.");
    } else {
      return true;
    }
  }
}
