#include <core/naive_bayes_model.h>
#include <gflags/gflags.h>

DEFINE_string(training_image, "null",
              "The file path containing a training image dataset");
DEFINE_string(training_label, "null",
              "The file path containing a training label dataset");
DEFINE_string(save, "null", "The file path to save the model to");
DEFINE_string(load, "null", "The file path to load the model from");

/** Determines if the model needs to be trained based on CLI args */
bool DetermineIsTraining();
/** Determines if the model needs to be loaded from file based on CLI args */
bool DetermineIsLoading();
/** Determines if the model needs to be saved to a file based on CLI args */
bool DetermineIsSaving();

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /* True if the model should be trained from datasets, false otherwise */
  bool is_training = DetermineIsTraining();
  /* True if the model should be loaded from file, false otherwise */
  bool is_loading = DetermineIsLoading();
  /* True if the model should save it state to a file, false otherwise */
  bool is_saving = DetermineIsSaving();

  if (is_training && is_loading) {
    throw std::invalid_argument(
        "Model cannot be both loaded from a file and trained with a "
        "dataset. Please try again.");
  }

  /* Create a model based on whether it needs to be trained/loaded/saved */
  if (is_training) {
    naivebayes::ImageDataset image_dataset;
    naivebayes::LabelDataset label_dataset;

    std::ifstream image_stream(FLAGS_training_image);
    std::ifstream label_stream(FLAGS_training_label);
    image_stream >> image_dataset;
    label_stream >> label_dataset;

    size_t laplace_k = 1.0;
    naivebayes::NaiveBayesModel model(image_dataset, label_dataset);
    model.Train(laplace_k);

    if (is_saving) {
      std::ofstream save_file(FLAGS_save);
      save_file << model;
    }

  } else if (is_loading) {
    naivebayes::NaiveBayesModel model;

    std::ifstream load_file(FLAGS_load);
    load_file >> model;

    if (is_saving) {
      std::ofstream save_file(FLAGS_save);
      save_file << model;
    }

  } else {
    throw std::invalid_argument(
        "Model cannot be generated. Input data not specified.");
  }
  return 0;
}

bool DetermineIsTraining() {
  if (FLAGS_training_image == "null") {
    if (FLAGS_training_label != "null") {
      throw std::invalid_argument(
          "Model must be trained using both images and labels, but you "
          "have only provided a path to labels. Please try again.");
    } else {
      return false;
    }
  } else {
    if (FLAGS_training_label == "null") {
      throw std::invalid_argument(
          "Model must be trained using both images and labels, but you "
          "have only provided a path to images. Please try again.");
    } else {
      return true;
    }
  }
}

bool DetermineIsLoading() {
  if (FLAGS_load == "null") {
    return false;
  } else {
    return true;
  }
}

bool DetermineIsSaving() {
  if (FLAGS_save == "null") {
    return false;
  } else {
    return true;
  }
}