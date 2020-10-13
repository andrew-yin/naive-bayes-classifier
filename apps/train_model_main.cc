#include <core/naive_bayes_model.h>
#include <gflags/gflags.h>

#include <iostream>

DEFINE_string(training_image, "null",
              "The file path containing a training image dataset");
DEFINE_string(training_label, "null",
              "The file path containing a training label dataset");
DEFINE_string(save, "null", "The file path to save the model to");
DEFINE_string(load, "null", "The file path to load the model from");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  bool train_from_files;
  bool load_from_file;
  bool save_to_file;

  if (FLAGS_training_image == "null") {
    if (FLAGS_training_label != "null") {
      throw std::invalid_argument(
          "Error: Model must be trained using both images and labels, but you "
          "have only provided a path to labels. Please try again.");
    } else {
      train_from_files = false;
    }
  } else {
    if (FLAGS_training_label == "null") {
      throw std::invalid_argument(
          "Error: Model must be trained using both images and labels, but you "
          "have only provided a path to images. Please try again.");
    } else {
      train_from_files = true;
    }
  }

  if (FLAGS_load == "null") {
    load_from_file = false;
  } else {
    load_from_file = true;
  }

  if (FLAGS_save == "null") {
    save_to_file = false;
  } else {
    save_to_file = true;
  }

  if (train_from_files && load_from_file) {
    throw std::invalid_argument(
        "Error: Model cannot both be loaded from a file and trained with a "
        "dataset. Please try again.");
  }

  if (train_from_files) {
    naivebayes::ImageDataset image_dataset;
    naivebayes::LabelDataset label_dataset;

    std::ifstream image_stream(FLAGS_training_image);
    std::ifstream label_stream(FLAGS_training_label);
    image_stream >> image_dataset;
    label_stream >> label_dataset;

    naivebayes::NaiveBayesModel model(image_dataset, label_dataset);
    model.Train();

    if (save_to_file) {
      std::ofstream save_file(FLAGS_save);
      save_file << model;
    }
  } else if (load_from_file) {
    naivebayes::NaiveBayesModel model;

    std::ifstream load_file(FLAGS_load);
    load_file >> model;

    if (save_to_file) {
      std::ofstream save_file(FLAGS_save);
      save_file << model;
    }
  }

  return 0;
}
