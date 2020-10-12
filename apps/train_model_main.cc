#include <core/naive_bayes_trainer.h>
#include <iostream>

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  // TODO: Replace this with code that reads the training data, trains a model,
  // and saves the trained model to a file.
  bool data_loaded_from_file = true;
  bool data_save_to_file = false;

  if (data_loaded_from_file) {
    naivebayes::NaiveBayesTrainer trainer;

    std::string load_file_path = "../data/test_save";
    std::ifstream load_file(load_file_path);

    load_file >> trainer;
  }
  else {
    naivebayes::ImageDataset image_dataset;
    naivebayes::LabelDataset label_dataset;

    std::ifstream training_images("../data/test_data/test_images");
    std::ifstream training_labels("../data/test_data/test_labels");

    if (training_images.is_open()) {
      training_images >> image_dataset;
      training_images.close();
    }
    if (training_labels.is_open()) {
      training_labels >> label_dataset;
      training_labels.close();
    }

    naivebayes::NaiveBayesTrainer trainer(image_dataset, label_dataset);
    trainer.Train();

    if (data_save_to_file) {
      std::string save_file_path = "../data/test_save";
      std::ofstream save_file(save_file_path);

      save_file << trainer;
    }
  }

  return 0;
}
