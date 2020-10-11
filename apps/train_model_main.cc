#include <core/naive_bayes_trainer.h>
#include <fstream>
#include <iostream>

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  // TODO: Replace this with code that reads the training data, trains a model,
  // and saves the trained model to a file.

  naivebayes::ImageDataset image_dataset;
  naivebayes::LabelDataset label_dataset;

  std::ifstream training_images ("../data/mnistdatatraining/trainingimages");
  std::ifstream training_labels ("../data/mnistdatatraining/traininglabels");

  training_images >> image_dataset;
  training_labels >> label_dataset;

  naivebayes::NaiveBayesTrainer trainer(image_dataset, label_dataset);
  std::cout << "size: " << trainer.GetImageDatasetSize() << std::endl;
  return 0;
}
