#include <core/naive_bayes_trainer.h>
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

  return 0;
}
