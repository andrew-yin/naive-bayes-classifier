#include <string>
#include <istream>
#include <core/training_image.h>

namespace naivebayes {

class NaiveBayesTrainer {
 public:
  NaiveBayesTrainer();
  friend std::istream &operator>>(std::istream &in, NaiveBayesTrainer &trainer);
  void AddTrainingImage(TrainingImage image);

 private:
  std::vector<TrainingImage> training_images_;
};

}  // namespace naivebayes