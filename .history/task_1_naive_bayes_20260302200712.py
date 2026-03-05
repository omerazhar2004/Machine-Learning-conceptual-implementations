# Task 1 [15 points out of 40] Naïve Bayes classifier

# Your first task is to implement the naïve Bayes classifier on your own. This involves calculating all the necessary
# probabilities from the provided data, and using them to make predictions for new unseen records.
# The required version is the one covered in the module. Implementing other naïve Bayes approaches (e.g. Gaussian) or
# using further modifications that do not correspond to the technique practiced in the module will lead to
# significant loss of points.
# The template contains a range of functions you must implement and use appropriately for this task. The template also
# uses a range of functions implemented by the module leader to support you in this task.

import pandas as pd


class NaiveBayes:

    # This function simply initializes an instance of NaiveBayes class. The constructor takes at input:
    # - class_info - pair that contains the name of the class column and its permitted values
    # - feature_info - dictionary that states attribute names and their permitted values

    def __init__(self, class_info: tuple[str, list[str]], feature_info: dict[str, list[str]]):
        self.class_info = class_info
        self.feature_info = feature_info
        self.prior_probabilities = {}
        self.conditional_probabilities = {}
        # You can add further variables/attributes/etc. here

    # This function trains the model, aka calculates all the necessary probabilities that a naive Bayes model needs.
    # How you store the computed probabilities internally is up to you - you may want to extend the init function.
    # For the purpose of this task, numerical values are treated just like categorical ones. Any new training
    # should purge old data.
    # At input, train_model takes:
    # - training_data - a pandas DataFrame that contains all the attribute values and class value for a given entry

    def train_model(self, training_data: pd.DataFrame):
        self.prior_probabilities = {}
        self.conditional_probabilities = {}
        self.trained = False
        for class_value in self.class_info[1]:
            class_data = training_data[training_data[self.class_info[0]] == class_value]
            class_count = len(class_data)
            for feature_name in self.feature_info[0]:
                feature_counts = class_data[feature_name].value_counts().to_dict()
                for feature_value in self.feature_info[feature_name]:
                    count = feature_counts.get(feature_value, 0)
                    self.conditional_probabilities[(class_value, feature_name, feature_value)] = count / class_count if class_count > 0 else 0
            self.prior_probabilities[class_value] = class_count / len(training_data) if len(training_data) > 0 else 0
        self.trained = True

                                
        print("Do something!")

    # This function predicts the classes for entries in the training_data and produces an extended data frame.
    # At input, it takes:
    # - training_data - a pandas DataFrame that contains all the attribute values and class value for a given entry
    # The function outputs:
    # classified_data - a pandas DataFrame which expands the training_data by adding the "PredictedClass" column
    #                   that for every entry states the class value predicted for that entry. In case of ties,
    #                   the chosen class is the one that appears earlier alphabetically.
    def predict(self, testing_data: pd.DataFrame) -> pd.DataFrame:
        classified_data = None
        predictions = []
        df_copy = testing_data.copy()
        for _, row in df_copy.iterrows():
            class_scores = {}
            for c in self.class_info[1]:
                score = self.retrieve_class_probability(c)
                for feature_name in self.feature_info:
                    value = row[feature_name]
                    cond_prob = self.retrieve_conditional_probability(c, feature_name, value)
                    score = score * cond_prob
                class_scores[c] = score
            chosen_class = sorted(class_scores.items(), key=lambda x:(-x[1], x[0]))[0][0]  
            predictions.append(chosen_class)
        df_copy["PredictedClass"] = predictions                           
        classified_data = df_copy
        return classified_data

    # The function returns the probability of a given class value. You can assume
    # that this function simply retrieves the desired probability after training rather than
    # recomputes them from scratch. A value of 0 should be returned if no training took place.
    # At input, it takes:
    # - class_value - the class value for which we want to calculate the probability
    # The function outputs:
    # - probability - float representing the probability of the given class value
    def retrieve_class_probability(self, class_value: str) -> float:
        if class_value in self.prior_probabilities:
            class_prob = self.prior_probabilities.get(class_value)
        else:
            class_prob = 0
        return class_prob

    # The function returns the conditional probably of a feature value assuming a given class value. You can assume
    # that this function simply retrieves the desired probability after training rather than
    # recomputes them from scratch. A value of 0 should be returned if no training took place.
    # At input, it takes:
    # - class_value - the class value on which the feature_value is conditional
    # - feature_name - the name of the feature we want to calculate for
    # - feature_value - the feature value we want to calculate the conditional probability for
    # The function outputs:
    # - probability - float representing the calculated conditional probability
    #
    def retrieve_conditional_probability(self, class_value: str, feature_name: str, feature_value: str) -> float:
        if class_value and feature_name and feature_value in self.conditional_probabilities:
            cond_prob = self.conditional_probabilities[(class_value, feature_name, feature_value)]
        else:
            cond_prob = 0 float
        return cond_prob    

        return -1
