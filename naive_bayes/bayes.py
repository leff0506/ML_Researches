# categorical
import numpy as np
import sklearn


class CatBayes(sklearn.base.BaseEstimator):
    # y,X contains classes from 0 to n
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)
        # print(X)
        # print(y)
        self.number_classes = len(np.unique(y))
        self.number_features = len(X[0])
        self.number_examples = len(y)
        self.numbers_of_each_class = []
        # C[i] - probabilty of class C[i]
        self.C = [0]*self.number_classes

        for one_class in range(self.number_classes):
            self.numbers_of_each_class.append((y == one_class).sum())
            self.C[one_class] = self.numbers_of_each_class[one_class]/self.number_examples

        # number of unique classes of each feature
        self.uniques = []
        # probabilities[i][j][c] = P(f_ij|C_c)
        self.probabilities = []
        for feature_number in range(self.number_features):
            column = X[:,feature_number]
            uniques = np.unique(column)
            self.probabilities.append([[0]*self.number_classes for i in range(len(uniques))])
            self.uniques.append(len(uniques))
        # calculate P(feature_i_j) i - number of feature j - class of feature
        for c in range(self.number_examples):
            for i in range(self.number_features):
                self.probabilities[i][X[c][i]][y[c]]+=1

        for i in range(self.number_features):
            for j in range(self.uniques[i]):
                for c in range(self.number_classes):
                    self.probabilities[i][j][c] = (self.probabilities[i][j][c] +1)/(self.numbers_of_each_class[c]+self.uniques[i])
        # print(self.probabilities)
    def predict_proba(self,X):
        X = np.array(X)
        result = []
        for example_number in range(len(X)):
            probabilities = []
            for class_number in range(self.number_classes):
                p = 1
                for feature_number in range(self.number_features):
                    if X[example_number][feature_number] >= self.uniques[feature_number]:
                        p=0
                        break
                    p*=self.probabilities[feature_number][X[example_number][feature_number]][class_number]
                probabilities.append(p*self.C[class_number])
            result.append(self.__softmax(probabilities))
        return result
    def predict(self,X):
        prob = self.predict_proba(X)
        result = np.argmax(prob,axis=1)
        return result

    def __softmax(self,x):
        x = np.array(x)
        x = np.exp(x)/np.sum(np.exp(x),axis = 0)
        return x
class GaussianDistribution:
    def fit(self,x):
        x = np.array(x)
        self.mean = x.mean()
        self.std = x.std()
    def get_probability(self,x):
        result = 1/(self.std*np.sqrt(2*np.pi))*np.exp(-((x-self.mean)**2)/(2*self.std**2))
        return result

class UniformDistribution:
    def fit(self,x):
        pass
    def get_probability(self,x):
        return 1
class ExponentialDstribution:
    def fit(self,x):
        x = np.array(x)
        self.l = 1/x.mean()
    def get_probability(self,x):
        result = self.l * np.exp(-self.l*x)
        return result

class MixedBayes(sklearn.base.BaseEstimator):
    categorical_distribution = "categorical"
    gaussian_distribution = "gaussian"
    uniform_distribution = "uniform"
    exponential_distribution = "exponential"
    def __init__(self,types):

        self.distribution_functions = {self.gaussian_distribution:GaussianDistribution,self.uniform_distribution:UniformDistribution,self.exponential_distribution:ExponentialDstribution}
        self.types = types
    # y,X contains classes from 0 to n types - list of string where each string define the feature categorical or numeral "categorical", "gaussian" (добавить показательное биномиальное равномерное и еще какие-то)
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)
        # print(X)
        # print(y)
        self.number_classes = len(np.unique(y))
        self.number_features = len(X[0])
        self.number_examples = len(y)
        self.numbers_of_each_class = []
        self.distributions = []


        # C[i] - probabilty of class C[i]
        self.C = [0]*self.number_classes

        for one_class in range(self.number_classes):
            self.numbers_of_each_class.append((y == one_class).sum())
            self.C[one_class] = self.numbers_of_each_class[one_class]/self.number_examples

        # number of unique classes of each feature
        self.uniques = []
        # probabilities[i][j][c] = P(f_ij|C_c)
        self.probabilities = []
        for feature_number in range(self.number_features):
            column = X[:,feature_number]
            uniques = np.unique(column)
            self.probabilities.append([[0]*self.number_classes for i in range(len(uniques))])
            self.uniques.append(len(uniques))
            self.distributions.append([None for i in range(self.number_classes)])

        # calculate P(feature_i_j) i - number of feature j - class of feature
        for c in range(self.number_examples):
            for i in range(self.number_features):
                if self.types[i] == self.categorical_distribution:
                    self.probabilities[i][X[c][i]][y[c]]+=1

        for i in range(self.number_features):
            if self.types[i] == self.categorical_distribution:
                for j in range(self.uniques[i]):
                    for c in range(self.number_classes):
                        self.probabilities[i][j][c] = (self.probabilities[i][j][c] +1)/(self.numbers_of_each_class[c]+self.uniques[i])
        for feature in range(self.number_features):
            if self.types[feature] == self.categorical_distribution:
                continue
            distributions = []
            for class_number in range(self.number_classes):
                y_mask = y == class_number
                column = X[y_mask,feature]
                gaussian = self.distribution_functions[self.types[feature]]()
                gaussian.fit(column)
                distributions.append(gaussian)
            self.distributions[feature] = distributions
        # print(self.probabilities)
    def predict_proba(self,X):
        X = np.array(X)
        result = []
        for example_number in range(len(X)):
            probabilities = []
            for class_number in range(self.number_classes):
                p = 1
                for feature_number in range(self.number_features):
                    if self.types[feature_number] == self.categorical_distribution:
                        if X[example_number][feature_number] >= self.uniques[feature_number]:
                            p=0
                            break
                        p*=self.probabilities[feature_number][X[example_number][feature_number]][class_number]
                    else:
                        p *= self.distributions[feature_number][class_number].get_probability(X[example_number][feature_number])
                probabilities.append(p*self.C[class_number])
            result.append(self.__softmax(probabilities))
        return result
    def predict(self,X):
        prob = self.predict_proba(X)
        result = np.argmax(prob,axis=1)
        return result

    def __softmax(self,x):
        x = np.array(x)
        x = np.exp(x)/np.sum(np.exp(x),axis = 0)
        return x